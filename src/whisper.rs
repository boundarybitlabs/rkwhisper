use crate::beam::BeamSearch;
use crate::decoder::{WhisperDecoder, WhisperDecoderState};
use crate::encoder::{EncKvModel, WhisperEncoder};
use crate::spec::WhisperSpec;
use crate::{MelSpectrogram, N_SAMPLES, load_audio_file};
use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;

/// Transcribe arbitrary-length WAV with chunked 30-second encoder windows.
pub fn transcribe<S: WhisperSpec>(
    audio_path: &str,
    tokenizer_path: &str,
    mel_spec: &MelSpectrogram,
    encoder: &WhisperEncoder<S>,
    enc_kv: &EncKvModel<S>,
    decoder: &mut WhisperDecoder<S>,
    lang: &str,
    task: &str,
    notimestamps: bool,
    max_new_tokens: usize,
    beam_size: usize,
) -> Result<String> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

    let audio = load_audio_file(audio_path)?;
    let mut full_text = String::new();

    let mut state = WhisperDecoderState::new::<S>();

    for (chunk_idx, wave) in audio.chunks(N_SAMPLES).enumerate() {
        let mels = mel_spec.log_mel_spectrogram(wave)?;

        let encoded = encoder
            .encode(&mels)
            .map_err(|e| anyhow!("encoder failed on chunk {chunk_idx}: {e}"))?;

        let (enc_k, enc_v) = enc_kv
            .compute(&encoded)
            .map_err(|e| anyhow!("enc-kv failed on chunk {chunk_idx}: {e}"))?;

        decoder.set_enc_kv(enc_k, enc_v);

        let prompt = control_prompt(&tokenizer, lang, task, notimestamps)?;

        state.reset();
        let mut last_logits = vec![0f32; S::VOCAB];
        for (i, &id) in prompt.iter().enumerate() {
            last_logits = decoder
                .step(&mut state, id)
                .map_err(|e| anyhow!("prime step {i} failed on chunk {chunk_idx}: {e}"))?;
        }

        let tok_eot = S::TOKEN_EOT;
        let tok_sot = tokenizer.token_to_id("<|startoftranscript|>").unwrap();
        let tok_notimestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap();
        let prompt_len = prompt.len();

        let suppress_fn = |gen_len: usize, logits: &mut [f32]| {
            // Suppress EOT on the first generated token
            // In beam search, gen_len includes the prompt
            if gen_len <= prompt_len {
                logits[tok_eot as usize] = -1e4;
            }

            // Suppress SOT and control tokens (SOT..=NoTimestamps)
            for i in (tok_sot as usize)..=(tok_notimestamps as usize) {
                if i < logits.len() {
                    logits[i] = -1e4;
                }
            }

            // Suppress timestamps (everything after NoTimestamps)
            for i in (tok_notimestamps as usize + 1)..logits.len() {
                logits[i] = -1e4;
            }
        };

        let mut generated: Vec<u32>;

        if beam_size > 1 {
            let mut beam_search =
                BeamSearch::<S>::new(beam_size, state.clone(), last_logits, prompt.clone(), 0.6);
            for _ in 0..max_new_tokens {
                beam_search.step(decoder, &suppress_fn)?;
                if beam_search.beams.is_empty() {
                    break;
                }
            }
            generated = beam_search.best_result().unwrap_or_default();
        } else {
            generated = Vec::new();
            let mut current_logits = last_logits;

            for _step in 0..max_new_tokens {
                let mut logits_1d = current_logits.clone();
                suppress_fn(prompt_len + generated.len(), &mut logits_1d);

                let token_id = argmax_token(&logits_1d);
                if token_id == tok_eot {
                    break;
                }

                generated.push(token_id);

                if generated.len() >= 8 {
                    let tail = &generated[generated.len() - 8..];
                    if tail[0..4] == tail[4..8] {
                        break;
                    }
                }

                current_logits = decoder
                    .step(&mut state, token_id)
                    .map_err(|e| anyhow!("decoder step failed on chunk {chunk_idx}: {e}"))?;
            }
        }

        // Filter out the prompt from the best result if using beam search
        let output_tokens = if beam_size > 1 {
            generated.get(prompt.len()..).unwrap_or(&[])
        } else {
            &generated
        };

        state.reset();
        full_text.push_str(&tokenizer.decode(output_tokens, true).unwrap());
        full_text.push(' ');
    }

    Ok(full_text.trim().to_string())
}

/// Build control prompt: <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]
fn control_prompt(
    tok: &Tokenizer,
    lang: &str, // bare language code, e.g. "en"
    task: &str, // "transcribe" or "translate"
    notimestamps: bool,
) -> Result<Vec<u32>> {
    let start = tok.token_to_id("<|startoftranscript|>").unwrap();
    let lang = tok.token_to_id(&format!("<|{lang}|>")).unwrap();
    let task = tok.token_to_id(&format!("<|{task}|>")).unwrap();
    let nots = tok.token_to_id("<|notimestamps|>").unwrap();
    let mut prompt = vec![start, lang, task];
    if notimestamps {
        prompt.push(nots);
    }
    Ok(prompt)
}

/// Simple greedy argmax over logits.
#[inline]
fn argmax_token(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}
