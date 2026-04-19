use crate::decoder::WhisperDecoder;
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
) -> Result<String> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

    let audio = load_audio_file(audio_path)?;
    let mut full_text = String::new();

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

        let mut last_logits = vec![0f32; S::VOCAB];
        for (i, &id) in prompt.iter().enumerate() {
            last_logits = decoder
                .step(id)
                .map_err(|e| anyhow!("prime step {i} failed on chunk {chunk_idx}: {e}"))?;
        }

        let tok_eot = S::TOKEN_EOT;
        let tok_sot = tokenizer.token_to_id("<|startoftranscript|>").unwrap();
        let tok_notimestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap();

        let mut tokens = prompt;
        let mut generated: Vec<u32> = Vec::new();

        for _step in 0..max_new_tokens {
            let mut logits_1d = last_logits.clone();

            // Suppress EOT on the first generated token
            if generated.is_empty() {
                logits_1d[tok_eot as usize] = -1e4;
            }

            // Suppress SOT and control tokens (SOT..=NoTimestamps)
            for i in (tok_sot as usize)..=(tok_notimestamps as usize) {
                if i < logits_1d.len() {
                    logits_1d[i] = -1e4;
                }
            }

            // Suppress timestamps (everything after NoTimestamps)
            for i in (tok_notimestamps as usize + 1)..logits_1d.len() {
                logits_1d[i] = -1e4;
            }

            let token_id = argmax_token(&logits_1d);

            if token_id == tok_eot {
                break;
            }

            generated.push(token_id);
            tokens.push(token_id);

            // Break repetition loops (tail of 8, 4+4)
            if generated.len() >= 8 {
                let tail = &generated[generated.len() - 8..];
                if tail[0..4] == tail[4..8] {
                    break;
                }
            }

            last_logits = decoder
                .step(token_id)
                .map_err(|e| anyhow!("decoder step failed on chunk {chunk_idx}: {e}"))?;
        }

        decoder.reset();
        full_text.push_str(&tokenizer.decode(&generated, true).unwrap());
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
