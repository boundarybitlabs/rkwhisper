use crate::beam::BeamSearch;
use crate::decoder::{WhisperDecoder, WhisperDecoderState};
use crate::encoder::{EncKvModel, WhisperEncoder};
use crate::spec::WhisperSpec;
use crate::suppression::{SuppressTokens, TokenSuppressor};
use crate::vad::{VadModel, VadSegment, samples_to_sec};
use crate::{MelSpectrogram, N_SAMPLES, load_audio_file};
use anyhow::{Result, anyhow};
use serde::Serialize;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    pub lang: String,
    pub task: String,
    pub notimestamps: bool,
    pub max_new_tokens: usize,
    pub beam_size: usize,
    pub suppress_tokens: SuppressTokens,
}

impl TranscribeOptions {
    pub fn new(
        lang: impl Into<String>,
        task: impl Into<String>,
        notimestamps: bool,
        max_new_tokens: usize,
        beam_size: usize,
        suppress_tokens: SuppressTokens,
    ) -> Self {
        Self {
            lang: lang.into(),
            task: task.into(),
            notimestamps,
            max_new_tokens,
            beam_size,
            suppress_tokens,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub vad_segments: Vec<VadSegment>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TranscriptSegment {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub window_index: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct AudioWindow {
    pub(crate) index: usize,
    pub(crate) start_sample: usize,
    pub(crate) end_sample: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct WindowTranscription {
    pub(crate) window_index: usize,
    pub(crate) text: String,
    pub(crate) segments: Vec<TranscriptSegment>,
}

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
    let options = TranscribeOptions::new(
        lang,
        task,
        notimestamps,
        max_new_tokens,
        beam_size,
        SuppressTokens::Default,
    );
    Ok(transcribe_with_options(
        audio_path,
        tokenizer_path,
        mel_spec,
        encoder,
        enc_kv,
        decoder,
        None,
        &options,
    )?
    .text)
}

pub fn transcribe_with_options<S: WhisperSpec>(
    audio_path: &str,
    tokenizer_path: &str,
    mel_spec: &MelSpectrogram,
    encoder: &WhisperEncoder<S>,
    enc_kv: &EncKvModel<S>,
    decoder: &mut WhisperDecoder<S>,
    vad: Option<&VadModel>,
    options: &TranscribeOptions,
) -> Result<Transcription> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
    let audio = load_audio_file(audio_path)?;
    transcribe_audio_with_options(
        &audio, &tokenizer, mel_spec, encoder, enc_kv, decoder, vad, options,
    )
}

pub fn transcribe_audio_with_options<S: WhisperSpec>(
    audio: &[f32],
    tokenizer: &Tokenizer,
    mel_spec: &MelSpectrogram,
    encoder: &WhisperEncoder<S>,
    enc_kv: &EncKvModel<S>,
    decoder: &mut WhisperDecoder<S>,
    vad: Option<&VadModel>,
    options: &TranscribeOptions,
) -> Result<Transcription> {
    let vad_enabled = vad.is_some();
    let vad_segments = if let Some(vad) = vad {
        vad.segments(audio)?
    } else {
        Vec::new()
    };
    let windows = if vad_enabled {
        vad_audio_windows(&vad_segments)
    } else {
        fixed_audio_windows(audio.len())
    };
    let mut full_text = String::new();
    let mut segments = Vec::new();

    for window in &windows {
        let result = transcribe_audio_window(
            audio, tokenizer, mel_spec, encoder, enc_kv, decoder, window, options,
        )?;
        full_text.push_str(&result.text);
        full_text.push(' ');
        segments.extend(result.segments);
    }

    Ok(Transcription {
        text: full_text.trim().to_string(),
        segments,
        vad_segments,
    })
}

pub(crate) fn transcription_windows(
    audio_len: usize,
    vad_segments: &[VadSegment],
) -> Vec<AudioWindow> {
    if vad_segments.is_empty() {
        fixed_audio_windows(audio_len)
    } else {
        vad_audio_windows(vad_segments)
    }
}

pub(crate) fn transcribe_audio_window<S: WhisperSpec>(
    audio: &[f32],
    tokenizer: &Tokenizer,
    mel_spec: &MelSpectrogram,
    encoder: &WhisperEncoder<S>,
    enc_kv: &EncKvModel<S>,
    decoder: &mut WhisperDecoder<S>,
    window: &AudioWindow,
    options: &TranscribeOptions,
) -> Result<WindowTranscription> {
    let wave = &audio[window.start_sample..window.end_sample];
    transcribe_window_samples(
        wave,
        tokenizer,
        mel_spec,
        encoder,
        enc_kv,
        decoder,
        window.index,
        window.start_sample,
        window.end_sample,
        options,
    )
}

pub(crate) fn transcribe_window_samples<S: WhisperSpec>(
    wave: &[f32],
    tokenizer: &Tokenizer,
    mel_spec: &MelSpectrogram,
    encoder: &WhisperEncoder<S>,
    enc_kv: &EncKvModel<S>,
    decoder: &mut WhisperDecoder<S>,
    window_index: usize,
    window_start_sample: usize,
    window_end_sample: usize,
    options: &TranscribeOptions,
) -> Result<WindowTranscription> {
    let mut state = WhisperDecoderState::new::<S>();
    let mels = mel_spec.log_mel_spectrogram(wave)?;

    let encoded = encoder
        .encode(&mels)
        .map_err(|e| anyhow!("encoder failed on window {window_index}: {e}"))?;

    let (enc_k, enc_v) = enc_kv
        .compute(&encoded)
        .map_err(|e| anyhow!("enc-kv failed on window {window_index}: {e}"))?;

    decoder.set_enc_kv(enc_k, enc_v);

    let prompt = control_prompt(
        tokenizer,
        &options.lang,
        &options.task,
        options.notimestamps,
    )?;

    state.reset();
    let mut last_logits = vec![0f32; S::VOCAB];
    for (i, &id) in prompt.iter().enumerate() {
        last_logits = decoder
            .step(&mut state, id)
            .map_err(|e| anyhow!("prime step {i} failed on window {window_index}: {e}"))?;
    }

    let tok_eot = S::TOKEN_EOT;
    let tok_notimestamps = tokenizer.token_to_id("<|notimestamps|>").unwrap();
    let prompt_len = prompt.len();
    let suppressor = TokenSuppressor::new::<S>(
        tokenizer,
        prompt_len,
        options.notimestamps,
        &options.suppress_tokens,
    )?;
    let suppress_fn = |tokens: &[u32], logits: &mut [f32]| suppressor.apply(tokens, logits);

    let mut generated: Vec<u32>;

    if options.beam_size > 1 {
        let mut beam_search = BeamSearch::<S>::new(
            options.beam_size,
            state.clone(),
            last_logits,
            prompt.clone(),
            1.0, // Alpha 1.0: standard length normalization for Whisper
        );
        for _ in 0..options.max_new_tokens {
            beam_search.step(decoder, &suppress_fn)?;
            if beam_search.beams.is_empty() {
                break;
            }
        }
        generated = beam_search.best_result().unwrap_or_default();
    } else {
        generated = Vec::new();
        let mut current_logits = last_logits;
        let mut tokens_with_prompt = prompt.clone();

        for _step in 0..options.max_new_tokens {
            let mut logits_1d = current_logits.clone();
            suppress_fn(&tokens_with_prompt, &mut logits_1d);

            let token_id = argmax_token(&logits_1d);
            if token_id == tok_eot {
                break;
            }

            generated.push(token_id);
            tokens_with_prompt.push(token_id);

            if generated.len() >= 8 {
                let tail = &generated[generated.len() - 8..];
                if tail[0..4] == tail[4..8] {
                    break;
                }
            }

            current_logits = decoder
                .step(&mut state, token_id)
                .map_err(|e| anyhow!("decoder step failed on window {window_index}: {e}"))?;
        }
    }

    let output_tokens = if options.beam_size > 1 {
        generated.get(prompt.len()..).unwrap_or(&[])
    } else {
        &generated
    };

    state.reset();
    let text = tokenizer.decode(output_tokens, true).unwrap();
    let segments = tokens_to_segments(
        tokenizer,
        output_tokens,
        window_index,
        samples_to_sec(window_start_sample),
        samples_to_sec(window_end_sample),
        tok_notimestamps + 1,
    );

    Ok(WindowTranscription {
        window_index,
        text,
        segments,
    })
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

fn fixed_audio_windows(audio_len: usize) -> Vec<AudioWindow> {
    (0..audio_len)
        .step_by(N_SAMPLES)
        .enumerate()
        .map(|(index, start_sample)| AudioWindow {
            index,
            start_sample,
            end_sample: (start_sample + N_SAMPLES).min(audio_len),
        })
        .collect()
}

fn vad_audio_windows(vad_segments: &[VadSegment]) -> Vec<AudioWindow> {
    let mut windows = Vec::new();
    for segment in vad_segments {
        let mut start = segment.start_sample;
        while start < segment.end_sample {
            let end = (start + N_SAMPLES).min(segment.end_sample);
            windows.push(AudioWindow {
                index: windows.len(),
                start_sample: start,
                end_sample: end,
            });
            start = end;
        }
    }
    windows
}

fn tokens_to_segments(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    window_index: usize,
    window_start_sec: f32,
    window_end_sec: f32,
    timestamp_begin: u32,
) -> Vec<TranscriptSegment> {
    let mut segments = Vec::new();
    let mut current_tokens = Vec::new();
    let mut current_start = None;
    let mut last_timestamp = None;

    for &token in tokens {
        if token >= timestamp_begin {
            let ts = window_start_sec + timestamp_token_to_sec(token, timestamp_begin);
            if !current_tokens.is_empty() {
                let start_sec = current_start.or(last_timestamp).unwrap_or(window_start_sec);
                let end_sec = ts.max(start_sec);
                let text = tokenizer.decode(&current_tokens, true).unwrap_or_default();
                if !text.trim().is_empty() {
                    segments.push(TranscriptSegment {
                        start_sec,
                        end_sec,
                        text,
                        tokens: current_tokens.clone(),
                        window_index,
                    });
                }
                current_tokens.clear();
            }
            current_start = Some(ts);
            last_timestamp = Some(ts);
        } else {
            current_tokens.push(token);
        }
    }

    if !current_tokens.is_empty() {
        let start_sec = current_start.or(last_timestamp).unwrap_or(window_start_sec);
        let text = tokenizer.decode(&current_tokens, true).unwrap_or_default();
        segments.push(TranscriptSegment {
            start_sec,
            end_sec: window_end_sec.max(start_sec),
            text,
            tokens: current_tokens,
            window_index,
        });
    }

    if segments.is_empty() && !tokens.is_empty() {
        let non_timestamp_tokens: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|&token| token < timestamp_begin)
            .collect();
        if !non_timestamp_tokens.is_empty() {
            let text = tokenizer
                .decode(&non_timestamp_tokens, true)
                .unwrap_or_default();
            if !text.trim().is_empty() {
                segments.push(TranscriptSegment {
                    start_sec: window_start_sec,
                    end_sec: window_end_sec,
                    text,
                    tokens: non_timestamp_tokens,
                    window_index,
                });
            }
        }
    }

    segments
}

#[inline]
fn timestamp_token_to_sec(token: u32, timestamp_begin: u32) -> f32 {
    (token - timestamp_begin) as f32 * 0.02
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

#[cfg(test)]
mod tests {
    use super::{fixed_audio_windows, timestamp_token_to_sec, vad_audio_windows};
    use crate::N_SAMPLES;
    use crate::vad::VadSegment;

    #[test]
    fn fixed_windows_cover_audio() {
        let windows = fixed_audio_windows(N_SAMPLES + 10);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start_sample, 0);
        assert_eq!(windows[0].end_sample, N_SAMPLES);
        assert_eq!(windows[1].start_sample, N_SAMPLES);
        assert_eq!(windows[1].end_sample, N_SAMPLES + 10);
    }

    #[test]
    fn vad_windows_split_long_segments() {
        let vad_segments = vec![VadSegment {
            start_sample: 100,
            end_sample: 100 + N_SAMPLES + 20,
            start_sec: 0.0,
            end_sec: 0.0,
        }];
        let windows = vad_audio_windows(&vad_segments);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start_sample, 100);
        assert_eq!(windows[0].end_sample, 100 + N_SAMPLES);
        assert_eq!(windows[1].end_sample, 100 + N_SAMPLES + 20);
    }

    #[test]
    fn vad_with_no_speech_has_no_windows() {
        assert!(vad_audio_windows(&[]).is_empty());
    }

    #[test]
    fn timestamp_tokens_are_20ms_steps() {
        assert_eq!(timestamp_token_to_sec(50364, 50364), 0.0);
        assert!((timestamp_token_to_sec(50369, 50364) - 0.1).abs() < 0.0001);
    }
}
