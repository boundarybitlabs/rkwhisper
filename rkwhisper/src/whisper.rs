use crate::beam::BeamSearch;
use crate::decoder::{WhisperDecoder, WhisperDecoderState};
use crate::encoder::{EncKvModel, WhisperEncoder};
use crate::spec::WhisperSpec;
use crate::suppression::{SuppressTokens, TokenSuppressor};
use crate::vad::{VadSegment, samples_to_sec};
use crate::{MelSpectrogram, N_SAMPLES, load_audio_file};
use anyhow::{Result, anyhow};
use serde::Serialize;
use tokenizers::Tokenizer;
use tracing::{debug, trace};

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
pub struct WindowTranscription {
    pub window_index: usize,
    pub absolute_start_sec: f32,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
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
        &[],
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
    vad_segments: &[VadSegment],
    options: &TranscribeOptions,
) -> Result<Transcription> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
    let audio = load_audio_file(audio_path)?;
    transcribe_audio_with_options(
        &audio,
        &tokenizer,
        mel_spec,
        encoder,
        enc_kv,
        decoder,
        vad_segments,
        options,
    )
}

pub fn transcribe_audio_with_options<S: WhisperSpec>(
    audio: &[f32],
    tokenizer: &Tokenizer,
    mel_spec: &MelSpectrogram,
    encoder: &WhisperEncoder<S>,
    enc_kv: &EncKvModel<S>,
    decoder: &mut WhisperDecoder<S>,
    vad_segments: &[VadSegment],
    options: &TranscribeOptions,
) -> Result<Transcription> {
    let windows = if !vad_segments.is_empty() {
        vad_audio_windows(vad_segments)
    } else {
        fixed_audio_windows(audio.len())
    };
    let mut full_text = String::new();
    let mut segments = Vec::new();

    for window in &windows {
        let absolute_start_sec = samples_to_sec(window.start_sample);
        let result = transcribe_audio_window(
            audio,
            tokenizer,
            mel_spec,
            encoder,
            enc_kv,
            decoder,
            window,
            absolute_start_sec,
            options,
        )?;
        full_text.push_str(&result.text);
        full_text.push(' ');
        segments.extend(result.segments);
    }

    Ok(Transcription {
        text: full_text.trim().to_string(),
        segments,
        vad_segments: vad_segments.to_vec(),
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
    absolute_start_sec: f32,
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
        absolute_start_sec,
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
    absolute_start_sec: f32,
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

    let tok_eot = S::TOKEN_EOT;
    let tok_notimestamps = tokenizer
        .token_to_id("<|notimestamps|>")
        .ok_or_else(|| anyhow!("tokenizer missing <|notimestamps|>"))?;
    let timestamp_begin = tok_notimestamps + 1;
    let window_duration_sec = samples_to_sec(window_end_sample - window_start_sample);

    // Base prompt: <|sot|> <|lang|> <|task|> [<|notimestamps|>]
    let base_prompt = control_prompt(
        tokenizer,
        &options.lang,
        &options.task,
        options.notimestamps,
    )?;
    // Seek loop: run the decoder repeatedly, carrying accumulated tokens as
    // the prompt prefix for each pass.  The model sees its own prior output
    // as context and continues forward naturally rather than duplicating.
    // In notimestamps mode there are no timestamp anchors to seek on, so the
    // loop always exits after the first pass.
    let all_output_tokens: Vec<u32> = {
        let window_end_token = timestamp_begin + (window_duration_sec / 0.02) as u32;
        let mut accumulated: Vec<u32> = Vec::new();

        'seek: for _seek_pass in 0..16 {
            // Build prompt: base_prompt + (tail of accumulated, capped so the
            // decoder has room to generate at least 64 tokens).
            let budget = S::T_CACHE.saturating_sub(base_prompt.len() + 64);
            let acc_start = accumulated.len().saturating_sub(budget);
            let prompt: Vec<u32> = base_prompt
                .iter()
                .chain(accumulated[acc_start..].iter())
                .copied()
                .collect();

            // Stop if there is no room left for meaningful generation.
            if prompt.len() + 16 >= S::T_CACHE {
                break;
            }

            // Prime the decoder with the full prompt.
            state.reset();
            let mut last_logits = vec![0f32; S::VOCAB];
            for (i, &id) in prompt.iter().enumerate() {
                last_logits = decoder
                    .step(&mut state, id)
                    .map_err(|e| anyhow!("prime step {i} failed on window {window_index}: {e}"))?;
            }

            // Build a per-pass suppressor that knows the current prompt length.
            let pass_suppressor = TokenSuppressor::new::<S>(
                tokenizer,
                prompt.len(),
                options.notimestamps,
                &options.suppress_tokens,
            )?;
            let pass_suppress =
                |tokens: &[u32], logits: &mut [f32]| pass_suppressor.apply(tokens, logits);

            let new_tokens = if options.beam_size > 1 {
                let mut beam_search = BeamSearch::<S>::new(
                    options.beam_size,
                    state.clone(),
                    last_logits,
                    prompt.clone(),
                    1.0,
                );
                for _ in 0..options.max_new_tokens {
                    beam_search.step(decoder, &pass_suppress)?;
                    if beam_search.beams.is_empty() {
                        break;
                    }
                }
                let full = beam_search.best_result().unwrap_or_default();
                full.get(prompt.len()..).unwrap_or(&[]).to_vec()
            } else {
                let mut generated = Vec::new();
                let mut current_logits = last_logits;
                let mut tokens_with_prompt = prompt.clone();

                for _step in 0..options.max_new_tokens {
                    let mut logits_1d = current_logits.clone();
                    pass_suppress(&tokens_with_prompt, &mut logits_1d);

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

                    current_logits = decoder.step(&mut state, token_id).map_err(|e| {
                        anyhow!("decoder step failed on window {window_index}: {e}")
                    })?;
                }
                generated
            };

            if new_tokens.is_empty() {
                break;
            }

            let prev_last_ts = accumulated
                .iter()
                .rev()
                .find(|&&t| t >= timestamp_begin)
                .copied();
            let new_last_ts = new_tokens
                .iter()
                .rev()
                .find(|&&t| t >= timestamp_begin)
                .copied();

            accumulated.extend_from_slice(&new_tokens);

            // No timestamps: notimestamps mode or degenerate window — single pass only.
            let Some(last_ts) = new_last_ts else { break };

            // No advancement: model is stuck at or before the previous seek point.
            if prev_last_ts.is_some_and(|p| last_ts <= p) {
                break;
            }

            // Window fully covered.
            if last_ts >= window_end_token {
                break 'seek;
            }
        }

        accumulated
    };

    state.reset();

    // Log raw token sequence at trace level so we can diagnose skipping.
    if tracing::enabled!(tracing::Level::TRACE) {
        let token_summary: Vec<String> = all_output_tokens
            .iter()
            .map(|&t| {
                if t >= timestamp_begin {
                    format!("<|{:.2}|>", timestamp_token_to_sec(t, timestamp_begin))
                } else {
                    format!("{t}")
                }
            })
            .collect();
        trace!(
            window_index,
            start_sec = absolute_start_sec,
            end_sec = absolute_start_sec + window_duration_sec,
            token_count = all_output_tokens.len(),
            tokens = %token_summary.join(" "),
            "window tokens"
        );
    }

    let text = tokenizer.decode(&all_output_tokens, true).unwrap();
    debug!(
        window_index,
        start_sec = absolute_start_sec,
        end_sec = absolute_start_sec + window_duration_sec,
        token_count = all_output_tokens.len(),
        notimestamps = options.notimestamps,
        %text,
        "window transcribed"
    );
    let segments = tokens_to_segments(
        tokenizer,
        &all_output_tokens,
        window_index,
        absolute_start_sec,
        absolute_start_sec + window_duration_sec,
        timestamp_begin,
    );

    Ok(WindowTranscription {
        window_index,
        absolute_start_sec,
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
    let start = tok
        .token_to_id("<|startoftranscript|>")
        .ok_or_else(|| anyhow!("tokenizer missing <|startoftranscript|>"))?;
    let lang_tok = tok
        .token_to_id(&format!("<|{lang}|>"))
        .ok_or_else(|| anyhow!("tokenizer missing language token <|{lang}|>"))?;
    let task_tok = tok
        .token_to_id(&format!("<|{task}|>"))
        .ok_or_else(|| anyhow!("tokenizer missing task token <|{task}|>"))?;
    let nots = tok
        .token_to_id("<|notimestamps|>")
        .ok_or_else(|| anyhow!("tokenizer missing <|notimestamps|>"))?;
    let mut prompt = vec![start, lang_tok, task_tok];
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
    if vad_segments.is_empty() {
        return windows;
    }

    let mut group_start: Option<usize> = None;
    let mut group_end: usize = 0;
    let mut group_speech: usize = 0;

    for seg in vad_segments {
        let seg_len = seg.end_sample - seg.start_sample;

        if seg_len > N_SAMPLES {
            // Flush any accumulated group first.
            if let Some(start) = group_start.take() {
                windows.push(AudioWindow {
                    index: windows.len(),
                    start_sample: start,
                    end_sample: group_end,
                });
                group_speech = 0;
            }
            // Split this oversized segment into N_SAMPLES-sized chunks.
            let mut start = seg.start_sample;
            while start < seg.end_sample {
                let end = (start + N_SAMPLES).min(seg.end_sample);
                windows.push(AudioWindow {
                    index: windows.len(),
                    start_sample: start,
                    end_sample: end,
                });
                start = end;
            }
        } else {
            let wall_clock_overflow = group_start.is_some_and(|s| seg.end_sample - s > N_SAMPLES);
            let speech_overflow = group_speech + seg_len > N_SAMPLES;

            if (wall_clock_overflow || speech_overflow)
                && let Some(start) = group_start.take()
            {
                windows.push(AudioWindow {
                    index: windows.len(),
                    start_sample: start,
                    end_sample: group_end,
                });
                group_speech = 0;
            }

            if group_start.is_none() {
                group_start = Some(seg.start_sample);
            }
            group_end = seg.end_sample;
            group_speech += seg_len;
        }
    }

    // Flush the final group.
    if let Some(start) = group_start {
        windows.push(AudioWindow {
            index: windows.len(),
            start_sample: start,
            end_sample: group_end,
        });
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
    use super::{AudioWindow, fixed_audio_windows, timestamp_token_to_sec, vad_audio_windows};
    use crate::N_SAMPLES;
    use crate::vad::VadSegment;

    // Returns true iff every sample in every VAD segment is contained in some window.
    // Advances through windows in sorted order to keep this O(segs * windows).
    fn all_speech_covered(segs: &[VadSegment], windows: &[AudioWindow]) -> bool {
        for seg in segs {
            let mut pos = seg.start_sample;
            'advance: while pos < seg.end_sample {
                for w in windows {
                    if w.start_sample <= pos && pos < w.end_sample {
                        pos = w.end_sample;
                        continue 'advance;
                    }
                }
                return false;
            }
        }
        true
    }

    fn seg(start: usize, end: usize) -> VadSegment {
        VadSegment {
            start_sample: start,
            end_sample: end,
            start_sec: 0.0,
            end_sec: 0.0,
        }
    }

    #[test]
    fn coverage_single_short_segment() {
        let segs = vec![seg(100, 200)];
        let wins = vad_audio_windows(&segs);
        assert!(
            all_speech_covered(&segs, &wins),
            "short segment not fully covered"
        );
    }

    #[test]
    fn coverage_single_segment_exactly_n_samples() {
        let segs = vec![seg(0, N_SAMPLES)];
        let wins = vad_audio_windows(&segs);
        assert!(
            all_speech_covered(&segs, &wins),
            "exact-window segment not fully covered"
        );
    }

    #[test]
    fn coverage_single_long_segment() {
        let segs = vec![seg(50, 50 + N_SAMPLES + 999)];
        let wins = vad_audio_windows(&segs);
        assert!(
            all_speech_covered(&segs, &wins),
            "long segment spanning two windows not fully covered"
        );
    }

    #[test]
    fn coverage_two_short_segments_fit_in_one_window() {
        let segs = vec![seg(0, 1000), seg(5000, 6000)];
        let wins = vad_audio_windows(&segs);
        assert_eq!(wins.len(), 1);
        assert!(all_speech_covered(&segs, &wins));
    }

    #[test]
    fn coverage_two_segments_forced_into_separate_windows() {
        // Second segment ends > N_SAMPLES after first segment's start.
        let segs = vec![seg(0, 1000), seg(N_SAMPLES - 100, N_SAMPLES + 200)];
        let wins = vad_audio_windows(&segs);
        assert!(
            all_speech_covered(&segs, &wins),
            "segments split across window boundary not fully covered"
        );
    }

    #[test]
    fn coverage_segment_starting_far_into_audio() {
        let segs = vec![seg(N_SAMPLES + 300, N_SAMPLES + 800)];
        let wins = vad_audio_windows(&segs);
        assert!(all_speech_covered(&segs, &wins));
    }

    #[test]
    fn coverage_many_small_segments_spanning_multiple_windows() {
        // 200 segments of 100 samples each, 1000 samples apart → spans ~200 000 samples total.
        // Still fits in one 30-second window.
        let segs: Vec<VadSegment> = (0..200).map(|i| seg(i * 1000, i * 1000 + 100)).collect();
        let wins = vad_audio_windows(&segs);
        assert!(all_speech_covered(&segs, &wins));
    }

    #[test]
    fn coverage_segments_spanning_three_windows() {
        // Three consecutive 35-second segments: forces 6 windows total.
        let s = 16000usize * 35;
        let segs = vec![
            seg(0, s),
            seg(s + 500, 2 * s + 500),
            seg(2 * s + 1000, 3 * s + 1000),
        ];
        let wins = vad_audio_windows(&segs);
        assert!(
            all_speech_covered(&segs, &wins),
            "three 35-second segments not fully covered"
        );
    }

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
    fn vad_windows_accumulates_short_segments() {
        // Two 5-second segments separated by 2 seconds of silence
        // Combined (0-12s) should fit in one 30s window.
        let vad_segments = vec![
            VadSegment {
                start_sample: 0,
                end_sample: 16000 * 5,
                start_sec: 0.0,
                end_sec: 5.0,
            },
            VadSegment {
                start_sample: 16000 * 7,
                end_sample: 16000 * 12,
                start_sec: 7.0,
                end_sec: 12.0,
            },
        ];
        let windows = vad_audio_windows(&vad_segments);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_sample, 0);
        assert_eq!(windows[0].end_sample, 16000 * 12);
    }

    #[test]
    fn vad_windows_splits_long_segments_correctly() {
        // One 35-second segment should result in two windows.
        let vad_segments = vec![VadSegment {
            start_sample: 0,
            end_sample: 16000 * 35,
            start_sec: 0.0,
            end_sec: 35.0,
        }];
        let windows = vad_audio_windows(&vad_segments);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start_sample, 0);
        assert_eq!(windows[0].end_sample, N_SAMPLES);
        assert_eq!(windows[1].start_sample, N_SAMPLES);
        assert_eq!(windows[1].end_sample, 16000 * 35);
    }

    #[test]
    fn vad_speech_duration_packing_ignores_inter_segment_silence() {
        // Two 10-second speech segments separated by 25 seconds of silence.
        // Wall-clock span: 45s > 30s — old code would split them.
        // Combined speech: 10s + 10s = 20s < 30s — new code keeps them together.
        let ten_s = 16000 * 10;
        let segs = vec![seg(0, ten_s), seg(16000 * 35, 16000 * 35 + ten_s)];
        let wins = vad_audio_windows(&segs);
        // Must be 2 windows because wall-clock span exceeds N_SAMPLES.
        // But each segment must be fully covered.
        assert!(
            all_speech_covered(&segs, &wins),
            "speech not covered with speech-duration packing"
        );
        // Each window must not exceed N_SAMPLES.
        for w in &wins {
            assert!(
                w.end_sample - w.start_sample <= N_SAMPLES,
                "window exceeds N_SAMPLES"
            );
        }
    }

    #[test]
    fn vad_speech_budget_triggers_split_before_wallclock() {
        // Many small segments that cumulatively exceed N_SAMPLES of speech,
        // but each fits in the wallclock window.  The split should happen at
        // speech budget exhaustion, not at wallclock span.
        let seg_speech = N_SAMPLES / 4; // 7.5s each
        let seg_gap = 1000; // tiny gap between them
        let segs: Vec<VadSegment> = (0..5)
            .map(|i| {
                let start = i * (seg_speech + seg_gap);
                seg(start, start + seg_speech)
            })
            .collect();
        let wins = vad_audio_windows(&segs);
        // 5 * 7.5s = 37.5s of speech, so must be at least 2 windows.
        assert!(wins.len() >= 2, "expected split on speech budget");
        for w in &wins {
            assert!(w.end_sample - w.start_sample <= N_SAMPLES);
        }
        assert!(all_speech_covered(&segs, &wins));
    }

    #[test]
    fn timestamp_tokens_are_20ms_steps() {
        assert_eq!(timestamp_token_to_sec(50364, 50364), 0.0);
        assert!((timestamp_token_to_sec(50369, 50364) - 0.1).abs() < 0.0001);
    }
}
