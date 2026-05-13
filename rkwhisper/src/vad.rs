use crate::SAMPLE_RATE;
use anyhow::Result;
use rknpu2::RKNN;
use rknpu2::api::runtime::RuntimeAPI;
use rknpu2::io::buffer::{BufMutView, BufView};
use rknpu2::io::input::Input;
use rknpu2::io::output::{Output, OutputKind};
use rknpu2::tensor::{TensorFormat, TensorFormatKind};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    pub window_samples: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 100,
            window_samples: 512,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VadSegment {
    pub start_sample: usize,
    pub end_sample: usize,
    pub start_sec: f32,
    pub end_sec: f32,
}

pub struct VadModel {
    rknn: RKNN<RuntimeAPI>,
    config: VadConfig,
}

impl VadModel {
    pub fn new(rknn: RKNN<RuntimeAPI>, config: VadConfig) -> Self {
        Self { rknn, config }
    }

    pub fn segments(&self, audio: &[f32]) -> Result<Vec<VadSegment>> {
        self.segments_with_config(audio, &self.config)
    }

    pub fn segments_with_config(
        &self,
        audio: &[f32],
        config: &VadConfig,
    ) -> Result<Vec<VadSegment>> {
        let mut probs = Vec::new();
        let mut state = vec![0.0f32; 2 * 128];
        for start in (0..audio.len()).step_by(config.window_samples) {
            let end = (start + config.window_samples).min(audio.len());
            probs.push((
                start,
                self.speech_probability(&audio[start..end], &mut state)?,
            ));
        }
        Ok(segments_from_probs(audio.len(), &probs, config))
    }

    pub fn speech_probability(&self, window: &[f32], state: &mut [f32]) -> Result<f32> {
        self.speech_probability_with_window_samples(window, state, self.config.window_samples)
    }

    pub fn speech_probability_with_window_samples(
        &self,
        window: &[f32],
        state: &mut [f32],
        window_samples: usize,
    ) -> Result<f32> {
        let mut padded_window;
        let window_to_use = if window.len() != window_samples {
            padded_window = vec![0.0f32; window_samples];
            let len = window.len().min(window_samples);
            padded_window[..len].copy_from_slice(&window[..len]);
            &padded_window
        } else {
            window
        };
        self.rknn.set_inputs(vec![
            Input {
                index: 0,
                buffer: BufView::F32(window_to_use),
                pass_through: false,
                fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
            },
            Input {
                index: 1,
                buffer: BufView::F32(state),
                pass_through: false,
                fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
            },
        ])?;
        self.rknn.run()?;

        let mut prob = [0.0f32; 1];
        let mut next_state = vec![0.0f32; state.len()];
        let mut outputs = vec![
            Output {
                index: 0,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F32(&mut prob),
                    want_float: true,
                },
            },
            Output {
                index: 1,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F32(&mut next_state),
                    want_float: true,
                },
            },
        ];
        self.rknn.get_outputs(&mut outputs)?;
        state.copy_from_slice(&next_state);
        Ok(prob[0])
    }
}

pub struct StreamingVad {
    state: Vec<f32>,
    config: VadConfig,
}

impl StreamingVad {
    pub fn new(config: VadConfig) -> Self {
        Self {
            state: vec![0.0f32; 2 * 128],
            config,
        }
    }

    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    pub fn process_window(&mut self, model: &VadModel, window: &[f32]) -> Result<f32> {
        model.speech_probability(window, &mut self.state)
    }
}

pub fn segments_from_probs(
    audio_len: usize,
    probs: &[(usize, f32)],
    config: &VadConfig,
) -> Vec<VadSegment> {
    let min_speech = ms_to_samples(config.min_speech_ms);
    let min_silence = ms_to_samples(config.min_silence_ms);
    let speech_pad = ms_to_samples(config.speech_pad_ms);

    let mut raw = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut last_speech_end = 0usize;

    for &(start, prob) in probs {
        let end = (start + config.window_samples).min(audio_len);
        if prob >= config.threshold {
            if current_start.is_none() {
                current_start = Some(start);
            }
            last_speech_end = end;
        } else if let Some(segment_start) = current_start {
            if start.saturating_sub(last_speech_end) >= min_silence {
                raw.push((segment_start, last_speech_end));
                current_start = None;
            }
        }
    }

    if let Some(segment_start) = current_start {
        raw.push((segment_start, last_speech_end.max(segment_start)));
    }

    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in raw {
        if end.saturating_sub(start) < min_speech {
            continue;
        }
        let padded_start = start.saturating_sub(speech_pad);
        let padded_end = (end + speech_pad).min(audio_len);
        if let Some((_, prev_end)) = merged.last_mut() {
            if padded_start <= *prev_end {
                *prev_end = (*prev_end).max(padded_end);
                continue;
            }
        }
        merged.push((padded_start, padded_end));
    }

    merged
        .into_iter()
        .map(|(start_sample, end_sample)| VadSegment {
            start_sample,
            end_sample,
            start_sec: samples_to_sec(start_sample),
            end_sec: samples_to_sec(end_sample),
        })
        .collect()
}

#[inline]
pub fn samples_to_sec(samples: usize) -> f32 {
    samples as f32 / SAMPLE_RATE as f32
}

#[inline]
fn ms_to_samples(ms: u32) -> usize {
    (ms as usize * SAMPLE_RATE as usize) / 1000
}

#[cfg(test)]
mod tests {
    use super::{VadConfig, segments_from_probs};

    #[test]
    fn merges_speech_across_short_silence_and_adds_padding() {
        let config = VadConfig {
            threshold: 0.5,
            min_speech_ms: 10,
            min_silence_ms: 100,
            speech_pad_ms: 10,
            window_samples: 160,
        };
        let probs = vec![
            (0, 0.0),
            (160, 0.9),
            (320, 0.0),
            (480, 0.8),
            (640, 0.0),
            (2400, 0.0),
        ];
        let segments = segments_from_probs(3200, &probs, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments[0].end_sample, 800);
    }

    #[test]
    fn drops_short_speech() {
        let config = VadConfig {
            threshold: 0.5,
            min_speech_ms: 100,
            min_silence_ms: 10,
            speech_pad_ms: 0,
            window_samples: 160,
        };
        let probs = vec![(0, 0.9), (160, 0.0), (320, 0.0)];
        let segments = segments_from_probs(1000, &probs, &config);
        assert!(segments.is_empty());
    }

    // Speech that ends with the final prob window (no trailing silence) must still
    // be captured — the post-loop flush must fire.
    #[test]
    fn captures_speech_at_end_of_audio_with_no_trailing_silence() {
        let config = VadConfig {
            threshold: 0.5,
            min_speech_ms: 10,
            min_silence_ms: 100,
            speech_pad_ms: 0,
            window_samples: 160,
        };
        let probs = vec![(0, 0.0), (160, 0.9), (320, 0.9)];
        let audio_len = 480;
        let segments = segments_from_probs(audio_len, &probs, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_sample, 160);
        assert_eq!(segments[0].end_sample, audio_len);
    }

    // A single window of speech at position 0 must produce a segment.
    #[test]
    fn captures_single_speech_window_at_start() {
        let config = VadConfig {
            threshold: 0.5,
            min_speech_ms: 10,
            min_silence_ms: 200,
            speech_pad_ms: 0,
            window_samples: 512,
        };
        let probs = vec![(0, 0.9), (512, 0.0), (1024, 0.0), (1536, 0.0), (2048, 0.0)];
        let audio_len = 2560;
        let segments = segments_from_probs(audio_len, &probs, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments[0].end_sample, 512);
    }

    // Speech at the very last window of the audio (no silence follows).
    #[test]
    fn captures_speech_only_in_last_window() {
        let config = VadConfig {
            threshold: 0.5,
            min_speech_ms: 10,
            min_silence_ms: 100,
            speech_pad_ms: 0,
            window_samples: 512,
        };
        let audio_len = 512;
        let probs = vec![(0, 0.9)];
        let segments = segments_from_probs(audio_len, &probs, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments[0].end_sample, 512);
    }

    // Two speech bursts with exactly min_silence of gap — they should remain separate.
    #[test]
    fn two_speech_bursts_separated_by_exact_min_silence() {
        // min_silence_ms=100 @ 16 kHz = 1600 samples. window_samples=512.
        // Gap from last_speech_end to start of next silence check must reach 1600.
        // Speech at 0..512, then silence at 512,1024,1536 (gap=1536 < 1600), then 2048 (gap=2048-512=1536 still < 1600)
        // Gap at 2560: 2560 - 512 = 2048 >= 1600 → segment closes.
        // Then speech at 3072.
        let config = VadConfig {
            threshold: 0.5,
            min_speech_ms: 10,
            min_silence_ms: 100, // 1600 samples
            speech_pad_ms: 0,
            window_samples: 512,
        };
        let probs = vec![
            (0, 0.9),
            (512, 0.0),
            (1024, 0.0),
            (1536, 0.0),
            (2048, 0.0),
            (2560, 0.0), // gap from 512 to 2560 = 2048 >= 1600 → closes first segment
            (3072, 0.9),
            (3584, 0.0),
            (4096, 0.0),
            (4608, 0.0),
            (5120, 0.0), // gap from 3584 to 5120 = 1536 >= 1600? No: 5120-3584=1536 < 1600
            (5632, 0.0), // 5632-3584=2048 >= 1600 → closes second segment
        ];
        let audio_len = 6144;
        let segments = segments_from_probs(audio_len, &probs, &config);
        assert_eq!(segments.len(), 2, "expected two separate speech segments");
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments[0].end_sample, 512);
        assert_eq!(segments[1].start_sample, 3072);
        assert_eq!(segments[1].end_sample, 3584);
    }
}
