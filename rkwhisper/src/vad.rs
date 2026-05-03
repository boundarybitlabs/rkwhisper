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
            speech_pad_ms: 200,
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
        let mut probs = Vec::new();
        let mut state = vec![0.0f32; 2 * 128];
        for start in (0..audio.len()).step_by(self.config.window_samples) {
            let end = (start + self.config.window_samples).min(audio.len());
            probs.push((
                start,
                self.speech_probability(&audio[start..end], &mut state)?,
            ));
        }
        Ok(segments_from_probs(audio.len(), &probs, &self.config))
    }

    pub fn speech_probability(&self, window: &[f32], state: &mut [f32]) -> Result<f32> {
        let mut padded_window;
        let window_to_use = if window.len() != self.config.window_samples {
            padded_window = vec![0.0f32; self.config.window_samples];
            let len = window.len().min(self.config.window_samples);
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
}
