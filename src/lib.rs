use anyhow::{Context, bail};
use rknpu2::{
    RKNN,
    api::runtime::RuntimeAPI,
    io::{
        buffer::{BufMutView, BufView},
        input::Input,
        output::{Output, OutputKind},
    },
    tensor::{TensorFormat, TensorFormatKind},
};

pub mod beam;
pub mod daemon;
pub mod decoder;
pub mod encoder;
pub mod parallel;
pub mod spec;
pub mod suppression;
pub mod vad;
pub mod whisper;

pub const N_SAMPLES: usize = SAMPLE_RATE as usize * 30;
pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
pub const N_FRAMES: usize = 3000;
pub const EPS: f32 = 1e-10;
pub const LOG_FLOOR: f32 = -10.0; // log10(EPS)
const PAD: usize = N_FFT / 2;

pub const TARGET: usize = PAD + N_FRAMES * HOP_LENGTH + PAD; // 480400

pub struct MelSpectrogram {
    rknn: RKNN<RuntimeAPI>,
}

impl MelSpectrogram {
    pub fn new(rknn: RKNN<RuntimeAPI>) -> Self {
        Self { rknn }
    }

    pub fn log_mel_spectrogram(&self, audio: &[f32]) -> anyhow::Result<Vec<f32>> {
        let mut wave = vec![0.0; N_SAMPLES];
        wave[..audio.len()].copy_from_slice(audio);
        let wave = polyphase_pre_process(&wave);
        let inputs = vec![Input {
            index: 0,
            buffer: BufView::F32(&wave),
            pass_through: false,
            fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
        }];

        let mut buf = vec![0.0; N_FRAMES * N_MELS];

        let mut outputs = vec![Output {
            index: 0,
            kind: OutputKind::Preallocated {
                buf: BufMutView::F32(&mut buf),
                want_float: true,
            },
        }];

        self.rknn.set_inputs(inputs)?;

        self.rknn.run()?;

        self.rknn.get_outputs(&mut outputs)?;

        let mut max_log_spec = f32::NEG_INFINITY;
        for val in buf.iter_mut() {
            *val = val.max(EPS).log10();
            if *val > max_log_spec {
                max_log_spec = *val;
            }
        }

        let log_spec_threshold = max_log_spec - 8.0;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;
        for val in buf.iter_mut() {
            *val = (val.max(log_spec_threshold) + 4.0) / 4.0;
            if *val < min {
                min = *val;
            }
            if *val > max {
                max = *val;
            }
            sum += *val;
        }

        let mean = sum / buf.len() as f32;
        eprintln!("Mel Spectrogram Stats: min={min:.4}, max={max:.4}, mean={mean:.4}");

        Ok(buf)
    }
}

pub fn load_audio_file(path: &str) -> Result<Vec<f32>, anyhow::Error> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {:?}", path))?;
    let spec = reader.spec();

    // Validate audio specs
    if spec.channels != 1 {
        bail!("Expected mono audio, got {} channels", spec.channels);
    }
    if spec.sample_rate != 16_000 {
        bail!("Expected 16 kHz sample rate, got {}", spec.sample_rate);
    }

    let wave: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            // normalize int samples to [-1, 1]
            let max_val = match spec.bits_per_sample {
                8 => i8::MAX as f32,
                16 => i16::MAX as f32,
                24 => (1 << 23) as f32 - 1.0,
                32 => i32::MAX as f32,
                _ => anyhow::bail!("unsupported PCM bit depth: {}", spec.bits_per_sample),
            };
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
    };

    Ok(wave)
}

pub fn polyphase_pre_process(input: &[f32]) -> Vec<f32> {
    let pad_len = 200;
    let gcd = 80;
    let original_len = input.len();

    // 1. Create the padded buffer
    // We need enough space for left_pad + data + right_pad + alignment_pad
    let mut padded_len = pad_len + original_len + pad_len;

    // Calculate alignment padding
    let rem = padded_len % gcd;
    let align_pad = if rem > 0 { gcd - rem } else { 0 };
    padded_len += align_pad;

    let mut x_padded = Vec::with_capacity(padded_len);

    // A. Left Reflect Padding: input[1..201].reversed()
    // Note: "reflect" usually means we skip index 0.
    // PyTorch: pad(1,2,3,4) -> (2,1, 2,3,4, 3,2)
    for i in (1..=pad_len).rev() {
        // Safety check: clamp index if input is shorter than pad_len
        let idx = if i < original_len {
            i
        } else {
            original_len - 1
        };
        x_padded.push(input[idx]);
    }

    // B. Original Data
    x_padded.extend_from_slice(input);

    // C. Right Reflect Padding: input[len-2..len-202].reversed()
    // PyTorch reflect logic at end: mirrors around the last element.
    for i in 0..pad_len {
        let idx = if original_len > 1 + i {
            original_len - 2 - i
        } else {
            0
        };
        x_padded.push(input[idx]);
    }

    // D. Alignment Zero Padding
    for _ in 0..align_pad {
        x_padded.push(0.0);
    }

    // 2. Polyphase Transpose
    // Current Layout: [Batch, Time, Channels=80] (Row Major)
    // Target Layout:  [Batch, Channels=80, Time] (Column Major equivalent)
    // We want to read columns of the "Time x 80" matrix and write them as rows.

    let num_channels = gcd; // 80
    let num_time_steps = x_padded.len() / num_channels;

    let mut output = vec![0.0; x_padded.len()];

    // Loop optimization: The target is contiguous in channel blocks.
    // Output layout:
    // [ Channel 0 (all times) ]
    // [ Channel 1 (all times) ]
    // ...

    for c in 0..num_channels {
        let dest_offset = c * num_time_steps;
        for t in 0..num_time_steps {
            // Source index: Time major. Every step advances by 80.
            let src_idx = t * num_channels + c;

            // Dest index: Channel major. Contiguous for this channel loop.
            let dest_idx = dest_offset + t;

            // Unchecked access is safe here because we calculated bounds exactly
            // typically gives a nice speedup in hot loops
            unsafe {
                *output.get_unchecked_mut(dest_idx) = *x_padded.get_unchecked(src_idx);
            }
        }
    }

    output
}
