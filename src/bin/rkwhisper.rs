use anyhow::Result;
use clap::{Parser, ValueEnum};
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    MelSpectrogram,
    decoder::WhisperDecoder,
    encoder::{EncKvModel, WhisperEncoder},
    parallel::ParallelModelPaths,
    spec::{
        WhisperBase, WhisperLargeV3Turbo, WhisperMedium, WhisperSmall, WhisperSpec, WhisperTiny,
    },
    suppression::SuppressTokens,
    vad::{VadConfig, VadModel},
    whisper::{TranscribeOptions, transcribe_audio_with_options},
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(ValueEnum, Clone, Debug)]
enum Model {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV3Turbo,
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Model size to use
    #[arg(long, value_enum, default_value_t = Model::Medium)]
    model: Model,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: PathBuf,

    /// Path to mel-spectrogram .rknn
    #[arg(long)]
    mel_spec: PathBuf,

    /// Path to encoder .rknn
    #[arg(long)]
    encoder: PathBuf,

    /// Path to enc-KV .rknn (encoder hidden → per-layer cross-attn K/V)
    #[arg(long)]
    enc_kv: PathBuf,

    /// Path to decoder step .rknn
    #[arg(long)]
    decoder: PathBuf,

    /// Input .wav file (mono, 16 kHz)
    #[arg(value_name = "WAV_FILE")]
    wav: PathBuf,

    /// Language code, e.g. "en"
    #[arg(long, default_value = "en")]
    lang: String,

    /// Task: "transcribe" or "translate"
    #[arg(long, default_value = "transcribe")]
    task: String,

    /// Maximum new tokens to generate per 30-second chunk
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Beam size for token selection (1 = greedy)
    #[arg(long, default_value_t = 5)]
    beam_size: usize,

    /// Suppress timestamp tokens
    #[arg(long, default_value_t = false)]
    notimestamps: bool,

    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,

    /// Whisper token suppression: "default", "none", or comma-separated token IDs
    #[arg(long, default_value = "default")]
    suppress_tokens: String,

    /// Run 30-second windows across three RK3588 NPU cores using Tokio workers
    #[arg(long)]
    multi_npu: bool,

    /// Optional Silero-style VAD .rknn model
    #[arg(long)]
    vad_model: Option<PathBuf>,

    /// VAD speech probability threshold
    #[arg(long, default_value_t = 0.5)]
    vad_threshold: f32,

    /// Minimum speech segment length in milliseconds
    #[arg(long, default_value_t = 250)]
    vad_min_speech_ms: u32,

    /// Minimum silence gap before ending a speech segment in milliseconds
    #[arg(long, default_value_t = 100)]
    vad_min_silence_ms: u32,

    /// Padding added around VAD speech segments in milliseconds
    #[arg(long, default_value_t = 200)]
    vad_speech_pad_ms: u32,

    /// Audio samples per VAD inference window
    #[arg(long, default_value_t = 512)]
    vad_window_samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let lib = find_rknn_library()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Could not find rknn library"))?;

    match args.model {
        Model::Tiny => run::<WhisperTiny>(args, &lib),
        Model::Base => run::<WhisperBase>(args, &lib),
        Model::Small => run::<WhisperSmall>(args, &lib),
        Model::Medium => run::<WhisperMedium>(args, &lib),
        Model::LargeV3Turbo => run::<WhisperLargeV3Turbo>(args, &lib),
    }
}

fn run<S: WhisperSpec + Send + 'static>(args: Args, lib: &Path) -> Result<()> {
    let options = TranscribeOptions::new(
        args.lang.clone(),
        args.task.clone(),
        args.notimestamps,
        args.max_new_tokens,
        args.beam_size,
        SuppressTokens::parse(&args.suppress_tokens)?,
    );

    let wav_path = args.wav.to_string_lossy();
    let tokenizer_path = args.tokenizer.to_string_lossy();

    if args.multi_npu {
        let model_paths = ParallelModelPaths::new(
            args.mel_spec.clone(),
            args.encoder.clone(),
            args.enc_kv.clone(),
            args.decoder.clone(),
        );

        let vad_segments = if let Some(path) = &args.vad_model {
            let config = VadConfig {
                threshold: args.vad_threshold,
                min_speech_ms: args.vad_min_speech_ms,
                min_silence_ms: args.vad_min_silence_ms,
                speech_pad_ms: args.vad_speech_pad_ms,
                window_samples: args.vad_window_samples,
            };
            let vad = VadModel::new(
                RKNN::new_with_library(lib, &mut std::fs::read(path)?, 0)?,
                config,
            );
            let audio = rkwhisper::load_audio_file(wav_path.as_ref())?;
            vad.segments(&audio)?
        } else {
            Vec::new()
        };

        let mut pool = rkwhisper::parallel::ParallelTranscriberPool::<S>::new(lib, &model_paths)?;
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?,
        );
        let audio = rkwhisper::load_audio_file(wav_path.as_ref())?;
        let transcription = pool.transcribe_audio_with_segment_callback(
            &audio,
            tokenizer,
            &vad_segments,
            &options,
            |_| Ok(()),
        )?;
        return print_transcription(transcription, &args.output);
    }

    let encoder = WhisperEncoder::<S>::new(RKNN::new_with_library(
        lib,
        &mut std::fs::read(&args.encoder)?,
        0,
    )?);

    let mel_spec = MelSpectrogram::new(RKNN::new_with_library(
        lib,
        &mut std::fs::read(&args.mel_spec)?,
        0,
    )?);

    let enc_kv = EncKvModel::<S>::new(RKNN::new_with_library(
        lib,
        &mut std::fs::read(&args.enc_kv)?,
        0,
    )?);

    let dec_rknn = RKNN::new_with_library(lib, &mut std::fs::read(&args.decoder)?, 0)?;
    let mut decoder = WhisperDecoder::<S>::new(&dec_rknn);

    let audio = rkwhisper::load_audio_file(wav_path.as_ref())?;
    let vad_segments = if let Some(path) = &args.vad_model {
        let config = VadConfig {
            threshold: args.vad_threshold,
            min_speech_ms: args.vad_min_speech_ms,
            min_silence_ms: args.vad_min_silence_ms,
            speech_pad_ms: args.vad_speech_pad_ms,
            window_samples: args.vad_window_samples,
        };
        let vad = VadModel::new(
            RKNN::new_with_library(lib, &mut std::fs::read(path)?, 0)?,
            config,
        );
        vad.segments(&audio)?
    } else {
        Vec::new()
    };

    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    let transcription = transcribe_audio_with_options(
        &audio,
        &tokenizer,
        &mel_spec,
        &encoder,
        &enc_kv,
        &mut decoder,
        &vad_segments,
        &options,
    )?;

    print_transcription(transcription, &args.output)
}

fn print_transcription(
    transcription: rkwhisper::whisper::Transcription,
    output: &OutputFormat,
) -> Result<()> {
    match output {
        OutputFormat::Text => println!("{}", transcription.text),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&transcription)?),
    }
    Ok(())
}
