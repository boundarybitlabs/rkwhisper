use anyhow::Result;
use clap::{Parser, ValueEnum};
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    MelSpectrogram,
    decoder::WhisperDecoder,
    encoder::{EncKvModel, WhisperEncoder},
    spec::{WhisperLargeV3Turbo, WhisperMedium, WhisperSmall, WhisperSpec},
    whisper::transcribe,
};
use std::path::{Path, PathBuf};

#[derive(ValueEnum, Clone, Debug)]
enum Model {
    Small,
    Medium,
    LargeV3Turbo,
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
}

fn main() -> Result<()> {
    let args = Args::parse();

    let lib = find_rknn_library()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Could not find rknn library"))?;

    match args.model {
        Model::Small => run::<WhisperSmall>(args, &lib),
        Model::Medium => run::<WhisperMedium>(args, &lib),
        Model::LargeV3Turbo => run::<WhisperLargeV3Turbo>(args, &lib),
    }
}

fn run<S: WhisperSpec>(args: Args, lib: &Path) -> Result<()> {
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

    let text = transcribe(
        &args.wav.to_string_lossy(),
        &args.tokenizer.to_string_lossy(),
        &mel_spec,
        &encoder,
        &enc_kv,
        &mut decoder,
        &args.lang,
        &args.task,
        args.notimestamps,
        args.max_new_tokens,
        args.beam_size,
    )?;

    println!("{text}");
    Ok(())
}
