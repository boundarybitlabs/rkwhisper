use anyhow::Result;
use clap::Parser;
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    MelSpectrogram,
    decoder::WhisperDecoder,
    encoder::{EncKvModel, WhisperEncoder},
    spec::WhisperMedium,
    whisper::transcribe,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
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

    /// Suppress timestamp tokens
    #[arg(long, default_value_t = false)]
    notimestamps: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let lib = find_rknn_library().next().unwrap();

    let encoder = WhisperEncoder::<WhisperMedium>::new(RKNN::new_with_library(
        &lib,
        &mut std::fs::read(&args.encoder)?,
        0,
    )?);

    let mel_spec = MelSpectrogram::new(RKNN::new_with_library(
        &lib,
        &mut std::fs::read(&args.mel_spec)?,
        0,
    )?);

    let enc_kv = EncKvModel::<WhisperMedium>::new(RKNN::new_with_library(
        &lib,
        &mut std::fs::read(&args.enc_kv)?,
        0,
    )?);

    let dec_rknn = RKNN::new_with_library(&lib, &mut std::fs::read(&args.decoder)?, 0)?;
    let mut decoder = WhisperDecoder::<WhisperMedium>::new(&dec_rknn);

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
    )?;

    println!("{text}");
    Ok(())
}
