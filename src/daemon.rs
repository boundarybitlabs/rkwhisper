use crate::SAMPLE_RATE;
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const DEFAULT_MODEL_ROOT: &str = "/usr/share/rkwhisper";
pub const MODEL_ROOT_ENV: &str = "RKWHISPER_MODEL_ROOT";
pub const DEFAULT_SOCKET_PATH: &str = "/run/rkwhisper/asr.sock";
pub const DEFAULT_CONFIG_PATH: &str = "/etc/rkwhisper.toml";
pub const MAX_HEADER_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize)]
pub struct RequestHeader {
    pub model: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_lang")]
    pub lang: String,
    #[serde(default = "default_task")]
    pub task: String,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_beam_size")]
    pub beam_size: usize,
    #[serde(default)]
    pub notimestamps: bool,
    #[serde(default = "default_suppress_tokens")]
    pub suppress_tokens: String,
    #[serde(default)]
    pub vad_threshold: Option<f32>,
    #[serde(default)]
    pub vad_min_speech_ms: Option<u32>,
    #[serde(default)]
    pub vad_min_silence_ms: Option<u32>,
    #[serde(default)]
    pub vad_speech_pad_ms: Option<u32>,
    #[serde(default)]
    pub vad_window_samples: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct DaemonRequest {
    pub header: RequestHeader,
    pub audio: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DaemonConfig {
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV3Turbo,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFiles {
    pub id: String,
    pub kind: ModelKind,
    pub dir: PathBuf,
    pub tokenizer: PathBuf,
    pub mel: PathBuf,
    pub encoder: PathBuf,
    pub enc_kv: PathBuf,
    pub decoder: PathBuf,
    pub vad: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum DaemonResponse<'a> {
    #[serde(rename = "segment")]
    Segment { text: &'a str, begin: f32, end: f32 },
    #[serde(rename = "done")]
    Done { audio_s: f32, rtf: f32 },
    #[serde(rename = "error")]
    Error { error: &'a str },
}

pub fn default_model_root() -> PathBuf {
    std::env::var_os(MODEL_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_ROOT))
}

pub fn load_config(path: &Path) -> Result<DaemonConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    parse_config(&contents).with_context(|| format!("failed to parse config {}", path.display()))
}

pub fn parse_config(contents: &str) -> Result<DaemonConfig> {
    let config: DaemonConfig = toml::from_str(contents)?;
    validate_config(&config)?;
    Ok(config)
}

pub fn read_request<R: Read>(reader: &mut R) -> Result<DaemonRequest> {
    let header = read_header(reader)?;
    read_request_body(header, reader)
}

pub fn read_request_body<R: Read>(header: RequestHeader, reader: &mut R) -> Result<DaemonRequest> {
    if header.mode != "batch" && header.mode != "stream" {
        bail!("unsupported mode {:?}", header.mode);
    }
    if header.beam_size == 0 {
        bail!("beam_size must be at least 1");
    }

    let audio = read_pcm_frame(reader)?.ok_or_else(|| anyhow!("empty audio"))?;

    Ok(DaemonRequest { header, audio })
}

pub fn read_pcm_frame<R: Read>(reader: &mut R) -> Result<Option<Vec<f32>>> {
    let mut len_buf = [0u8; 4];
    let n = reader
        .read(&mut len_buf[..1])
        .context("failed to read PCM byte count")?;
    if n == 0 {
        return Ok(None);
    }
    reader
        .read_exact(&mut len_buf[1..])
        .context("failed to read PCM byte count")?;

    let pcm_len = i32::from_le_bytes(len_buf);
    if pcm_len < 0 {
        bail!("negative PCM byte count");
    }
    let pcm_len = pcm_len as usize;
    if pcm_len == 0 {
        return Ok(None);
    }
    if pcm_len % 2 != 0 {
        bail!("PCM byte count must be even for s16le audio");
    }

    let mut pcm = vec![0u8; pcm_len];
    reader
        .read_exact(&mut pcm)
        .context("failed to read PCM body")?;
    let audio = pcm_s16le_to_f32(&pcm)?;

    Ok(Some(audio))
}

pub fn read_header<R: Read>(reader: &mut R) -> Result<RequestHeader> {
    let mut bytes = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = reader
            .read(&mut byte)
            .context("failed to read JSON header")?;
        if n == 0 {
            bail!("connection closed before JSON header newline");
        }
        if byte[0] == b'\n' {
            break;
        }
        bytes.push(byte[0]);
        if bytes.len() > MAX_HEADER_BYTES {
            bail!("JSON header exceeds {MAX_HEADER_BYTES} bytes");
        }
    }

    let header = std::str::from_utf8(&bytes).context("JSON header is not valid UTF-8")?;
    serde_json::from_str(header).context("failed to parse JSON header")
}

pub fn pcm_s16le_to_f32(pcm: &[u8]) -> Result<Vec<f32>> {
    if pcm.len() % 2 != 0 {
        bail!("PCM byte count must be even for s16le audio");
    }
    Ok(pcm
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / i16::MAX as f32)
        .collect())
}

pub fn audio_seconds(sample_count: usize) -> f32 {
    sample_count as f32 / SAMPLE_RATE as f32
}

pub fn real_time_factor(elapsed: Duration, audio_s: f32) -> f32 {
    if audio_s <= 0.0 {
        return 0.0;
    }
    elapsed.as_secs_f32() / audio_s
}

pub fn resolve_model_files(root: &Path, model_id: &str) -> Result<ModelFiles> {
    validate_model_id(model_id)?;
    let kind = model_kind(model_id).ok_or_else(|| anyhow!("model not found"))?;
    resolve_model_files_for_kind(root, model_id, kind)
}

pub fn resolve_enabled_model_files(
    root: &Path,
    config: &DaemonConfig,
    model_id: &str,
) -> Result<ModelFiles> {
    validate_model_id(model_id)?;
    if !config.models.iter().any(|id| id == model_id) {
        bail!("model not found");
    }
    let kind = model_kind(model_id).ok_or_else(|| anyhow!("model not found"))?;
    resolve_model_files_for_kind(root, model_id, kind)
}

pub fn model_is_enabled(config: &DaemonConfig, model_id: &str) -> bool {
    config.models.iter().any(|id| id == model_id)
}

fn resolve_model_files_for_kind(
    root: &Path,
    model_id: &str,
    kind: ModelKind,
) -> Result<ModelFiles> {
    let dir = root.join(model_id);
    if !dir.is_dir() {
        bail!("model not found");
    }

    let tokenizer = required_file(&dir, "tokenizer.json")?;
    let mel = required_file(&dir, "mel.rknn")?;
    let encoder = required_file(&dir, "encoder.rknn")?;
    let enc_kv = required_file(&dir, "enc_kv.rknn")?;
    let decoder = required_file(&dir, "decoder.rknn")?;
    let vad_path = dir.join("vad.rknn");
    let vad = vad_path.is_file().then_some(vad_path);

    Ok(ModelFiles {
        id: model_id.to_string(),
        kind,
        dir,
        tokenizer,
        mel,
        encoder,
        enc_kv,
        decoder,
        vad,
    })
}

pub fn response_line(response: &DaemonResponse<'_>) -> Result<String> {
    let mut line = serde_json::to_string(response)?;
    line.push('\n');
    Ok(line)
}

fn required_file(dir: &Path, name: &str) -> Result<PathBuf> {
    let path = dir.join(name);
    if path.is_file() {
        Ok(path)
    } else {
        bail!("model file missing: {name}");
    }
}

fn validate_model_id(model_id: &str) -> Result<()> {
    if model_id.is_empty()
        || model_id
            .bytes()
            .any(|b| !(b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
    {
        bail!("invalid model id");
    }
    Ok(())
}

fn validate_config(config: &DaemonConfig) -> Result<()> {
    if config.models.is_empty() {
        bail!("config must list at least one model");
    }

    let mut seen = HashSet::new();
    for model_id in &config.models {
        validate_model_id(model_id)?;
        if model_kind(model_id).is_none() {
            bail!("unsupported model id: {model_id}");
        }
        if !seen.insert(model_id) {
            bail!("duplicate model id: {model_id}");
        }
    }

    Ok(())
}

fn model_kind(model_id: &str) -> Option<ModelKind> {
    match model_id.strip_suffix("-30s").unwrap_or(model_id) {
        "whisper-tiny" => Some(ModelKind::Tiny),
        "whisper-base" => Some(ModelKind::Base),
        "whisper-small" => Some(ModelKind::Small),
        "whisper-medium" => Some(ModelKind::Medium),
        "whisper-large-v3-turbo" => Some(ModelKind::LargeV3Turbo),
        _ => None,
    }
}

fn default_mode() -> String {
    "batch".to_string()
}

fn default_lang() -> String {
    "en".to_string()
}

fn default_task() -> String {
    "transcribe".to_string()
}

fn default_max_new_tokens() -> usize {
    128
}

fn default_beam_size() -> usize {
    5
}

fn default_suppress_tokens() -> String {
    "default".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn reads_framed_batch_request() {
        let mut bytes = br#"{"model":"whisper-small-30s","beam_size":2,"mode":"batch"}"#.to_vec();
        bytes.push(b'\n');
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.extend_from_slice(&i16::MAX.to_le_bytes());

        let request = read_request(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(request.header.model, "whisper-small-30s");
        assert_eq!(request.header.beam_size, 2);
        assert_eq!(request.audio, vec![0.0, 1.0]);
    }

    #[test]
    fn reads_framed_stream_request() {
        let mut bytes = br#"{"model":"whisper-small-30s","mode":"stream"}"#.to_vec();
        bytes.push(b'\n');
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());

        let request = read_request(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(request.header.mode, "stream");
        assert_eq!(request.audio, vec![0.0]);
    }

    #[test]
    fn rejects_invalid_pcm_lengths() {
        let mut negative = br#"{"model":"whisper-small-30s"}"#.to_vec();
        negative.push(b'\n');
        negative.extend_from_slice(&(-1i32).to_le_bytes());
        assert!(read_request(&mut Cursor::new(negative)).is_err());

        let mut odd = br#"{"model":"whisper-small-30s"}"#.to_vec();
        odd.push(b'\n');
        odd.extend_from_slice(&1i32.to_le_bytes());
        odd.push(0);
        assert!(read_request(&mut Cursor::new(odd)).is_err());
    }

    #[test]
    fn rejects_truncated_frames() {
        let missing_newline = br#"{"model":"whisper-small-30s"}"#.to_vec();
        assert!(read_request(&mut Cursor::new(missing_newline)).is_err());

        let mut short_body = br#"{"model":"whisper-small-30s"}"#.to_vec();
        short_body.push(b'\n');
        short_body.extend_from_slice(&4i32.to_le_bytes());
        short_body.extend_from_slice(&0i16.to_le_bytes());
        assert!(read_request(&mut Cursor::new(short_body)).is_err());
    }

    #[test]
    fn rejects_zero_beam_size() {
        let mut bytes = br#"{"model":"whisper-small-30s","beam_size":0}"#.to_vec();
        bytes.push(b'\n');
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());

        let err = read_request(&mut Cursor::new(bytes))
            .unwrap_err()
            .to_string();
        assert!(err.contains("beam_size"));
    }

    #[test]
    fn converts_s16le_pcm() {
        let mut pcm = Vec::new();
        pcm.extend_from_slice(&i16::MIN.to_le_bytes());
        pcm.extend_from_slice(&0i16.to_le_bytes());
        pcm.extend_from_slice(&i16::MAX.to_le_bytes());

        let audio = pcm_s16le_to_f32(&pcm).unwrap();
        assert!((audio[0] + 32768.0 / 32767.0).abs() < 0.0001);
        assert_eq!(audio[1], 0.0);
        assert_eq!(audio[2], 1.0);
    }

    #[test]
    fn resolves_fixed_model_layout() {
        let root = unique_temp_dir();
        let model_dir = root.join("whisper-small-30s");
        create_model_dir(&model_dir);

        let files = resolve_model_files(&root, "whisper-small-30s").unwrap();
        assert_eq!(files.kind, ModelKind::Small);
        assert_eq!(files.tokenizer, model_dir.join("tokenizer.json"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_valid_config() {
        let config = parse_config(
            r#"
models = [
  "whisper-small-30s",
  "whisper-medium-30s",
]
"#,
        )
        .unwrap();

        assert!(model_is_enabled(&config, "whisper-small-30s"));
        assert!(model_is_enabled(&config, "whisper-medium-30s"));
        assert!(!model_is_enabled(&config, "whisper-base-30s"));
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(parse_config("models = []").is_err());
        assert!(
            parse_config(
                r#"
models = [
  "whisper-small-30s",
  "whisper-small-30s",
]
"#
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate")
        );
        assert!(
            parse_config(r#"models = ["../whisper-small-30s"]"#)
                .unwrap_err()
                .to_string()
                .contains("invalid model id")
        );
        assert!(
            parse_config(r#"models = ["whisper-unknown-30s"]"#)
                .unwrap_err()
                .to_string()
                .contains("unsupported model id")
        );
    }

    #[test]
    fn missing_config_file_errors() {
        let path = unique_temp_dir().join("rkwhisper.toml");
        assert!(load_config(&path).is_err());
    }

    #[test]
    fn resolves_only_enabled_models() {
        let root = unique_temp_dir();
        create_model_dir(&root.join("whisper-small-30s"));
        create_model_dir(&root.join("whisper-medium-30s"));
        let config = parse_config(r#"models = ["whisper-small-30s"]"#).unwrap();

        let files = resolve_enabled_model_files(&root, &config, "whisper-small-30s").unwrap();
        assert_eq!(files.kind, ModelKind::Small);

        let err = resolve_enabled_model_files(&root, &config, "whisper-medium-30s")
            .unwrap_err()
            .to_string();
        assert_eq!(err, "model not found");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn listed_model_missing_file_reports_missing_file() {
        let root = unique_temp_dir();
        let model_dir = root.join("whisper-small-30s");
        fs::create_dir_all(&model_dir).unwrap();
        File::create(model_dir.join("tokenizer.json")).unwrap();
        let config = parse_config(r#"models = ["whisper-small-30s"]"#).unwrap();

        let err = resolve_enabled_model_files(&root, &config, "whisper-small-30s")
            .unwrap_err()
            .to_string();
        assert!(err.contains("model file missing: mel.rknn"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn serializes_single_line_responses() {
        let line = response_line(&DaemonResponse::Segment {
            text: "Eeenie, meanie, minie, mo",
            begin: 45.6,
            end: 52.7,
        })
        .unwrap();
        assert!(line.ends_with('\n'));
        assert!(line.contains(r#""type":"segment""#));
        assert!(line.contains(r#""begin":45.6"#));
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rkwhisper-test-{nanos}"))
    }

    fn create_model_dir(model_dir: &Path) {
        fs::create_dir_all(model_dir).unwrap();
        for file in [
            "tokenizer.json",
            "mel.rknn",
            "encoder.rknn",
            "enc_kv.rknn",
            "decoder.rknn",
        ] {
            File::create(model_dir.join(file)).unwrap();
        }
    }
}
