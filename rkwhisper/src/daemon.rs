use crate::SAMPLE_RATE;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const DEFAULT_MODEL_ROOT: &str = "/usr/share/rkwhisper";
pub const MODEL_ROOT_ENV: &str = "RKWHISPER_MODEL_ROOT";
pub const DEFAULT_SOCKET_PATH: &str = "/run/rkwhisper/asr.sock";
pub const DEFAULT_CONFIG_PATH: &str = "/etc/rkwhisper.toml";
pub const MAX_HEADER_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct RequestHeader {
    pub model: String,
    pub mode: String,
    pub lang: String,
    pub task: String,
    pub max_new_tokens: usize,
    pub beam_size: usize,
    pub notimestamps: bool,
    pub suppress_tokens: String,
    pub vad_threshold: Option<f32>,
    pub vad_min_speech_ms: Option<u32>,
    pub vad_min_silence_ms: Option<u32>,
    pub vad_speech_pad_ms: Option<u32>,
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
    #[serde(default)]
    pub concurrency: ConcurrencyConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConcurrencyConfig {
    #[serde(default = "default_model_queue_depth")]
    pub model_queue_depth: usize,
    #[serde(default = "default_max_active_jobs_per_model")]
    pub max_active_jobs_per_model: usize,
    #[serde(default = "default_max_in_flight_windows_per_job")]
    pub max_in_flight_windows_per_job: usize,
    #[serde(default = "default_client_window_queue_depth")]
    pub client_window_queue_depth: usize,
    #[serde(default = "default_client_response_queue_depth")]
    pub client_response_queue_depth: usize,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            model_queue_depth: default_model_queue_depth(),
            max_active_jobs_per_model: default_max_active_jobs_per_model(),
            max_in_flight_windows_per_job: default_max_in_flight_windows_per_job(),
            client_window_queue_depth: default_client_window_queue_depth(),
            client_response_queue_depth: default_client_response_queue_depth(),
        }
    }
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
    validate_concurrency_config(&config.concurrency)?;

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

fn validate_concurrency_config(config: &ConcurrencyConfig) -> Result<()> {
    if config.model_queue_depth == 0 {
        bail!("concurrency.model_queue_depth must be at least 1");
    }
    if config.max_active_jobs_per_model == 0 {
        bail!("concurrency.max_active_jobs_per_model must be at least 1");
    }
    if config.max_in_flight_windows_per_job == 0 {
        bail!("concurrency.max_in_flight_windows_per_job must be at least 1");
    }
    if config.client_window_queue_depth == 0 {
        bail!("concurrency.client_window_queue_depth must be at least 1");
    }
    if config.client_response_queue_depth == 0 {
        bail!("concurrency.client_response_queue_depth must be at least 1");
    }
    Ok(())
}

fn default_model_queue_depth() -> usize {
    1
}

fn default_max_active_jobs_per_model() -> usize {
    1
}

fn default_max_in_flight_windows_per_job() -> usize {
    1
}

fn default_client_window_queue_depth() -> usize {
    4
}

fn default_client_response_queue_depth() -> usize {
    16
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::time::{SystemTime, UNIX_EPOCH};

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
        assert_eq!(config.concurrency.model_queue_depth, 1);
        assert_eq!(config.concurrency.max_active_jobs_per_model, 1);
        assert_eq!(config.concurrency.max_in_flight_windows_per_job, 1);
        assert_eq!(config.concurrency.client_window_queue_depth, 4);
        assert_eq!(config.concurrency.client_response_queue_depth, 16);
    }

    #[test]
    fn parses_concurrency_config() {
        let config = parse_config(
            r#"
models = ["whisper-small-30s"]

[concurrency]
model_queue_depth = 2
max_active_jobs_per_model = 3
max_in_flight_windows_per_job = 2
client_window_queue_depth = 8
client_response_queue_depth = 32
"#,
        )
        .unwrap();

        assert_eq!(config.concurrency.model_queue_depth, 2);
        assert_eq!(config.concurrency.max_active_jobs_per_model, 3);
        assert_eq!(config.concurrency.max_in_flight_windows_per_job, 2);
        assert_eq!(config.concurrency.client_window_queue_depth, 8);
        assert_eq!(config.concurrency.client_response_queue_depth, 32);
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
        assert!(
            parse_config(
                r#"
models = ["whisper-small-30s"]

[concurrency]
model_queue_depth = 0
"#
            )
            .unwrap_err()
            .to_string()
            .contains("model_queue_depth")
        );
        assert!(
            parse_config(
                r#"
models = ["whisper-small-30s"]

[concurrency]
max_active_jobs_per_model = 0
"#
            )
            .unwrap_err()
            .to_string()
            .contains("max_active_jobs_per_model")
        );
        assert!(
            parse_config(
                r#"
models = ["whisper-small-30s"]

[concurrency]
max_in_flight_windows_per_job = 0
"#
            )
            .unwrap_err()
            .to_string()
            .contains("max_in_flight_windows_per_job")
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
