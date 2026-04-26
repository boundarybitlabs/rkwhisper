use anyhow::{Context, Result, bail};
use clap::Parser;
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    daemon::{
        DEFAULT_CONFIG_PATH, DEFAULT_SOCKET_PATH, DaemonConfig, DaemonRequest, DaemonResponse,
        ModelFiles, ModelKind, default_model_root, load_config, resolve_enabled_model_files,
        response_line,
    },
    parallel::{ParallelModelPaths, ParallelTranscriberPool},
    spec::{
        WhisperBase, WhisperLargeV3Turbo, WhisperMedium, WhisperSmall, WhisperSpec, WhisperTiny,
    },
    suppression::SuppressTokens,
    vad::{VadConfig, VadModel},
    whisper::{TranscribeOptions, Transcription},
};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::{BufWriter, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(version, about = "RKWhisper Unix socket ASR daemon")]
struct Args {
    /// Unix domain socket path.
    #[arg(long, default_value = DEFAULT_SOCKET_PATH)]
    socket: PathBuf,

    /// Model root. Defaults to RKWHISPER_MODEL_ROOT or /usr/share/rkwhisper.
    #[arg(long)]
    model_root: Option<PathBuf>,

    /// Daemon config listing enabled models.
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    config: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_root = args.model_root.unwrap_or_else(default_model_root);
    let config = load_config(&args.config)?;
    let lib = find_rknn_library()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Could not find rknn library"))?;
    let mut pools = DaemonPools::load(&model_root, &config, &lib)?;

    let listener = bind_socket(&args.socket)?;
    eprintln!(
        "rkwhisperd listening on {} with model root {} and {} enabled model pool(s)",
        args.socket.display(),
        model_root.display(),
        pools.len()
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(error) = handle_connection(stream, &mut pools, &lib) {
                    eprintln!("request failed: {error:#}");
                }
            }
            Err(error) => eprintln!("accept failed: {error}"),
        }
    }

    Ok(())
}

fn bind_socket(socket: &Path) -> Result<UnixListener> {
    if let Some(parent) = socket.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket directory {}", parent.display()))?;
    }

    if let Ok(metadata) = fs::symlink_metadata(socket) {
        if metadata.file_type().is_socket() {
            fs::remove_file(socket)
                .with_context(|| format!("failed to remove stale socket {}", socket.display()))?;
        } else {
            bail!(
                "socket path exists and is not a socket: {}",
                socket.display()
            );
        }
    }

    let listener = UnixListener::bind(socket)
        .with_context(|| format!("failed to bind socket {}", socket.display()))?;
    fs::set_permissions(socket, fs::Permissions::from_mode(0o660))
        .with_context(|| format!("failed to chmod socket {}", socket.display()))?;
    chown_socket(socket, "rkwhisper")
        .with_context(|| format!("failed to chown socket {}", socket.display()))?;
    Ok(listener)
}

fn chown_socket(socket: &Path, group_name: &str) -> Result<()> {
    let gid = group_id(group_name)?.with_context(|| format!("group {group_name:?} not found"))?;
    let c_path = CString::new(socket.as_os_str().as_bytes())
        .context("socket path contains an interior NUL byte")?;
    let rc = unsafe { libc::chown(c_path.as_ptr(), 0, gid) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("chown root:rkwhisper failed");
    }
    Ok(())
}

fn group_id(name: &str) -> Result<Option<u32>> {
    let groups = fs::read_to_string("/etc/group").context("failed to read /etc/group")?;
    for line in groups.lines() {
        let mut parts = line.split(':');
        let Some(group_name) = parts.next() else {
            continue;
        };
        if group_name != name {
            continue;
        }
        let gid = parts
            .nth(1)
            .context("malformed group entry")?
            .parse::<u32>()
            .context("group gid is not a number")?;
        return Ok(Some(gid));
    }
    Ok(None)
}

fn handle_connection(mut stream: UnixStream, pools: &mut DaemonPools, lib: &Path) -> Result<()> {
    let request = match rkwhisper::daemon::read_request(&mut stream) {
        Ok(request) => request,
        Err(error) => {
            write_error(&mut stream, &error.to_string())?;
            return Ok(());
        }
    };

    let mut writer = BufWriter::new(stream);
    let started = Instant::now();
    let audio_s = rkwhisper::daemon::audio_seconds(request.audio.len());

    match pools.transcribe_request(lib, &request) {
        Ok(transcription) => {
            for segment in &transcription.segments {
                writer.write_all(
                    response_line(&DaemonResponse::Segment {
                        text: &segment.text,
                        begin: segment.start_sec,
                        end: segment.end_sec,
                    })?
                    .as_bytes(),
                )?;
            }
            writer.write_all(
                response_line(&DaemonResponse::Done {
                    audio_s,
                    rtf: rkwhisper::daemon::real_time_factor(started.elapsed(), audio_s),
                })?
                .as_bytes(),
            )?;
        }
        Err(error) => {
            writer.write_all(
                response_line(&DaemonResponse::Error {
                    error: &error.to_string(),
                })?
                .as_bytes(),
            )?;
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_error(stream: &mut UnixStream, error: &str) -> Result<()> {
    stream.write_all(response_line(&DaemonResponse::Error { error })?.as_bytes())?;
    stream.flush()?;
    Ok(())
}

struct DaemonPools {
    pools: HashMap<String, ModelPool>,
}

impl DaemonPools {
    fn load(model_root: &Path, config: &DaemonConfig, lib: &Path) -> Result<Self> {
        let mut pools = HashMap::new();
        for model_id in &config.models {
            let files = resolve_enabled_model_files(model_root, config, model_id)
                .with_context(|| format!("failed to resolve model {model_id}"))?;
            let pool = ModelPool::load(lib, files)
                .with_context(|| format!("failed to load model pool {model_id}"))?;
            pools.insert(model_id.clone(), pool);
        }
        Ok(Self { pools })
    }

    fn len(&self) -> usize {
        self.pools.len()
    }

    fn transcribe_request(&mut self, lib: &Path, request: &DaemonRequest) -> Result<Transcription> {
        let pool = self
            .pools
            .get_mut(&request.header.model)
            .ok_or_else(|| anyhow::anyhow!("model not found"))?;
        pool.transcribe(lib, request)
    }
}

enum ModelPool {
    Tiny(TypedModelPool<WhisperTiny>),
    Base(TypedModelPool<WhisperBase>),
    Small(TypedModelPool<WhisperSmall>),
    Medium(TypedModelPool<WhisperMedium>),
    LargeV3Turbo(TypedModelPool<WhisperLargeV3Turbo>),
}

impl ModelPool {
    fn load(lib: &Path, files: ModelFiles) -> Result<Self> {
        match files.kind {
            ModelKind::Tiny => Ok(Self::Tiny(TypedModelPool::<WhisperTiny>::load(lib, files)?)),
            ModelKind::Base => Ok(Self::Base(TypedModelPool::<WhisperBase>::load(lib, files)?)),
            ModelKind::Small => Ok(Self::Small(TypedModelPool::<WhisperSmall>::load(
                lib, files,
            )?)),
            ModelKind::Medium => Ok(Self::Medium(TypedModelPool::<WhisperMedium>::load(
                lib, files,
            )?)),
            ModelKind::LargeV3Turbo => Ok(Self::LargeV3Turbo(
                TypedModelPool::<WhisperLargeV3Turbo>::load(lib, files)?,
            )),
        }
    }

    fn transcribe(&mut self, lib: &Path, request: &DaemonRequest) -> Result<Transcription> {
        match self {
            Self::Tiny(pool) => pool.transcribe(lib, request),
            Self::Base(pool) => pool.transcribe(lib, request),
            Self::Small(pool) => pool.transcribe(lib, request),
            Self::Medium(pool) => pool.transcribe(lib, request),
            Self::LargeV3Turbo(pool) => pool.transcribe(lib, request),
        }
    }
}

struct TypedModelPool<S: WhisperSpec + Send + 'static> {
    files: ModelFiles,
    tokenizer: Arc<Tokenizer>,
    pool: ParallelTranscriberPool<S>,
}

impl<S: WhisperSpec + Send + 'static> TypedModelPool<S> {
    fn load(lib: &Path, files: ModelFiles) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let model_paths = ParallelModelPaths::new(
            files.mel.clone(),
            files.encoder.clone(),
            files.enc_kv.clone(),
            files.decoder.clone(),
        );
        let pool = ParallelTranscriberPool::<S>::new(lib, &model_paths)?;
        Ok(Self {
            files,
            tokenizer: Arc::new(tokenizer),
            pool,
        })
    }

    fn transcribe(&mut self, lib: &Path, request: &DaemonRequest) -> Result<Transcription> {
        let vad = if let Some(path) = &self.files.vad {
            let config = vad_config_from_request(request);
            Some(VadModel::new(
                RKNN::new_with_library(lib, &mut fs::read(path)?, 0)?,
                config,
            ))
        } else {
            None
        };

        let options = TranscribeOptions::new(
            request.header.lang.clone(),
            request.header.task.clone(),
            request.header.notimestamps,
            request.header.max_new_tokens,
            request.header.beam_size,
            SuppressTokens::parse(&request.header.suppress_tokens)?,
        );

        self.pool.transcribe_audio_with_options(
            &request.audio,
            self.tokenizer.clone(),
            vad.as_ref(),
            &options,
        )
    }
}

fn vad_config_from_request(request: &DaemonRequest) -> VadConfig {
    let mut config = VadConfig::default();
    if let Some(value) = request.header.vad_threshold {
        config.threshold = value;
    }
    if let Some(value) = request.header.vad_min_speech_ms {
        config.min_speech_ms = value;
    }
    if let Some(value) = request.header.vad_min_silence_ms {
        config.min_silence_ms = value;
    }
    if let Some(value) = request.header.vad_speech_pad_ms {
        config.speech_pad_ms = value;
    }
    if let Some(value) = request.header.vad_window_samples {
        config.window_samples = value;
    }
    config
}
