use anyhow::{Context, Result, bail};
use clap::Parser;
use rknpu2::utils::find_rknn_library;
use rkwhisper::{
    N_SAMPLES,
    daemon::{
        DEFAULT_CONFIG_PATH, DEFAULT_SOCKET_PATH, DaemonConfig, ModelFiles, ModelKind,
        RequestHeader, default_model_root, load_config, pcm_s16le_to_f32,
        resolve_enabled_model_files,
    },
    parallel::{LiveTranscriptionStats, LiveWindow, ParallelModelPaths, ParallelTranscriberPool},
    protocol::{
        RING_DATA_BYTES, RING_HEADER_BYTES, Response, SIGNAL_CANCEL, SIGNAL_DATA_READY,
        SIGNAL_END_OF_STREAM, ServerHello, SharedAudioRing, read_client_hello,
        supported_audio_format, validate_client_hello, write_response,
    },
    spec::{
        WhisperBase, WhisperLargeV3Turbo, WhisperMedium, WhisperSmall, WhisperSpec, WhisperTiny,
    },
    suppression::SuppressTokens,
    whisper::{TranscribeOptions, Transcription},
};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Read};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

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
    Ok(listener)
}

fn handle_connection(mut stream: UnixStream, pools: &mut DaemonPools, lib: &Path) -> Result<()> {
    let hello = match read_client_hello(&mut stream) {
        Ok(hello) => hello,
        Err(error) => {
            write_error(&mut stream, &error.to_string())?;
            return Ok(());
        }
    };
    if let Err(error) = validate_client_hello(&hello) {
        write_error(&mut stream, &error.to_string())?;
        return Ok(());
    }

    let header = request_header_from_hello(hello);
    let ring = match SharedAudioRing::create(RING_DATA_BYTES) {
        Ok(ring) => ring,
        Err(error) => {
            write_error(&mut stream, &error.to_string())?;
            return Ok(());
        }
    };
    ring.send_fd(&stream)?;
    write_response(
        &mut stream,
        Response::ServerHello(ServerHello {
            audio_format: supported_audio_format(),
            ring_capacity_bytes: ring.capacity() as u64,
            ring_header_bytes: RING_HEADER_BYTES as u32,
        }),
    )?;

    let reader = stream
        .try_clone()
        .context("failed to clone stream for live reader")?;
    let mut writer = BufWriter::new(stream);
    let started = Instant::now();
    let (window_tx, window_rx) = mpsc::channel::<Result<LiveWindow>>(4);

    let reader = std::thread::Builder::new()
        .name("rkwhisper-live-reader".to_string())
        .spawn(move || read_live_windows(reader, ring, window_tx))
        .context("failed to spawn live stream reader")?;

    let result = pools.transcribe_live_stream(lib, &header, window_rx, |segment| {
        write_response(
            &mut writer,
            Response::Segment {
                text: segment.text.clone(),
                begin: segment.start_sec,
                end: segment.end_sec,
            },
        )?;
        Ok(())
    });

    let (total_samples, total_windows) = match reader.join() {
        Ok(Ok(stats)) => stats,
        Ok(Err(error)) => {
            write_response(
                &mut writer,
                Response::Error {
                    error: error.to_string(),
                },
            )?;
            return Ok(());
        }
        Err(_) => {
            write_response(
                &mut writer,
                Response::Error {
                    error: "live stream reader thread panicked".to_string(),
                },
            )?;
            return Ok(());
        }
    };

    match result {
        Ok((_transcription, stats)) => {
            eprintln!(
                "stream completed: produced={} dispatched={} completed={}",
                total_windows, stats.windows_dispatched, stats.windows_completed
            );
            write_response(
                &mut writer,
                Response::Done {
                    audio_s: rkwhisper::daemon::audio_seconds(total_samples),
                    rtf: rkwhisper::daemon::real_time_factor(
                        started.elapsed(),
                        rkwhisper::daemon::audio_seconds(total_samples),
                    ),
                },
            )?
        }
        Err(error) => write_response(
            &mut writer,
            Response::Error {
                error: error.to_string(),
            },
        )?,
    }
    Ok(())
}

fn read_live_windows(
    mut stream: UnixStream,
    ring: SharedAudioRing,
    window_tx: mpsc::Sender<Result<LiveWindow>>,
) -> Result<(usize, usize)> {
    let mut pcm = Vec::<u8>::new();
    let mut total_samples = 0usize;
    let mut total_windows = 0usize;
    let mut next_window_index = 0usize;
    let mut next_window_start = 0usize;

    loop {
        let mut signal = [0u8; 1];
        let n = stream
            .read(&mut signal)
            .context("failed to read shared-memory signal")?;
        if n == 0 {
            flush_pcm_windows(
                &window_tx,
                &mut pcm,
                &mut total_samples,
                &mut total_windows,
                &mut next_window_index,
                &mut next_window_start,
                true,
            )?;
            break;
        }

        match signal[0] {
            SIGNAL_DATA_READY => {
                ring.drain_available(&mut pcm);
                flush_pcm_windows(
                    &window_tx,
                    &mut pcm,
                    &mut total_samples,
                    &mut total_windows,
                    &mut next_window_index,
                    &mut next_window_start,
                    false,
                )?;
            }
            SIGNAL_END_OF_STREAM => {
                ring.drain_available(&mut pcm);
                flush_pcm_windows(
                    &window_tx,
                    &mut pcm,
                    &mut total_samples,
                    &mut total_windows,
                    &mut next_window_index,
                    &mut next_window_start,
                    true,
                )?;
                break;
            }
            SIGNAL_CANCEL => bail!("request canceled"),
            other => {
                let message = format!("unsupported shared-memory signal {other}");
                let _ = window_tx.blocking_send(Err(anyhow::anyhow!(message.clone())));
                bail!("{message}");
            }
        }
    }

    Ok((total_samples, total_windows))
}

fn flush_pcm_windows(
    window_tx: &mpsc::Sender<Result<LiveWindow>>,
    pcm: &mut Vec<u8>,
    total_samples: &mut usize,
    total_windows: &mut usize,
    next_window_index: &mut usize,
    next_window_start: &mut usize,
    final_flush: bool,
) -> Result<()> {
    let window_bytes = N_SAMPLES * 2;
    while pcm.len() >= window_bytes {
        let chunk = pcm.drain(..window_bytes).collect::<Vec<_>>();
        let samples = pcm_s16le_to_f32(&chunk)?;
        *total_samples += samples.len();
        send_live_window(window_tx, *next_window_index, *next_window_start, samples)?;
        *total_windows += 1;
        *next_window_index += 1;
        *next_window_start += N_SAMPLES;
    }

    if final_flush && !pcm.is_empty() {
        let chunk = std::mem::take(pcm);
        let samples = pcm_s16le_to_f32(&chunk)?;
        *total_samples += samples.len();
        send_live_window(window_tx, *next_window_index, *next_window_start, samples)?;
        *total_windows += 1;
    }
    Ok(())
}

fn send_live_window(
    window_tx: &mpsc::Sender<Result<LiveWindow>>,
    index: usize,
    start_sample: usize,
    samples: Vec<f32>,
) -> Result<()> {
    let end_sample = start_sample + samples.len();
    window_tx
        .blocking_send(Ok(LiveWindow {
            index,
            start_sample,
            end_sample,
            samples,
        }))
        .map_err(|_| anyhow::anyhow!("live stream worker stopped"))?;
    Ok(())
}

fn write_error(stream: &mut UnixStream, error: &str) -> Result<()> {
    write_response(
        stream,
        Response::Error {
            error: error.to_string(),
        },
    )
}

fn request_header_from_hello(hello: rkwhisper::protocol::ClientHello) -> RequestHeader {
    RequestHeader {
        model: hello.model,
        mode: hello.mode,
        lang: hello.lang,
        task: hello.task,
        max_new_tokens: hello.max_new_tokens,
        beam_size: hello.beam_size,
        notimestamps: hello.notimestamps,
        suppress_tokens: hello.suppress_tokens,
        vad_threshold: hello.vad.threshold,
        vad_min_speech_ms: hello.vad.min_speech_ms,
        vad_min_silence_ms: hello.vad.min_silence_ms,
        vad_speech_pad_ms: hello.vad.speech_pad_ms,
        vad_window_samples: hello.vad.window_samples,
    }
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

    fn transcribe_live_stream<F>(
        &mut self,
        lib: &Path,
        header: &RequestHeader,
        window_rx: mpsc::Receiver<Result<LiveWindow>>,
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        let pool = self
            .pools
            .get_mut(&header.model)
            .ok_or_else(|| anyhow::anyhow!("model not found"))?;
        pool.transcribe_live_stream(lib, header, window_rx, on_segment)
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

    fn transcribe_live_stream<F>(
        &mut self,
        lib: &Path,
        header: &RequestHeader,
        window_rx: mpsc::Receiver<Result<LiveWindow>>,
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        match self {
            Self::Tiny(pool) => pool.transcribe_live_stream(lib, header, window_rx, on_segment),
            Self::Base(pool) => pool.transcribe_live_stream(lib, header, window_rx, on_segment),
            Self::Small(pool) => pool.transcribe_live_stream(lib, header, window_rx, on_segment),
            Self::Medium(pool) => pool.transcribe_live_stream(lib, header, window_rx, on_segment),
            Self::LargeV3Turbo(pool) => {
                pool.transcribe_live_stream(lib, header, window_rx, on_segment)
            }
        }
    }
}

struct TypedModelPool<S: WhisperSpec + Send + 'static> {
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
            tokenizer: Arc::new(tokenizer),
            pool,
        })
    }

    fn transcribe_live_stream<F>(
        &mut self,
        _lib: &Path,
        header: &RequestHeader,
        window_rx: mpsc::Receiver<Result<LiveWindow>>,
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        let options = TranscribeOptions::new(
            header.lang.clone(),
            header.task.clone(),
            header.notimestamps,
            header.max_new_tokens,
            header.beam_size,
            SuppressTokens::parse(&header.suppress_tokens)?,
        );

        self.pool.transcribe_live_windows_with_callback(
            window_rx,
            self.tokenizer.clone(),
            &options,
            on_segment,
        )
    }
}
