use anyhow::{Context, Result, bail};
use clap::Parser;
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    N_SAMPLES,
    daemon::{
        ConcurrencyConfig, DEFAULT_CONFIG_PATH, DEFAULT_SOCKET_PATH, DaemonConfig, ModelFiles,
        ModelKind, RequestHeader, default_model_root, load_config, pcm_s16le_to_f32,
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
    vad::{VadConfig, VadModel},
    whisper::{TranscribeOptions, Transcription},
};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Read};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

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
    let schedulers = ModelSchedulers::load(&model_root, &config, &lib)?;

    let listener = bind_socket(&args.socket)?;
    eprintln!(
        "rkwhisperd listening on {} with model root {} and {} enabled model pool(s)",
        args.socket.display(),
        model_root.display(),
        schedulers.len()
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let schedulers = schedulers.clone();
                let concurrency = config.concurrency.clone();
                std::thread::Builder::new()
                    .name("rkwhisper-session".to_string())
                    .spawn(move || {
                        if let Err(error) = handle_connection(stream, schedulers, concurrency) {
                            eprintln!("request failed: {error:#}");
                        }
                    })
                    .context("failed to spawn session thread")?;
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

fn handle_connection(
    mut stream: UnixStream,
    schedulers: ModelSchedulers,
    concurrency: ConcurrencyConfig,
) -> Result<()> {
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
    if !schedulers.contains_model(&header.model) {
        write_error(&mut stream, "model not found")?;
        return Ok(());
    }

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

    if header.mode == "batch" {
        return handle_batch_connection(&mut writer, reader, ring, schedulers, header, started);
    }

    let fail_fast_windows = true;
    let (window_tx, window_rx) =
        mpsc::channel::<Result<LiveWindow>>(concurrency.client_window_queue_depth);
    let (response_tx, response_rx) =
        std_mpsc::sync_channel::<JobResponse>(concurrency.client_response_queue_depth);

    if let Err(error) = schedulers.submit(header.clone(), window_rx, response_tx) {
        let response = match error {
            SubmitError::QueueFull(reason) => Response::BackOff {
                reason,
                retry_after_ms: 250,
            },
            SubmitError::UnknownModel => Response::Error {
                error: "model not found".to_string(),
            },
            SubmitError::SchedulerStopped(reason) => Response::Error { error: reason },
        };
        write_response(&mut writer, response)?;
        return Ok(());
    }

    let reader = std::thread::Builder::new()
        .name("rkwhisper-live-reader".to_string())
        .spawn(move || read_live_windows(reader, ring, window_tx, fail_fast_windows))
        .context("failed to spawn live stream reader")?;

    while let Ok(response) = response_rx.recv() {
        match response {
            JobResponse::Segment { text, begin, end } => {
                write_response(&mut writer, Response::Segment { text, begin, end })?;
            }
            JobResponse::Finished(result) => {
                let read_result = match reader.join() {
                    Ok(result) => result,
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
                write_final_response(&mut writer, started, read_result, result)?;
                return Ok(());
            }
        }
    }

    Ok(())
}

fn handle_batch_connection(
    writer: &mut BufWriter<UnixStream>,
    mut reader_stream: UnixStream,
    ring: SharedAudioRing,
    schedulers: ModelSchedulers,
    header: RequestHeader,
    started: Instant,
) -> Result<()> {
    let mut pcm = Vec::<u8>::new();
    loop {
        let mut signal = [0u8; 1];
        let n = reader_stream
            .read(&mut signal)
            .context("failed to read shared-memory signal")?;
        if n == 0 {
            break;
        }
        match signal[0] {
            SIGNAL_DATA_READY => {
                ring.drain_available(&mut pcm)?;
            }
            SIGNAL_END_OF_STREAM => {
                ring.drain_available(&mut pcm)?;
                break;
            }
            SIGNAL_CANCEL => return Ok(()),
            _ => {}
        }
    }

    let audio = pcm_s16le_to_f32(&pcm)?;
    let audio_s = rkwhisper::daemon::audio_seconds(audio.len());

    let (response_tx, response_rx) = std_mpsc::sync_channel::<JobResponse>(128);

    if let Err(error) = schedulers.submit_batch(header.clone(), audio, response_tx) {
        let response = match error {
            SubmitError::QueueFull(reason) => Response::BackOff {
                reason,
                retry_after_ms: 250,
            },
            SubmitError::UnknownModel => Response::Error {
                error: "model not found".to_string(),
            },
            SubmitError::SchedulerStopped(reason) => Response::Error { error: reason },
        };
        write_response(writer, response)?;
        return Ok(());
    }

    while let Ok(response) = response_rx.recv() {
        match response {
            JobResponse::Segment { text, begin, end } => {
                write_response(writer, Response::Segment { text, begin, end })?;
            }
            JobResponse::Finished(result) => {
                match result {
                    Ok(stats) => {
                        eprintln!(
                            "batch completed: dispatched={} completed={}",
                            stats.windows_dispatched, stats.windows_completed
                        );
                        write_response(
                            writer,
                            Response::Done {
                                audio_s,
                                rtf: rkwhisper::daemon::real_time_factor(
                                    started.elapsed(),
                                    audio_s,
                                ),
                            },
                        )?;
                    }
                    Err(error) => {
                        write_response(
                            writer,
                            Response::Error {
                                error: error.to_string(),
                            },
                        )?;
                    }
                }
                return Ok(());
            }
        }
    }

    Ok(())
}

fn write_final_response(
    writer: &mut BufWriter<UnixStream>,
    started: Instant,
    read_result: Result<ReadOutcome>,
    transcribe_result: Result<LiveTranscriptionStats>,
) -> Result<()> {
    let outcome = match read_result {
        Ok(outcome) => outcome,
        Err(error) => {
            let error = error.to_string();
            if error.contains("client window queue full") {
                write_response(
                    writer,
                    Response::BackOff {
                        reason: error,
                        retry_after_ms: 250,
                    },
                )?;
            } else {
                write_response(writer, Response::Error { error })?;
            }
            return Ok(());
        }
    };

    let live_stats = match transcribe_result {
        Ok(stats) => stats,
        Err(error) => {
            write_response(
                writer,
                Response::Error {
                    error: error.to_string(),
                },
            )?;
            return Ok(());
        }
    };

    let read_stats = outcome.stats();
    eprintln!(
        "stream completed: produced={} dispatched={} completed={}",
        read_stats.total_windows, live_stats.windows_dispatched, live_stats.windows_completed
    );
    let audio_s = rkwhisper::daemon::audio_seconds(read_stats.total_samples);
    let rtf = rkwhisper::daemon::real_time_factor(started.elapsed(), audio_s);
    match outcome {
        ReadOutcome::Completed(_) => write_response(writer, Response::Done { audio_s, rtf })?,
        ReadOutcome::Cancelled(_) => write_response(
            writer,
            Response::Cancelled {
                audio_s,
                rtf,
                windows_dispatched: live_stats.windows_dispatched as u64,
                windows_completed: live_stats.windows_completed as u64,
            },
        )?,
    }
    Ok(())
}

fn read_live_windows(
    mut stream: UnixStream,
    ring: SharedAudioRing,
    window_tx: mpsc::Sender<Result<LiveWindow>>,
    fail_fast_windows: bool,
) -> Result<ReadOutcome> {
    let mut pcm = Vec::<u8>::new();
    let mut stats = StreamReadStats::default();
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
                &mut stats,
                &mut next_window_index,
                &mut next_window_start,
                true,
                fail_fast_windows,
            )?;
            break;
        }

        match signal[0] {
            SIGNAL_DATA_READY => {
                ring.drain_available(&mut pcm)?;
                flush_pcm_windows(
                    &window_tx,
                    &mut pcm,
                    &mut stats,
                    &mut next_window_index,
                    &mut next_window_start,
                    false,
                    fail_fast_windows,
                )?;
            }
            SIGNAL_END_OF_STREAM => {
                ring.drain_available(&mut pcm)?;
                flush_pcm_windows(
                    &window_tx,
                    &mut pcm,
                    &mut stats,
                    &mut next_window_index,
                    &mut next_window_start,
                    true,
                    fail_fast_windows,
                )?;
                break;
            }
            SIGNAL_CANCEL => return Ok(ReadOutcome::Cancelled(stats)),
            other => {
                let message = format!("unsupported shared-memory signal {other}");
                let _ = window_tx.blocking_send(Err(anyhow::anyhow!(message.clone())));
                bail!("{message}");
            }
        }
    }

    Ok(ReadOutcome::Completed(stats))
}

fn flush_pcm_windows(
    window_tx: &mpsc::Sender<Result<LiveWindow>>,
    pcm: &mut Vec<u8>,
    stats: &mut StreamReadStats,
    next_window_index: &mut usize,
    next_window_start: &mut usize,
    final_flush: bool,
    fail_fast_windows: bool,
) -> Result<()> {
    let window_bytes = N_SAMPLES * 2;
    while pcm.len() >= window_bytes {
        let chunk = pcm.drain(..window_bytes).collect::<Vec<_>>();
        let samples = pcm_s16le_to_f32(&chunk)?;
        stats.total_samples += samples.len();
        send_live_window(
            window_tx,
            *next_window_index,
            *next_window_start,
            samples,
            fail_fast_windows,
        )?;
        stats.total_windows += 1;
        *next_window_index += 1;
        *next_window_start += N_SAMPLES;
    }

    if final_flush && !pcm.is_empty() {
        let chunk = std::mem::take(pcm);
        let samples = pcm_s16le_to_f32(&chunk)?;
        stats.total_samples += samples.len();
        send_live_window(
            window_tx,
            *next_window_index,
            *next_window_start,
            samples,
            fail_fast_windows,
        )?;
        stats.total_windows += 1;
    }
    Ok(())
}

fn send_live_window(
    window_tx: &mpsc::Sender<Result<LiveWindow>>,
    index: usize,
    start_sample: usize,
    samples: Vec<f32>,
    fail_fast: bool,
) -> Result<()> {
    let end_sample = start_sample + samples.len();
    let window = Ok(LiveWindow {
        index,
        start_sample,
        end_sample,
        samples,
    });
    if fail_fast {
        window_tx.try_send(window).map_err(|error| match error {
            TrySendError::Full(_) => anyhow::anyhow!("client window queue full"),
            TrySendError::Closed(_) => anyhow::anyhow!("live stream worker stopped"),
        })?;
    } else {
        window_tx
            .blocking_send(window)
            .map_err(|_| anyhow::anyhow!("live stream worker stopped"))?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct StreamReadStats {
    total_samples: usize,
    total_windows: usize,
}

#[derive(Clone, Copy, Debug)]
enum ReadOutcome {
    Completed(StreamReadStats),
    Cancelled(StreamReadStats),
}

impl ReadOutcome {
    fn stats(self) -> StreamReadStats {
        match self {
            Self::Completed(stats) | Self::Cancelled(stats) => stats,
        }
    }
}

enum JobResponse {
    Segment { text: String, begin: f32, end: f32 },
    Finished(Result<LiveTranscriptionStats>),
}

enum ModelJob {
    Live {
        header: RequestHeader,
        window_rx: mpsc::Receiver<Result<LiveWindow>>,
        response_tx: std_mpsc::SyncSender<JobResponse>,
    },
    Batch {
        header: RequestHeader,
        audio: Vec<f32>,
        response_tx: std_mpsc::SyncSender<JobResponse>,
    },
}

#[derive(Clone)]
struct ModelSchedulers {
    schedulers: Arc<HashMap<String, mpsc::Sender<ModelJob>>>,
}

impl ModelSchedulers {
    fn load(model_root: &Path, config: &DaemonConfig, lib: &Path) -> Result<Self> {
        let mut schedulers = HashMap::new();
        for model_id in &config.models {
            let files = resolve_enabled_model_files(model_root, config, model_id)
                .with_context(|| format!("failed to resolve model {model_id}"))?;
            let pool = ModelPool::load(lib, files)
                .with_context(|| format!("failed to load model pool {model_id}"))?;
            let (job_tx, job_rx) = mpsc::channel::<ModelJob>(config.concurrency.model_queue_depth);
            spawn_model_scheduler(model_id.clone(), pool, job_rx)?;
            schedulers.insert(model_id.clone(), job_tx);
        }
        Ok(Self {
            schedulers: Arc::new(schedulers),
        })
    }

    fn len(&self) -> usize {
        self.schedulers.len()
    }

    fn contains_model(&self, model: &str) -> bool {
        self.schedulers.contains_key(model)
    }

    fn submit(
        &self,
        header: RequestHeader,
        window_rx: mpsc::Receiver<Result<LiveWindow>>,
        response_tx: std_mpsc::SyncSender<JobResponse>,
    ) -> Result<(), SubmitError> {
        let job_tx = self
            .schedulers
            .get(&header.model)
            .ok_or(SubmitError::UnknownModel)?;
        job_tx
            .try_send(ModelJob::Live {
                header,
                window_rx,
                response_tx,
            })
            .map_err(|error| match error {
                TrySendError::Full(_) => SubmitError::QueueFull("model queue full".to_string()),
                TrySendError::Closed(_) => {
                    SubmitError::SchedulerStopped("model scheduler stopped".to_string())
                }
            })
    }

    fn submit_batch(
        &self,
        header: RequestHeader,
        audio: Vec<f32>,
        response_tx: std_mpsc::SyncSender<JobResponse>,
    ) -> Result<(), SubmitError> {
        let job_tx = self
            .schedulers
            .get(&header.model)
            .ok_or(SubmitError::UnknownModel)?;
        job_tx
            .try_send(ModelJob::Batch {
                header,
                audio,
                response_tx,
            })
            .map_err(|error| match error {
                TrySendError::Full(_) => SubmitError::QueueFull("model queue full".to_string()),
                TrySendError::Closed(_) => {
                    SubmitError::SchedulerStopped("model scheduler stopped".to_string())
                }
            })
    }
}

#[derive(Debug)]
enum SubmitError {
    QueueFull(String),
    UnknownModel,
    SchedulerStopped(String),
}

fn spawn_model_scheduler(
    model_id: String,
    mut pool: ModelPool,
    mut job_rx: mpsc::Receiver<ModelJob>,
) -> Result<()> {
    std::thread::Builder::new()
        .name(format!("rkwhisper-scheduler-{model_id}"))
        .spawn(move || {
            while let Some(job) = job_rx.blocking_recv() {
                match job {
                    ModelJob::Live {
                        header,
                        window_rx,
                        response_tx,
                    } => {
                        let segment_tx = response_tx.clone();
                        let result = pool.transcribe_live_stream(&header, window_rx, |segment| {
                            segment_tx
                                .try_send(JobResponse::Segment {
                                    text: segment.text.clone(),
                                    begin: segment.start_sec,
                                    end: segment.end_sec,
                                })
                                .map_err(|error| match error {
                                    std_mpsc::TrySendError::Full(_) => {
                                        anyhow::anyhow!("client response queue full")
                                    }
                                    std_mpsc::TrySendError::Disconnected(_) => {
                                        anyhow::anyhow!("client response channel closed")
                                    }
                                })?;
                            Ok(())
                        });
                        let result = result.map(|(_transcription, stats)| stats);
                        let _ = response_tx.send(JobResponse::Finished(result));
                    }
                    ModelJob::Batch {
                        header,
                        audio,
                        response_tx,
                    } => {
                        let segment_tx = response_tx.clone();
                        let result = pool.transcribe_batch(&header, &audio, |segment| {
                            segment_tx
                                .try_send(JobResponse::Segment {
                                    text: segment.text.clone(),
                                    begin: segment.start_sec,
                                    end: segment.end_sec,
                                })
                                .map_err(|error| match error {
                                    std_mpsc::TrySendError::Full(_) => {
                                        anyhow::anyhow!("client response queue full")
                                    }
                                    std_mpsc::TrySendError::Disconnected(_) => {
                                        anyhow::anyhow!("client response channel closed")
                                    }
                                })?;
                            Ok(())
                        });
                        let result = result.map(|(_transcription, stats)| stats);
                        let _ = response_tx.send(JobResponse::Finished(result));
                    }
                }
            }
        })
        .context("failed to spawn model scheduler")?;
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
        header: &RequestHeader,
        window_rx: mpsc::Receiver<Result<LiveWindow>>,
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        match self {
            Self::Tiny(pool) => pool.transcribe_live_stream(header, window_rx, on_segment),
            Self::Base(pool) => pool.transcribe_live_stream(header, window_rx, on_segment),
            Self::Small(pool) => pool.transcribe_live_stream(header, window_rx, on_segment),
            Self::Medium(pool) => pool.transcribe_live_stream(header, window_rx, on_segment),
            Self::LargeV3Turbo(pool) => pool.transcribe_live_stream(header, window_rx, on_segment),
        }
    }

    fn transcribe_batch<F>(
        &mut self,
        header: &RequestHeader,
        audio: &[f32],
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        match self {
            Self::Tiny(pool) => pool.transcribe_batch(header, audio, on_segment),
            Self::Base(pool) => pool.transcribe_batch(header, audio, on_segment),
            Self::Small(pool) => pool.transcribe_batch(header, audio, on_segment),
            Self::Medium(pool) => pool.transcribe_batch(header, audio, on_segment),
            Self::LargeV3Turbo(pool) => pool.transcribe_batch(header, audio, on_segment),
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

    fn transcribe_live_stream<F>(
        &mut self,
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

    fn transcribe_batch<F>(
        &mut self,
        header: &RequestHeader,
        audio: &[f32],
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        let lib = find_rknn_library()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Could not find rknn library"))?;
        let vad = if let Some(path) = &self.files.vad {
            let config = VadConfig {
                threshold: header.vad_threshold.unwrap_or(0.5),
                min_speech_ms: header.vad_min_speech_ms.unwrap_or(250),
                min_silence_ms: header.vad_min_silence_ms.unwrap_or(100),
                speech_pad_ms: header.vad_speech_pad_ms.unwrap_or(200),
                window_samples: header.vad_window_samples.unwrap_or(512),
            };
            Some(VadModel::new(
                RKNN::new_with_library(&lib, &mut std::fs::read(path)?, 0)?,
                config,
            ))
        } else {
            None
        };

        let options = TranscribeOptions::new(
            header.lang.clone(),
            header.task.clone(),
            header.notimestamps,
            header.max_new_tokens,
            header.beam_size,
            SuppressTokens::parse(&header.suppress_tokens)?,
        );

        let transcription = self.pool.transcribe_audio_with_segment_callback(
            audio,
            self.tokenizer.clone(),
            vad.as_ref(),
            &options,
            on_segment,
        )?;

        // Approximate stats for batch
        let stats = LiveTranscriptionStats {
            windows_dispatched: (audio.len() + 480000 - 1) / 480000,
            windows_completed: (audio.len() + 480000 - 1) / 480000,
        };

        Ok((transcription, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_submit_reports_unknown_model() {
        let schedulers = ModelSchedulers {
            schedulers: Arc::new(HashMap::new()),
        };
        let (_window_tx, window_rx) = mpsc::channel(1);
        let (response_tx, _response_rx) = std_mpsc::sync_channel(1);

        let error = schedulers
            .submit(test_header("missing-model"), window_rx, response_tx)
            .unwrap_err();
        assert!(matches!(error, SubmitError::UnknownModel));
    }

    #[test]
    fn scheduler_submit_reports_queue_full() {
        let (job_tx, _job_rx) = mpsc::channel(1);
        let mut map = HashMap::new();
        map.insert("whisper-small-30s".to_string(), job_tx);
        let schedulers = ModelSchedulers {
            schedulers: Arc::new(map),
        };

        let (_window_tx_1, window_rx_1) = mpsc::channel(1);
        let (response_tx_1, _response_rx_1) = std_mpsc::sync_channel(1);
        schedulers
            .submit(test_header("whisper-small-30s"), window_rx_1, response_tx_1)
            .unwrap();

        let (_window_tx_2, window_rx_2) = mpsc::channel(1);
        let (response_tx_2, _response_rx_2) = std_mpsc::sync_channel(1);
        let error = schedulers
            .submit(test_header("whisper-small-30s"), window_rx_2, response_tx_2)
            .unwrap_err();
        assert!(matches!(error, SubmitError::QueueFull(_)));
    }

    fn test_header(model: &str) -> RequestHeader {
        RequestHeader {
            model: model.to_string(),
            mode: "stream".to_string(),
            lang: "en".to_string(),
            task: "transcribe".to_string(),
            max_new_tokens: 128,
            beam_size: 5,
            notimestamps: false,
            suppress_tokens: "default".to_string(),
            vad_threshold: None,
            vad_min_speech_ms: None,
            vad_min_silence_ms: None,
            vad_speech_pad_ms: None,
            vad_window_samples: None,
        }
    }
}
