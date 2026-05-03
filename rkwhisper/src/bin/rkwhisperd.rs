use anyhow::{Context, Result, bail};
use clap::Parser;
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    daemon::{
        ConcurrencyConfig, DEFAULT_CONFIG_PATH, DEFAULT_SOCKET_PATH, DaemonConfig, ModelFiles,
        ModelKind, RequestHeader, default_model_root, load_config, pcm_s16le_to_f32,
        resolve_enabled_model_files,
    },
    parallel::{LiveTranscriptionStats, ParallelModelPaths, ParallelTranscriberPool, WhisperJob},
    protocol::{
        RING_DATA_BYTES, Response, SIGNAL_CANCEL, SIGNAL_DATA_READY, SIGNAL_END_OF_STREAM,
        ServerHello, SharedAudioRing, read_client_hello, validate_client_hello, write_response,
    },
    spec::{
        WhisperBase, WhisperLargeV3Turbo, WhisperMedium, WhisperSmall, WhisperSpec, WhisperTiny,
    },
    suppression::SuppressTokens,
    vad::{VadConfig, VadModel},
    whisper::{TranscribeOptions, Transcription, WindowTranscription},
};
use std::collections::{BTreeMap, HashMap, VecDeque};
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
            write_error(&stream, &error.to_string())?;
            return Ok(());
        }
    };
    if let Err(error) = validate_client_hello(&hello) {
        write_error(&stream, &error.to_string())?;
        return Ok(());
    }

    let client_id = hello.client_id.clone();
    let header = request_header_from_hello(hello);
    if !schedulers.contains_model(&header.model) {
        write_error(&stream, "model not found")?;
        return Ok(());
    }

    eprintln!(
        "session started: client_id={:?} model={} mode={}",
        client_id, header.model, header.mode
    );

    let ring = match SharedAudioRing::create(RING_DATA_BYTES) {
        Ok(ring) => ring,
        Err(error) => {
            write_error(&stream, &error.to_string())?;
            return Ok(());
        }
    };

    rkwhisper::protocol::send_response_with_fd(
        &stream,
        Response::ServerHello(ServerHello {
            audio_format: rkwhisper::protocol::supported_audio_format(),
            ring_capacity_bytes: ring.capacity() as u64,
            ring_header_bytes: rkwhisper::protocol::RING_HEADER_BYTES as u32,
        }),
        Some(ring.fd()),
    )?;

    let reader = stream
        .try_clone()
        .context("failed to clone stream for live reader")?;
    let mut writer = BufWriter::new(stream);
    let started = Instant::now();

    if header.mode == "batch" {
        return handle_batch_connection(&mut writer, reader, ring, schedulers, header, started);
    }

    let (window_tx, window_rx) =
        mpsc::channel::<Result<LiveChunk>>(concurrency.client_window_queue_depth);
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
        .spawn(move || read_live_chunks(reader, ring, window_tx))
        .context("failed to spawn live stream reader")?;

    let mut job_finished = false;
    while let Ok(response) = response_rx.recv() {
        match response {
            JobResponse::Segment { text, begin, end } => {
                write_response(&mut writer, Response::Segment { text, begin, end })?;
            }
            JobResponse::SpeechStarted { begin } => {
                write_response(&mut writer, Response::SpeechStarted { begin })?;
            }
            JobResponse::SpeechEnded { end } => {
                write_response(&mut writer, Response::SpeechEnded { end })?;
            }
            JobResponse::Finished(result) => {
                job_finished = true;
                let read_result = match reader.join() {
                    Ok(result) => result,
                    Err(_) => {
                        write_response(
                            &mut writer,
                            Response::Error {
                                error: "live stream reader thread panicked".to_string(),
                            },
                        )?;
                        break;
                    }
                };
                write_final_response(&mut writer, started, read_result, result)?;
                break;
            }
        }
    }

    if !job_finished {
        write_error(&mut writer.into_inner()?, "model scheduler thread exited unexpectedly")?;
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
            JobResponse::SpeechStarted { begin } => {
                write_response(writer, Response::SpeechStarted { begin })?;
            }
            JobResponse::SpeechEnded { end } => {
                write_response(writer, Response::SpeechEnded { end })?;
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

fn read_live_chunks(
    mut stream: UnixStream,
    ring: SharedAudioRing,
    chunk_tx: mpsc::Sender<Result<LiveChunk>>,
) -> Result<ReadOutcome> {
    let mut pcm = Vec::<u8>::new();
    let mut stats = StreamReadStats::default();

    loop {
        let mut signal = [0u8; 1];
        let n = stream
            .read(&mut signal)
            .context("failed to read shared-memory signal")?;
        if n == 0 {
            flush_pcm_chunks(&chunk_tx, &mut pcm, &mut stats, true)?;
            break;
        }

        match signal[0] {
            SIGNAL_DATA_READY => {
                ring.drain_available(&mut pcm)?;
                flush_pcm_chunks(&chunk_tx, &mut pcm, &mut stats, false)?;
            }
            SIGNAL_END_OF_STREAM => {
                ring.drain_available(&mut pcm)?;
                flush_pcm_chunks(&chunk_tx, &mut pcm, &mut stats, true)?;
                break;
            }
            SIGNAL_CANCEL => return Ok(ReadOutcome::Cancelled(stats)),
            other => {
                let message = format!("unsupported shared-memory signal {other}");
                let _ = chunk_tx.blocking_send(Err(anyhow::anyhow!(message.clone())));
                bail!("{message}");
            }
        }
    }

    Ok(ReadOutcome::Completed(stats))
}

fn flush_pcm_chunks(
    chunk_tx: &mpsc::Sender<Result<LiveChunk>>,
    pcm: &mut Vec<u8>,
    stats: &mut StreamReadStats,
    final_flush: bool,
) -> Result<()> {
    // We send chunks matching the VAD window (e.g. 512 samples = 1024 bytes)
    let chunk_bytes = 1024;
    while pcm.len() >= chunk_bytes {
        let chunk = pcm.drain(..chunk_bytes).collect::<Vec<_>>();
        let samples = pcm_s16le_to_f32(&chunk)?;
        stats.total_samples += samples.len();
        chunk_tx
            .blocking_send(Ok(LiveChunk { samples }))
            .map_err(|_| anyhow::anyhow!("live stream scheduler stopped"))?;
        stats.total_windows += 1;
    }

    if final_flush && !pcm.is_empty() {
        let chunk = std::mem::take(pcm);
        let samples = pcm_s16le_to_f32(&chunk)?;
        stats.total_samples += samples.len();
        chunk_tx
            .blocking_send(Ok(LiveChunk { samples }))
            .map_err(|_| anyhow::anyhow!("live stream scheduler stopped"))?;
        stats.total_windows += 1;
    }
    Ok(())
}

struct LiveChunk {
    pub samples: Vec<f32>,
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
    SpeechStarted { begin: f32 },
    SpeechEnded { end: f32 },
    Finished(Result<LiveTranscriptionStats>),
}

enum ModelJob {
    Live {
        header: RequestHeader,
        chunk_rx: mpsc::Receiver<Result<LiveChunk>>,
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
            let (pool, vad) = ModelPool::load(lib, files)
                .with_context(|| format!("failed to load model pool {model_id}"))?;
            let (job_tx, job_rx) = mpsc::channel::<ModelJob>(config.concurrency.model_queue_depth);
            spawn_model_scheduler(model_id.clone(), pool, vad, job_rx)?;
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
        chunk_rx: mpsc::Receiver<Result<LiveChunk>>,
        response_tx: std_mpsc::SyncSender<JobResponse>,
    ) -> Result<(), SubmitError> {
        let job_tx = self
            .schedulers
            .get(&header.model)
            .ok_or(SubmitError::UnknownModel)?;
        job_tx
            .try_send(ModelJob::Live {
                header,
                chunk_rx,
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
    vad_model: Option<VadModel>,
    mut job_rx: mpsc::Receiver<ModelJob>,
) -> Result<()> {
    std::thread::Builder::new()
        .name(format!("rkwhisper-scheduler-{model_id}"))
        .spawn(move || {
            while let Some(job) = job_rx.blocking_recv() {
                match job {
                    ModelJob::Live {
                        header,
                        mut chunk_rx,
                        response_tx,
                    } => {
                        let mut audio_buffer = Vec::new();
                        let mut absolute_offset_samples = 0usize;
                        let mut streaming_vad = vad_model.as_ref().map(|_m| {
                            rkwhisper::vad::StreamingVad::new(rkwhisper::vad::VadConfig {
                                threshold: header.vad_threshold.unwrap_or(0.5),
                                min_speech_ms: header.vad_min_speech_ms.unwrap_or(250),
                                min_silence_ms: header.vad_min_silence_ms.unwrap_or(100),
                                speech_pad_ms: header.vad_speech_pad_ms.unwrap_or(200),
                                window_samples: header.vad_window_samples.unwrap_or(512),
                            })
                        });

                        let mut probs = Vec::new();
                        let mut total_stats = LiveTranscriptionStats::default();
                        let mut in_flight = 0usize;
                        let mut next_window_index = 0usize;
                        let mut producer_closed = false;
                        let mut pending_results = BTreeMap::<usize, WindowTranscription>::new();
                        let mut next_result_index = 0usize;
                        let mut ready_workers = VecDeque::new();
                        let mut speech_active = false;

                        let (pool_ready_rx, pool_result_rx, worker_txs, tokenizer) = match &mut pool {
                            ModelPool::Tiny(p) => {
                                let txs = p.pool.worker_txs();
                                (&mut p.pool.ready_rx, &mut p.pool.result_rx, txs, p.tokenizer.clone())
                            }
                            ModelPool::Base(p) => {
                                let txs = p.pool.worker_txs();
                                (&mut p.pool.ready_rx, &mut p.pool.result_rx, txs, p.tokenizer.clone())
                            }
                            ModelPool::Small(p) => {
                                let txs = p.pool.worker_txs();
                                (&mut p.pool.ready_rx, &mut p.pool.result_rx, txs, p.tokenizer.clone())
                            }
                            ModelPool::Medium(p) => {
                                let txs = p.pool.worker_txs();
                                (&mut p.pool.ready_rx, &mut p.pool.result_rx, txs, p.tokenizer.clone())
                            }
                            ModelPool::LargeV3Turbo(p) => {
                                let txs = p.pool.worker_txs();
                                (&mut p.pool.ready_rx, &mut p.pool.result_rx, txs, p.tokenizer.clone())
                            }
                        };

                        let options = Arc::new(TranscribeOptions::new(
                            header.lang.clone(),
                            header.task.clone(),
                            header.notimestamps,
                            header.max_new_tokens,
                            header.beam_size,
                            SuppressTokens::parse(&header.suppress_tokens).unwrap_or(SuppressTokens::Default),
                        ));

                        // Local runtime for the scheduler loop
                        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();

                        rt.block_on(async {
                            loop {
                                if producer_closed && audio_buffer.is_empty() && in_flight == 0 {
                                    break;
                                }

                                tokio::select! {
                                    // 1. Accept new audio chunks only if we have fewer than 3 windows in flight
                                    chunk = chunk_rx.recv(), if !producer_closed && in_flight < 3 => {
                                        match chunk {
                                            Some(Ok(c)) => {
                                                let start_idx = audio_buffer.len();
                                                audio_buffer.extend_from_slice(&c.samples);

                                                if let (Some(vad), Some(v_model)) = (&mut streaming_vad, &vad_model) {
                                                    let prob = match vad.process_window(v_model, &c.samples) {
                                                        Ok(p) => p,
                                                        Err(e) => {
                                                            let _ = response_tx.send(JobResponse::Finished(Err(e)));
                                                            return;
                                                        }
                                                    };
                                                    probs.push((start_idx, prob));
                                                }
                                            }
                                            Some(Err(e)) => {
                                                let _ = response_tx.send(JobResponse::Finished(Err(e)));
                                                return;
                                            }
                                            None => {
                                                producer_closed = true;
                                            }
                                        }
                                    }

                                    // 2. Accept ready signals from NPU workers
                                    ready = pool_ready_rx.recv() => {
                                        if let Some(r) = ready {
                                            ready_workers.push_back(r.worker_id);
                                        }
                                    }

                                    // 3. Accept transcription results from NPU workers
                                    result = pool_result_rx.recv(), if in_flight > 0 => {
                                        match result {
                                            Some(Ok(res)) => {
                                                in_flight -= 1;
                                                total_stats.windows_completed += 1;
                                                pending_results.insert(res.window_index, res);

                                                while let Some(res) = pending_results.remove(&next_result_index) {
                                                    for segment in res.segments {
                                                        let _ = response_tx.send(JobResponse::Segment {
                                                            text: segment.text.clone(),
                                                            begin: segment.start_sec,
                                                            end: segment.end_sec,
                                                        });
                                                    }
                                                    next_result_index += 1;
                                                }
                                            }
                                            Some(Err(e)) => {
                                                let _ = response_tx.send(JobResponse::Finished(Err(e)));
                                                return;
                                            }
                                            None => {}
                                        }
                                    }
                                }

                                // 4. Check if we can dispatch a new window
                                if !audio_buffer.is_empty() {
                                    if let (Some(vad), Some(_v_model)) = (&mut streaming_vad, &vad_model) {
                                        let segments = rkwhisper::vad::segments_from_probs(
                                            audio_buffer.len(),
                                            &probs,
                                            vad.config(),
                                        );

                                        if !segments.is_empty() && !speech_active {
                                            speech_active = true;
                                            let begin = rkwhisper::vad::samples_to_sec(absolute_offset_samples + segments[0].start_sample);
                                            let _ = response_tx.send(JobResponse::SpeechStarted { begin });
                                        } else if segments.is_empty() && speech_active {
                                            speech_active = false;
                                            let end = rkwhisper::vad::samples_to_sec(absolute_offset_samples);
                                            let _ = response_tx.send(JobResponse::SpeechEnded { end });
                                        }

                                        if segments.is_empty() && producer_closed {
                                            // Stream closed and VAD found no more speech in buffer.
                                            // Drain the rest to allow exit.
                                            absolute_offset_samples += audio_buffer.len();
                                            audio_buffer.clear();
                                            probs.clear();
                                        }
                                    }
                                }

                                while !ready_workers.is_empty() {
                                    let mut segment_to_dispatch = None;

                                    if let (Some(vad), Some(_v_model)) = (&mut streaming_vad, &vad_model) {
                                        let segments = rkwhisper::vad::segments_from_probs(
                                            audio_buffer.len(),
                                            &probs,
                                            vad.config(),
                                        );

                                        if let Some(seg) = segments.first() {
                                            let silence_threshold = vad.config().min_silence_ms as usize * 16000 / 1000;
                                            let is_silence_gap = audio_buffer.len() - seg.end_sample >= silence_threshold;
                                            let is_full_window = seg.end_sample - seg.start_sample >= 480000;

                                            if is_silence_gap || is_full_window || (producer_closed && !audio_buffer.is_empty()) {
                                                // Dispatch first 30s or the whole segment if it's a gap
                                                let dispatch_end = if is_full_window { seg.start_sample + 480000 } else { seg.end_sample };
                                                segment_to_dispatch = Some((seg.start_sample, dispatch_end));
                                            }
                                        }
                                    } else {
                                        // No VAD, fixed 30s windows
                                        if audio_buffer.len() >= 480000 || (producer_closed && !audio_buffer.is_empty()) {
                                            let dispatch_end = 480000.min(audio_buffer.len());
                                            segment_to_dispatch = Some((0, dispatch_end));
                                        }
                                    }

                                    if let Some((start, end)) = segment_to_dispatch {
                                        let worker_id = ready_workers.pop_front().unwrap();
                                        let samples = audio_buffer.drain(..end).collect::<Vec<_>>();
                                        let segment_samples = samples[start..].to_vec();

                                        let window_start_sec = rkwhisper::vad::samples_to_sec(absolute_offset_samples + start);
                                        absolute_offset_samples += end;
                                        probs = probs.into_iter().filter(|(idx, _)| *idx >= end).map(|(idx, p)| (idx - end, p)).collect();

                                        let job = WhisperJob {
                                            window_index: next_window_index,
                                            absolute_start_sec: window_start_sec,
                                            start_sample: 0,
                                            end_sample: segment_samples.len(),
                                            samples: Arc::from(segment_samples),
                                            tokenizer: tokenizer.clone(),
                                            options: options.clone(),
                                        };

                                        let _ = worker_txs[worker_id].send(job).await;
                                        in_flight += 1;
                                        total_stats.windows_dispatched += 1;
                                        next_window_index += 1;
                                    } else {
                                        break;
                                    }
                                }

                            }
                            let _ = response_tx.send(JobResponse::Finished(Ok(total_stats)));
                        });
                    }
                    ModelJob::Batch {
                        header,
                        audio,
                        response_tx,
                    } => {
                        let segment_tx = response_tx.clone();
                        let vad_segments = if let Some(v_model) = &vad_model {
                            v_model.segments(&audio).unwrap_or_default()
                        } else {
                            Vec::new()
                        };

                        for seg in &vad_segments {
                            let _ = response_tx.send(JobResponse::SpeechStarted {
                                begin: seg.start_sec,
                            });
                            let _ = response_tx.send(JobResponse::SpeechEnded { end: seg.end_sec });
                        }

                        let result = pool.transcribe_batch_with_vad(
                            &header,
                            &audio,
                            &vad_segments,
                            |segment| {
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
                            },
                        );
                        let result = result.map(|(_transcription, stats)| stats);
                        let _ = response_tx.send(JobResponse::Finished(result));
                    }
                }
            }
        })
        .context("failed to spawn model scheduler")?;
    Ok(())
}

fn write_error(stream: &UnixStream, error: &str) -> Result<()> {
    rkwhisper::protocol::send_response_with_fd(
        stream,
        Response::Error {
            error: error.to_string(),
        },
        None,
    )?;
    Ok(())
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
    fn load(lib: &Path, files: ModelFiles) -> Result<(Self, Option<VadModel>)> {
        let vad = if let Some(path) = &files.vad {
            let config = VadConfig::default();
            Some(VadModel::new(
                RKNN::new_with_library(lib, &mut std::fs::read(path)?, 0)?,
                config,
            ))
        } else {
            None
        };

        let pool = match files.kind {
            ModelKind::Tiny => Self::Tiny(TypedModelPool::<WhisperTiny>::load(lib, files)?),
            ModelKind::Base => Self::Base(TypedModelPool::<WhisperBase>::load(lib, files)?),
            ModelKind::Small => Self::Small(TypedModelPool::<WhisperSmall>::load(lib, files)?),
            ModelKind::Medium => Self::Medium(TypedModelPool::<WhisperMedium>::load(lib, files)?),
            ModelKind::LargeV3Turbo => {
                Self::LargeV3Turbo(TypedModelPool::<WhisperLargeV3Turbo>::load(lib, files)?)
            }
        };

        Ok((pool, vad))
    }

    fn transcribe_batch_with_vad<F>(
        &mut self,
        header: &RequestHeader,
        audio: &[f32],
        vad_segments: &[rkwhisper::vad::VadSegment],
        on_segment: F,
    ) -> Result<(Transcription, LiveTranscriptionStats)>
    where
        F: FnMut(&rkwhisper::whisper::TranscriptSegment) -> Result<()>,
    {
        match self {
            Self::Tiny(pool) => pool.transcribe_batch(header, audio, vad_segments, on_segment),
            Self::Base(pool) => pool.transcribe_batch(header, audio, vad_segments, on_segment),
            Self::Small(pool) => pool.transcribe_batch(header, audio, vad_segments, on_segment),
            Self::Medium(pool) => pool.transcribe_batch(header, audio, vad_segments, on_segment),
            Self::LargeV3Turbo(pool) => {
                pool.transcribe_batch(header, audio, vad_segments, on_segment)
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

    fn transcribe_batch<F>(
        &mut self,
        header: &RequestHeader,
        audio: &[f32],
        vad_segments: &[rkwhisper::vad::VadSegment],
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

        let transcription = self.pool.transcribe_audio_with_segment_callback(
            audio,
            self.tokenizer.clone(),
            vad_segments,
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
        let (_chunk_tx, chunk_rx) = mpsc::channel(1);
        let (response_tx, _response_rx) = std_mpsc::sync_channel(1);

        let error = schedulers
            .submit(test_header("missing-model"), chunk_rx, response_tx)
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

        let (_chunk_tx_1, chunk_rx_1) = mpsc::channel(1);
        let (response_tx_1, _response_rx_1) = std_mpsc::sync_channel(1);
        schedulers
            .submit(test_header("whisper-small-30s"), chunk_rx_1, response_tx_1)
            .unwrap();

        let (_chunk_tx_2, chunk_rx_2) = mpsc::channel(1);
        let (response_tx_2, _response_rx_2) = std_mpsc::sync_channel(1);
        let error = schedulers
            .submit(test_header("whisper-small-30s"), chunk_rx_2, response_tx_2)
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
