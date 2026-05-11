use anyhow::{Context, Result, bail};
use clap::Parser;
use rknpu2::{RKNN, utils::find_rknn_library};
use rkwhisper::{
    daemon::{
        ConcurrencyConfig, DEFAULT_CONFIG_PATH, DEFAULT_SOCKET_PATH, DaemonConfig, ModelFiles,
        ModelKind, RequestHeader, default_model_root, load_config, pcm_s16le_to_f32,
        resolve_enabled_model_files,
    },
    daemon_stream::{LiveChunk, ReadOutcome, read_live_chunks},
    parallel::{
        LiveTranscriptionStats, NPU_WORKERS, ParallelModelPaths, ParallelTranscriberPool, Ready,
        WhisperJob, WorkerTranscription,
    },
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
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tracing::{debug, trace};

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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

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
        write_error(
            &mut writer.into_inner()?,
            "model scheduler thread exited unexpectedly",
        )?;
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
            spawn_model_scheduler(
                model_id.clone(),
                pool,
                vad,
                config.concurrency.clone(),
                job_rx,
            )?;
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

struct ActiveLiveJob {
    job_id: usize,
    chunk_rx: mpsc::Receiver<Result<LiveChunk>>,
    response_tx: std_mpsc::SyncSender<JobResponse>,
    audio_buffer: Vec<f32>,
    absolute_offset_samples: usize,
    streaming_vad: Option<rkwhisper::vad::StreamingVad>,
    probs: Vec<(usize, f32)>,
    stats: LiveTranscriptionStats,
    in_flight: usize,
    next_window_index: usize,
    producer_closed: bool,
    pending_results: BTreeMap<usize, WindowTranscription>,
    next_result_index: usize,
    tokenizer: Arc<Tokenizer>,
    options: Arc<TranscribeOptions>,
}

struct DispatchPlan {
    start_sample: usize,
    end_sample: usize,
    speech_events: Vec<(usize, usize)>,
}

impl ActiveLiveJob {
    fn new(
        job_id: usize,
        header: RequestHeader,
        chunk_rx: mpsc::Receiver<Result<LiveChunk>>,
        response_tx: std_mpsc::SyncSender<JobResponse>,
        tokenizer: Arc<Tokenizer>,
        vad_enabled: bool,
    ) -> Self {
        let streaming_vad = vad_enabled.then(|| {
            let defaults = rkwhisper::vad::VadConfig::default();
            rkwhisper::vad::StreamingVad::new(rkwhisper::vad::VadConfig {
                threshold: header.vad_threshold.unwrap_or(defaults.threshold),
                min_speech_ms: header.vad_min_speech_ms.unwrap_or(defaults.min_speech_ms),
                min_silence_ms: header.vad_min_silence_ms.unwrap_or(defaults.min_silence_ms),
                speech_pad_ms: header.vad_speech_pad_ms.unwrap_or(defaults.speech_pad_ms),
                window_samples: header.vad_window_samples.unwrap_or(defaults.window_samples),
            })
        });
        let options = Arc::new(TranscribeOptions::new(
            header.lang,
            header.task,
            header.notimestamps,
            header.max_new_tokens,
            header.beam_size,
            SuppressTokens::parse(&header.suppress_tokens).unwrap_or(SuppressTokens::Default),
        ));

        Self {
            job_id,
            chunk_rx,
            response_tx,
            audio_buffer: Vec::new(),
            absolute_offset_samples: 0,
            streaming_vad,
            probs: Vec::new(),
            stats: LiveTranscriptionStats::default(),
            in_flight: 0,
            next_window_index: 0,
            producer_closed: false,
            pending_results: BTreeMap::new(),
            next_result_index: 0,
            tokenizer,
            options,
        }
    }

    fn is_complete(&self) -> bool {
        self.producer_closed && self.audio_buffer.is_empty() && self.in_flight == 0
    }

    fn finish_if_complete(&mut self) -> bool {
        if !self.is_complete() {
            return false;
        }
        let _ = self.response_tx.send(JobResponse::Finished(Ok(self.stats)));
        true
    }

    fn accept_chunk(&mut self, chunk: Option<Result<LiveChunk>>, vad_model: Option<&VadModel>) {
        match chunk {
            Some(Ok(chunk)) => {
                let start_idx = self.audio_buffer.len();
                self.audio_buffer.extend_from_slice(&chunk.samples);

                if let (Some(vad), Some(v_model)) = (&mut self.streaming_vad, vad_model) {
                    let win = vad.config().window_samples;
                    let mut offset = 0;
                    while offset < chunk.samples.len() {
                        let end = (offset + win).min(chunk.samples.len());
                        let prob = match vad.process_window(v_model, &chunk.samples[offset..end]) {
                            Ok(prob) => prob,
                            Err(error) => {
                                let _ = self.response_tx.send(JobResponse::Finished(Err(error)));
                                self.producer_closed = true;
                                return;
                            }
                        };
                        self.probs.push((start_idx + offset, prob));
                        offset += win;
                    }
                }
            }
            Some(Err(error)) => {
                let _ = self.response_tx.send(JobResponse::Finished(Err(error)));
                self.producer_closed = true;
            }
            None => {
                self.producer_closed = true;
            }
        }
    }

    fn refresh_speech_state(&mut self) {
        if self.audio_buffer.is_empty() {
            return;
        }

        if let Some(vad) = &self.streaming_vad {
            let segments = rkwhisper::vad::segments_from_probs(
                self.audio_buffer.len(),
                &self.probs,
                vad.config(),
            );

            if segments.is_empty() && self.producer_closed {
                self.absolute_offset_samples += self.audio_buffer.len();
                self.audio_buffer.clear();
                self.probs.clear();
            }
        }
    }

    fn next_dispatch_plan(&self) -> Option<DispatchPlan> {
        if let Some(vad) = &self.streaming_vad {
            let segments = rkwhisper::vad::segments_from_probs(
                self.audio_buffer.len(),
                &self.probs,
                vad.config(),
            );
            let first_seg = segments.first()?;
            let mut last_fitting_idx = 0;
            for (i, seg) in segments.iter().enumerate() {
                if seg.end_sample - first_seg.start_sample <= 480000 {
                    last_fitting_idx = i;
                } else {
                    break;
                }
            }

            let last_fitting_seg = &segments[last_fitting_idx];
            let silence_timeout_samples = 32000;
            let is_timeout =
                self.audio_buffer.len() - last_fitting_seg.end_sample >= silence_timeout_samples;
            let has_overflowing_segment = last_fitting_idx + 1 < segments.len();
            let is_full_window = last_fitting_seg.end_sample - first_seg.start_sample >= 480000;

            if has_overflowing_segment
                || is_full_window
                || is_timeout
                || (self.producer_closed && !self.audio_buffer.is_empty())
            {
                let dispatch_end = if is_full_window {
                    first_seg.start_sample + 480000
                } else {
                    last_fitting_seg.end_sample
                };
                let speech_events = segments
                    .iter()
                    .take(last_fitting_idx + 1)
                    .map(|segment| (segment.start_sample, segment.end_sample.min(dispatch_end)))
                    .collect();
                return Some(DispatchPlan {
                    start_sample: first_seg.start_sample,
                    end_sample: dispatch_end,
                    speech_events,
                });
            }
        } else if self.audio_buffer.len() >= 480000
            || (self.producer_closed && !self.audio_buffer.is_empty())
        {
            return Some(DispatchPlan {
                start_sample: 0,
                end_sample: 480000.min(self.audio_buffer.len()),
                speech_events: Vec::new(),
            });
        }

        None
    }

    async fn dispatch_to_worker(
        &mut self,
        worker_id: usize,
        worker_txs: &[mpsc::Sender<WhisperJob>],
    ) {
        let Some(plan) = self.next_dispatch_plan() else {
            return;
        };
        for (start, end) in &plan.speech_events {
            let begin = rkwhisper::vad::samples_to_sec(self.absolute_offset_samples + start);
            let end = rkwhisper::vad::samples_to_sec(self.absolute_offset_samples + end);
            let _ = self.response_tx.send(JobResponse::SpeechStarted { begin });
            let _ = self.response_tx.send(JobResponse::SpeechEnded { end });
        }
        let start = plan.start_sample;
        let end = plan.end_sample;
        let samples = self.audio_buffer.drain(..end).collect::<Vec<_>>();
        let segment_samples = samples[start..].to_vec();
        let window_start_sec = rkwhisper::vad::samples_to_sec(self.absolute_offset_samples + start);
        self.absolute_offset_samples += end;
        self.probs = self
            .probs
            .drain(..)
            .filter(|(idx, _)| *idx >= end)
            .map(|(idx, prob)| (idx - end, prob))
            .collect();

        let job = WhisperJob {
            job_id: self.job_id,
            window_index: self.next_window_index,
            absolute_start_sec: window_start_sec,
            start_sample: 0,
            end_sample: segment_samples.len(),
            samples: Arc::from(segment_samples),
            tokenizer: self.tokenizer.clone(),
            options: self.options.clone(),
        };

        if worker_txs[worker_id].send(job).await.is_ok() {
            self.in_flight += 1;
            self.stats.windows_dispatched += 1;
            self.next_window_index += 1;
        }
    }

    fn handle_result(&mut self, result: WindowTranscription) {
        self.in_flight = self.in_flight.saturating_sub(1);
        self.stats.windows_completed += 1;
        self.pending_results.insert(result.window_index, result);

        while let Some(result) = self.pending_results.remove(&self.next_result_index) {
            for segment in result.segments {
                let _ = self.response_tx.send(JobResponse::Segment {
                    text: segment.text.clone(),
                    begin: segment.start_sec,
                    end: segment.end_sec,
                });
            }
            self.next_result_index += 1;
        }
    }
}

async fn run_live_jobs(
    initial_job: ActiveLiveJob,
    next_job_id: &mut usize,
    job_rx: &mut mpsc::Receiver<ModelJob>,
    vad_model: Option<&VadModel>,
    ready_rx: &mut mpsc::Receiver<Ready>,
    result_rx: &mut mpsc::Receiver<Result<WorkerTranscription>>,
    worker_txs: Vec<mpsc::Sender<WhisperJob>>,
    concurrency: &ConcurrencyConfig,
) -> VecDeque<ModelJob> {
    let max_active_jobs = concurrency.max_active_jobs_per_model;
    let max_in_flight_windows_per_job = concurrency.max_in_flight_windows_per_job.min(NPU_WORKERS);
    let mut active_jobs = Vec::<ActiveLiveJob>::from([initial_job]);
    let mut deferred_jobs = VecDeque::new();
    let mut ready_workers = VecDeque::new();
    let mut dispatch_cursor = 0usize;

    while !active_jobs.is_empty() {
        drain_live_chunks(&mut active_jobs, vad_model, max_in_flight_windows_per_job);
        remove_completed_jobs(&mut active_jobs);
        if active_jobs.is_empty() {
            break;
        }

        tokio::select! {
            job = job_rx.recv(), if deferred_jobs.is_empty() && active_jobs.len() < max_active_jobs => {
                match job {
                    Some(ModelJob::Live { header, chunk_rx, response_tx }) => {
                        let tokenizer = active_jobs[0].tokenizer.clone();
                        let active_job = ActiveLiveJob::new(
                            *next_job_id,
                            header,
                            chunk_rx,
                            response_tx,
                            tokenizer,
                            vad_model.is_some(),
                        );
                        *next_job_id += 1;
                        active_jobs.push(active_job);
                    }
                    Some(job @ ModelJob::Batch { .. }) => {
                        deferred_jobs.push_back(job);
                    }
                    None => {}
                }
            }

            ready = ready_rx.recv() => {
                if let Some(ready) = ready {
                    ready_workers.push_back(ready.worker_id);
                }
            }

            result = result_rx.recv(), if active_jobs.iter().any(|job| job.in_flight > 0) => {
                match result {
                    Some(Ok(result)) => {
                        if let Some(job) = active_jobs.iter_mut().find(|job| job.job_id == result.job_id) {
                            job.handle_result(result.transcription);
                        }
                    }
                    Some(Err(error)) => {
                        for job in &active_jobs {
                            let _ = job.response_tx.send(JobResponse::Finished(Err(anyhow::anyhow!("{error:#}"))));
                        }
                        active_jobs.clear();
                        break;
                    }
                    None => {}
                }
            }

            _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {}
        }

        drain_live_chunks(&mut active_jobs, vad_model, max_in_flight_windows_per_job);
        for job in &mut active_jobs {
            job.refresh_speech_state();
        }
        dispatch_ready_live_windows(
            &mut active_jobs,
            &mut ready_workers,
            &worker_txs,
            max_in_flight_windows_per_job,
            &mut dispatch_cursor,
        )
        .await;
        remove_completed_jobs(&mut active_jobs);
    }

    deferred_jobs
}

fn drain_live_chunks(
    active_jobs: &mut [ActiveLiveJob],
    vad_model: Option<&VadModel>,
    max_in_flight_windows_per_job: usize,
) {
    for job in active_jobs {
        if job.producer_closed || job.in_flight >= max_in_flight_windows_per_job {
            continue;
        }

        loop {
            match job.chunk_rx.try_recv() {
                Ok(chunk) => job.accept_chunk(Some(chunk), vad_model),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    job.accept_chunk(None, vad_model);
                    break;
                }
            }
        }
    }
}

async fn dispatch_ready_live_windows(
    active_jobs: &mut [ActiveLiveJob],
    ready_workers: &mut VecDeque<usize>,
    worker_txs: &[mpsc::Sender<WhisperJob>],
    max_in_flight_windows_per_job: usize,
    dispatch_cursor: &mut usize,
) {
    while !ready_workers.is_empty() && !active_jobs.is_empty() {
        let mut selected = None;
        for _ in 0..active_jobs.len() {
            let idx = *dispatch_cursor % active_jobs.len();
            *dispatch_cursor = (*dispatch_cursor + 1) % active_jobs.len();
            let job = &active_jobs[idx];
            if job.in_flight < max_in_flight_windows_per_job && job.next_dispatch_plan().is_some() {
                selected = Some(idx);
                break;
            }
        }

        let Some(job_idx) = selected else {
            break;
        };
        let worker_id = ready_workers.pop_front().unwrap();
        active_jobs[job_idx]
            .dispatch_to_worker(worker_id, worker_txs)
            .await;
    }
}

fn remove_completed_jobs(active_jobs: &mut Vec<ActiveLiveJob>) {
    let mut idx = 0;
    while idx < active_jobs.len() {
        if active_jobs[idx].finish_if_complete() {
            active_jobs.remove(idx);
        } else {
            idx += 1;
        }
    }
}

fn spawn_model_scheduler(
    model_id: String,
    mut pool: ModelPool,
    vad_model: Option<VadModel>,
    concurrency: ConcurrencyConfig,
    mut job_rx: mpsc::Receiver<ModelJob>,
) -> Result<()> {
    std::thread::Builder::new()
        .name(format!("rkwhisper-scheduler-{model_id}"))
        .spawn(move || {
            let mut deferred_jobs = VecDeque::new();
            let mut next_job_id = 1usize;

            loop {
                let job = if let Some(job) = deferred_jobs.pop_front() {
                    job
                } else {
                    match job_rx.blocking_recv() {
                        Some(job) => job,
                        None => break,
                    }
                };
                match job {
                    ModelJob::Live {
                        header,
                        chunk_rx,
                        response_tx,
                    } => {
                        let (pool_ready_rx, pool_result_rx, worker_txs, tokenizer) =
                            pool.worker_parts();
                        let active_job = ActiveLiveJob::new(
                            next_job_id,
                            header,
                            chunk_rx,
                            response_tx,
                            tokenizer,
                            vad_model.is_some(),
                        );
                        next_job_id += 1;
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap();
                        let newly_deferred = rt.block_on(run_live_jobs(
                            active_job,
                            &mut next_job_id,
                            &mut job_rx,
                            vad_model.as_ref(),
                            pool_ready_rx,
                            pool_result_rx,
                            worker_txs,
                            &concurrency,
                        ));
                        deferred_jobs.extend(newly_deferred);
                    }
                    ModelJob::Batch {
                        header,
                        audio,
                        response_tx,
                    } => {
                        let segment_tx = response_tx.clone();
                        let audio_samples = audio.len();
                        let audio_sec = audio_samples as f32 / rkwhisper::SAMPLE_RATE as f32;
                        debug!(
                            audio_samples,
                            audio_sec,
                            notimestamps = header.notimestamps,
                            max_new_tokens = header.max_new_tokens,
                            vad_threshold = ?header.vad_threshold,
                            "batch job received"
                        );
                        let vad_segments = if let Some(v_model) = &vad_model {
                            let defaults = rkwhisper::vad::VadConfig::default();
                            let vad_config = rkwhisper::vad::VadConfig {
                                threshold: header.vad_threshold.unwrap_or(defaults.threshold),
                                min_speech_ms: header
                                    .vad_min_speech_ms
                                    .unwrap_or(defaults.min_speech_ms),
                                min_silence_ms: header
                                    .vad_min_silence_ms
                                    .unwrap_or(defaults.min_silence_ms),
                                speech_pad_ms: header
                                    .vad_speech_pad_ms
                                    .unwrap_or(defaults.speech_pad_ms),
                                window_samples: header
                                    .vad_window_samples
                                    .unwrap_or(defaults.window_samples),
                            };
                            let segs = v_model
                                .segments_with_config(&audio, &vad_config)
                                .unwrap_or_default();
                            debug!(vad_segments = segs.len(), "VAD segmentation complete");
                            for (i, seg) in segs.iter().enumerate() {
                                trace!(
                                    i,
                                    start_sec = seg.start_sec,
                                    end_sec = seg.end_sec,
                                    "VAD segment"
                                );
                            }
                            segs
                        } else {
                            debug!("no VAD model; using fixed windows");
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
                                debug!(
                                    start_sec = segment.start_sec,
                                    end_sec = segment.end_sec,
                                    text = %segment.text,
                                    "batch segment emitted"
                                );
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
    fn worker_parts(
        &mut self,
    ) -> (
        &mut mpsc::Receiver<Ready>,
        &mut mpsc::Receiver<Result<WorkerTranscription>>,
        Vec<mpsc::Sender<WhisperJob>>,
        Arc<Tokenizer>,
    ) {
        match self {
            Self::Tiny(p) => {
                let txs = p.pool.worker_txs();
                (
                    &mut p.pool.ready_rx,
                    &mut p.pool.result_rx,
                    txs,
                    p.tokenizer.clone(),
                )
            }
            Self::Base(p) => {
                let txs = p.pool.worker_txs();
                (
                    &mut p.pool.ready_rx,
                    &mut p.pool.result_rx,
                    txs,
                    p.tokenizer.clone(),
                )
            }
            Self::Small(p) => {
                let txs = p.pool.worker_txs();
                (
                    &mut p.pool.ready_rx,
                    &mut p.pool.result_rx,
                    txs,
                    p.tokenizer.clone(),
                )
            }
            Self::Medium(p) => {
                let txs = p.pool.worker_txs();
                (
                    &mut p.pool.ready_rx,
                    &mut p.pool.result_rx,
                    txs,
                    p.tokenizer.clone(),
                )
            }
            Self::LargeV3Turbo(p) => {
                let txs = p.pool.worker_txs();
                (
                    &mut p.pool.ready_rx,
                    &mut p.pool.result_rx,
                    txs,
                    p.tokenizer.clone(),
                )
            }
        }
    }

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
