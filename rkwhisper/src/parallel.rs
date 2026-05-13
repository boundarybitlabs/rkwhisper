use crate::decoder::WhisperDecoder;
use crate::encoder::{EncKvModel, WhisperEncoder};
use crate::spec::WhisperSpec;
use crate::vad::{VadConfig, VadModel};
use crate::whisper::{
    AudioWindow, TranscribeOptions, TranscriptSegment, Transcription, WindowTranscription,
    transcribe_window_samples, transcription_windows,
};
use crate::{MelSpectrogram, load_audio_file};
use anyhow::{Context, Result, anyhow};
use rknpu2::api::runtime::RuntimeAPI;
use rknpu2::{RKNN, utils::find_rknn_library};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

pub const NPU_WORKERS: usize = 3;

pub const CORE_MASKS: [u32; NPU_WORKERS] = [
    RKNN::<RuntimeAPI>::NPU_CORE_0,
    RKNN::<RuntimeAPI>::NPU_CORE_1,
    RKNN::<RuntimeAPI>::NPU_CORE_2,
];

#[derive(Clone, Debug)]
pub struct ParallelModelPaths {
    pub mel: PathBuf,
    pub encoder: PathBuf,
    pub enc_kv: PathBuf,
    pub decoder: PathBuf,
    pub vad: Option<PathBuf>,
}

impl ParallelModelPaths {
    pub fn new(
        mel: impl Into<PathBuf>,
        encoder: impl Into<PathBuf>,
        enc_kv: impl Into<PathBuf>,
        decoder: impl Into<PathBuf>,
    ) -> Self {
        Self {
            mel: mel.into(),
            encoder: encoder.into(),
            enc_kv: enc_kv.into(),
            decoder: decoder.into(),
            vad: None,
        }
    }

    pub fn with_vad(mut self, vad: impl Into<PathBuf>) -> Self {
        self.vad = Some(vad.into());
        self
    }
}

#[derive(Clone)]
struct ModelBytes {
    mel: Vec<u8>,
    encoder: Vec<u8>,
    enc_kv: Vec<u8>,
    decoder: Vec<u8>,
    vad: Option<Vec<u8>>,
}

impl ModelBytes {
    fn read(paths: &ParallelModelPaths) -> Result<Self> {
        Ok(Self {
            mel: std::fs::read(&paths.mel)
                .with_context(|| format!("failed to read {}", paths.mel.display()))?,
            encoder: std::fs::read(&paths.encoder)
                .with_context(|| format!("failed to read {}", paths.encoder.display()))?,
            enc_kv: std::fs::read(&paths.enc_kv)
                .with_context(|| format!("failed to read {}", paths.enc_kv.display()))?,
            decoder: std::fs::read(&paths.decoder)
                .with_context(|| format!("failed to read {}", paths.decoder.display()))?,
            vad: paths
                .vad
                .as_ref()
                .map(|path| {
                    std::fs::read(path)
                        .with_context(|| format!("failed to read {}", path.display()))
                })
                .transpose()?,
        })
    }
}

#[derive(Clone)]
pub struct WhisperJob {
    pub job_id: usize,
    pub window_index: usize,
    pub absolute_start_sec: f32,
    pub start_sample: usize,
    pub end_sample: usize,
    pub samples: Arc<[f32]>,
    pub tokenizer: Arc<Tokenizer>,
    pub options: Arc<TranscribeOptions>,
}

pub struct WorkerTranscription {
    pub job_id: usize,
    pub transcription: WindowTranscription,
}

pub struct VadJob {
    pub job_id: usize,
    pub window_index: usize,
    pub start_sample: usize,
    pub samples: Vec<f32>,
    pub state: Vec<f32>,
    pub window_samples: usize,
}

pub struct VadResult {
    pub job_id: usize,
    pub window_index: usize,
    pub start_sample: usize,
    pub probability: f32,
    pub state: Vec<f32>,
}

pub enum NpuJob {
    Vad(VadJob),
    Whisper(WhisperJob),
}

pub enum NpuResult {
    Vad(VadResult),
    Whisper(WorkerTranscription),
}

pub struct LiveWindow {
    pub index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub samples: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LiveTranscriptionStats {
    pub windows_dispatched: usize,
    pub windows_completed: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Ready {
    pub worker_id: usize,
}

struct Worker {
    job_tx: mpsc::Sender<NpuJob>,
    join: Option<JoinHandle<Result<()>>>,
}

struct PipelineCtx<S: WhisperSpec> {
    mel_spec: MelSpectrogram,
    encoder: WhisperEncoder<S>,
    enc_kv: EncKvModel<S>,
    decoder_rknn: RKNN<RuntimeAPI>,
    vad: Option<VadModel>,
}

impl<S: WhisperSpec> PipelineCtx<S> {
    fn load(lib: &Path, model_bytes: &ModelBytes, core_mask: u32) -> Result<Self> {
        let mel_rknn = pinned_rknn(lib, &model_bytes.mel, core_mask, "mel")?;
        let encoder_rknn = pinned_rknn(lib, &model_bytes.encoder, core_mask, "encoder")?;
        let enc_kv_rknn = pinned_rknn(lib, &model_bytes.enc_kv, core_mask, "enc-kv")?;
        let decoder_rknn = pinned_rknn(lib, &model_bytes.decoder, core_mask, "decoder")?;
        let vad = model_bytes
            .vad
            .as_ref()
            .map(|model| {
                pinned_rknn(lib, model, core_mask, "vad")
                    .map(|rknn| VadModel::new(rknn, VadConfig::default()))
            })
            .transpose()?;

        Ok(Self {
            mel_spec: MelSpectrogram::new(mel_rknn),
            encoder: WhisperEncoder::<S>::new(encoder_rknn),
            enc_kv: EncKvModel::<S>::new(enc_kv_rknn),
            decoder_rknn,
            vad,
        })
    }

    fn transcribe_window(&self, job: WhisperJob) -> Result<WindowTranscription> {
        let mut decoder = WhisperDecoder::<S>::new(&self.decoder_rknn);
        transcribe_window_samples(
            &job.samples,
            &job.tokenizer,
            &self.mel_spec,
            &self.encoder,
            &self.enc_kv,
            &mut decoder,
            job.window_index,
            job.absolute_start_sec,
            job.start_sample,
            job.end_sample,
            &job.options,
        )
    }

    fn run_vad(&self, job: VadJob) -> Result<VadResult> {
        let vad = self
            .vad
            .as_ref()
            .ok_or_else(|| anyhow!("VAD model is not loaded"))?;
        let mut state = job.state;
        let probability = vad.speech_probability_with_window_samples(
            &job.samples,
            &mut state,
            job.window_samples,
        )?;
        Ok(VadResult {
            job_id: job.job_id,
            window_index: job.window_index,
            start_sample: job.start_sample,
            probability,
            state,
        })
    }
}

fn pinned_rknn(lib: &Path, model: &[u8], core_mask: u32, label: &str) -> Result<RKNN<RuntimeAPI>> {
    let mut model = model.to_vec();
    let rknn = RKNN::new_with_library(lib, &mut model, 0)
        .with_context(|| format!("failed to initialize {label} RKNN context"))?;
    rknn.set_core_mask(core_mask)
        .with_context(|| format!("failed to pin {label} RKNN context to core mask {core_mask}"))?;
    Ok(rknn)
}

/// Transcribe a WAV using three RK3588 NPU workers coordinated by Tokio.
pub fn transcribe_file_parallel_with_options<S: WhisperSpec + Send + 'static>(
    audio_path: &str,
    tokenizer_path: &str,
    model_paths: &ParallelModelPaths,
    options: &TranscribeOptions,
) -> Result<Transcription> {
    let lib = find_rknn_library()
        .next()
        .ok_or_else(|| anyhow!("Could not find rknn library"))?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
    let audio = load_audio_file(audio_path)?;
    transcribe_audio_parallel_with_options::<S>(&audio, tokenizer, &lib, model_paths, options)
}

/// Transcribe in-memory audio using three RK3588 NPU workers coordinated by Tokio.
pub fn transcribe_audio_parallel_with_options<S: WhisperSpec + Send + 'static>(
    audio: &[f32],
    tokenizer: Tokenizer,
    lib: &Path,
    model_paths: &ParallelModelPaths,
    options: &TranscribeOptions,
) -> Result<Transcription> {
    let mut pool = ParallelTranscriberPool::<S>::new(lib, model_paths)?;
    pool.transcribe_audio_with_options(audio, Arc::new(tokenizer), options)
}

pub struct ParallelTranscriberPool<S: WhisperSpec + Send + 'static> {
    workers: Vec<Worker>,
    pub ready_rx: mpsc::Receiver<Ready>,
    pub result_rx: mpsc::Receiver<Result<NpuResult>>,
    runtime: tokio::runtime::Runtime,
    _spec: std::marker::PhantomData<S>,
}

impl<S: WhisperSpec + Send + 'static> ParallelTranscriberPool<S> {
    pub fn new(lib: &Path, model_paths: &ParallelModelPaths) -> Result<Self> {
        let model_bytes = Arc::new(ModelBytes::read(model_paths)?);
        let lib = lib.to_path_buf();
        let (ready_tx, ready_rx) = mpsc::channel::<Ready>(NPU_WORKERS);
        let (result_tx, result_rx) = mpsc::channel::<Result<NpuResult>>(NPU_WORKERS);

        let mut workers = Vec::with_capacity(NPU_WORKERS);
        for (worker_id, &core_mask) in CORE_MASKS.iter().enumerate() {
            workers.push(spawn_npu_worker::<S>(
                worker_id,
                core_mask,
                lib.clone(),
                model_bytes.clone(),
                ready_tx.clone(),
                result_tx.clone(),
            )?);
        }
        drop(ready_tx);
        drop(result_tx);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build Tokio runtime")?;

        Ok(Self {
            workers,
            ready_rx,
            result_rx,
            runtime,
            _spec: std::marker::PhantomData,
        })
    }

    pub fn transcribe_audio_with_options(
        &mut self,
        audio: &[f32],
        tokenizer: Arc<Tokenizer>,
        options: &TranscribeOptions,
    ) -> Result<Transcription> {
        self.transcribe_audio_with_segment_callback(audio, tokenizer, &[], options, |_| Ok(()))
    }

    pub fn transcribe_audio_with_segment_callback<F>(
        &mut self,
        audio: &[f32],
        tokenizer: Arc<Tokenizer>,
        vad_segments: &[crate::vad::VadSegment],
        options: &TranscribeOptions,
        on_segment: F,
    ) -> Result<Transcription>
    where
        F: FnMut(&TranscriptSegment) -> Result<()>,
    {
        let windows = transcription_windows(audio.len(), vad_segments);
        if windows.is_empty() {
            return Ok(Transcription {
                text: String::new(),
                segments: Vec::new(),
                vad_segments: vad_segments.to_vec(),
            });
        }

        let total_windows = windows.len();
        let audio = Arc::<[f32]>::from(audio);
        let options = Arc::new(options.clone());
        let worker_txs = self
            .workers
            .iter()
            .map(|worker| worker.job_tx.clone())
            .collect::<Vec<_>>();
        let ready_rx = &mut self.ready_rx;
        let result_rx = &mut self.result_rx;

        let vad_segments_vec = vad_segments.to_vec();
        self.runtime.block_on(async {
            let dispatched =
                dispatch_windows(windows, worker_txs, ready_rx, audio, tokenizer, options);
            let collected = collect_ordered_with_callback(
                result_rx,
                total_windows,
                vad_segments_vec,
                on_segment,
            );
            let (_, transcription) = tokio::try_join!(dispatched, collected)?;
            Ok(transcription)
        })
    }

    pub fn worker_txs(&self) -> Vec<mpsc::Sender<NpuJob>> {
        self.workers.iter().map(|w| w.job_tx.clone()).collect()
    }

    pub fn vad_segments_with_config(
        &mut self,
        audio: &[f32],
        config: &VadConfig,
    ) -> Result<Vec<crate::vad::VadSegment>> {
        let audio = Arc::<[f32]>::from(audio);
        let worker_txs = self.worker_txs();
        let ready_rx = &mut self.ready_rx;
        let result_rx = &mut self.result_rx;
        let config = config.clone();

        self.runtime.block_on(async {
            let mut probs = Vec::new();
            let mut state = vec![0.0f32; 2 * 128];
            let mut window_index = 0usize;
            for start in (0..audio.len()).step_by(config.window_samples) {
                let end = (start + config.window_samples).min(audio.len());
                let ready = ready_rx
                    .recv()
                    .await
                    .ok_or_else(|| anyhow!("all NPU workers stopped"))?;
                let worker_tx = worker_txs
                    .get(ready.worker_id)
                    .ok_or_else(|| anyhow!("invalid worker id {}", ready.worker_id))?;
                worker_tx
                    .send(NpuJob::Vad(VadJob {
                        job_id: 0,
                        window_index,
                        start_sample: start,
                        samples: audio[start..end].to_vec(),
                        state,
                        window_samples: config.window_samples,
                    }))
                    .await
                    .map_err(|_| {
                        anyhow!(
                            "worker {} stopped before accepting a VAD job",
                            ready.worker_id
                        )
                    })?;

                loop {
                    let result = result_rx
                        .recv()
                        .await
                        .ok_or_else(|| anyhow!("result channel closed before VAD completed"))??;
                    match result {
                        NpuResult::Vad(result) => {
                            state = result.state;
                            probs.push((result.start_sample, result.probability));
                            break;
                        }
                        NpuResult::Whisper(_) => {}
                    }
                }
                window_index += 1;
            }
            Ok(crate::vad::segments_from_probs(
                audio.len(),
                &probs,
                &config,
            ))
        })
    }
}

impl<S: WhisperSpec + Send + 'static> Drop for ParallelTranscriberPool<S> {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            let (replacement_tx, _replacement_rx) = mpsc::channel(1);
            let old_tx = std::mem::replace(&mut worker.job_tx, replacement_tx);
            drop(old_tx);
        }
        for worker in &mut self.workers {
            if let Some(join) = worker.join.take() {
                let _ = join.join();
            }
        }
    }
}

fn spawn_npu_worker<S: WhisperSpec + Send + 'static>(
    worker_id: usize,
    core_mask: u32,
    lib: PathBuf,
    model_bytes: Arc<ModelBytes>,
    ready_tx: mpsc::Sender<Ready>,
    result_tx: mpsc::Sender<Result<NpuResult>>,
) -> Result<Worker> {
    let (job_tx, mut job_rx) = mpsc::channel::<NpuJob>(1);
    let join = std::thread::Builder::new()
        .name(format!("rkwhisper-npu-{worker_id}"))
        .spawn(move || {
            let ctx = match PipelineCtx::<S>::load(&lib, &model_bytes, core_mask) {
                Ok(ctx) => ctx,
                Err(error) => {
                    let _ = result_tx.blocking_send(Err(
                        error.context(format!("worker {worker_id} failed to initialize"))
                    ));
                    return Ok(());
                }
            };

            ready_tx
                .blocking_send(Ready { worker_id })
                .map_err(|_| anyhow!("worker {worker_id} ready channel closed"))?;

            while let Some(job) = job_rx.blocking_recv() {
                let result = match job {
                    NpuJob::Whisper(job) => {
                        let job_id = job.job_id;
                        let window_index = job.window_index;
                        ctx.transcribe_window(job)
                            .map(|transcription| {
                                NpuResult::Whisper(WorkerTranscription {
                                    job_id,
                                    transcription,
                                })
                            })
                            .with_context(|| {
                                format!(
                                    "worker {worker_id} failed on Whisper window {window_index}"
                                )
                            })
                    }
                    NpuJob::Vad(job) => {
                        let window_index = job.window_index;
                        ctx.run_vad(job).map(NpuResult::Vad).with_context(|| {
                            format!("worker {worker_id} failed on VAD window {window_index}")
                        })
                    }
                };
                result_tx
                    .blocking_send(result)
                    .map_err(|_| anyhow!("worker {worker_id} result channel closed"))?;
                ready_tx
                    .blocking_send(Ready { worker_id })
                    .map_err(|_| anyhow!("worker {worker_id} ready channel closed"))?;
            }

            Ok(())
        })
        .context("failed to spawn NPU worker")?;

    Ok(Worker {
        job_tx,
        join: Some(join),
    })
}

async fn dispatch_windows(
    windows: Vec<AudioWindow>,
    worker_txs: Vec<mpsc::Sender<NpuJob>>,
    ready_rx: &mut mpsc::Receiver<Ready>,
    audio: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    options: Arc<TranscribeOptions>,
) -> Result<()> {
    let mut pending = VecDeque::from(windows);

    while let Some(window) = pending.pop_front() {
        let ready = ready_rx
            .recv()
            .await
            .ok_or_else(|| anyhow!("all NPU workers stopped"))?;
        let worker_tx = worker_txs
            .get(ready.worker_id)
            .ok_or_else(|| anyhow!("invalid worker id {}", ready.worker_id))?;
        worker_tx
            .send(NpuJob::Whisper(WhisperJob {
                window_index: window.index,
                absolute_start_sec: crate::vad::samples_to_sec(window.start_sample),
                job_id: 0,
                start_sample: window.start_sample,
                end_sample: window.end_sample,
                samples: Arc::<[f32]>::from(&audio[window.start_sample..window.end_sample]),
                tokenizer: tokenizer.clone(),
                options: options.clone(),
            }))
            .await
            .map_err(|_| anyhow!("worker {} stopped before accepting a job", ready.worker_id))?;
    }

    Ok(())
}

async fn collect_ordered_with_callback<F>(
    result_rx: &mut mpsc::Receiver<Result<NpuResult>>,
    total_windows: usize,
    vad_segments: Vec<crate::vad::VadSegment>,
    mut on_segment: F,
) -> Result<Transcription>
where
    F: FnMut(&TranscriptSegment) -> Result<()>,
{
    let mut pending = BTreeMap::<usize, WindowTranscription>::new();
    let mut next = 0usize;
    let mut full_text = String::new();
    let mut segments = Vec::new();

    while next < total_windows {
        let result = result_rx
            .recv()
            .await
            .ok_or_else(|| anyhow!("result channel closed before all windows completed"))??;
        let NpuResult::Whisper(result) = result else {
            continue;
        };
        let result = result.transcription;
        pending.insert(result.window_index, result);

        while let Some(result) = pending.remove(&next) {
            full_text.push_str(&result.text);
            full_text.push(' ');
            for segment in &result.segments {
                on_segment(segment)?;
            }
            segments.extend(result.segments);
            next += 1;
        }
    }

    Ok(Transcription {
        text: full_text.trim().to_string(),
        segments,
        vad_segments,
    })
}

#[cfg(test)]
mod tests {
    use super::collect_ordered_with_callback;
    use crate::whisper::WindowTranscription;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn collect_ordered_reorders_results() {
        let (tx, rx) = mpsc::channel(3);
        tx.send(Ok(worker_result(2, "three"))).await.unwrap();
        tx.send(Ok(worker_result(0, "one"))).await.unwrap();
        tx.send(Ok(worker_result(1, "two"))).await.unwrap();
        drop(tx);

        let mut rx = rx;
        let transcription = collect_ordered_with_callback(&mut rx, 3, Vec::new(), |_| Ok(()))
            .await
            .unwrap();
        assert_eq!(transcription.text, "one two three");
    }

    #[tokio::test]
    async fn collect_ordered_reports_early_close() {
        let (tx, rx) = mpsc::channel(1);
        drop(tx);

        let mut rx = rx;
        let error = collect_ordered_with_callback(&mut rx, 1, Vec::new(), |_| Ok(()))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("result channel closed"));
    }

    fn worker_result(window_index: usize, text: &str) -> super::NpuResult {
        super::NpuResult::Whisper(super::WorkerTranscription {
            job_id: 0,
            transcription: WindowTranscription {
                window_index,
                absolute_start_sec: 0.0,
                text: text.to_string(),
                segments: Vec::new(),
            },
        })
    }
}
