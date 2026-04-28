# RK3588 RKNN Whisper pipeline - Tokio multi-NPU architecture

## Overview

The RK3588 NPU exposes three independent cores. The current implementation runs three
independent Whisper pipelines in parallel, with one long-lived worker per NPU
core. Tokio coordinates input windows, worker readiness, backpressure, result
collection, and shutdown; RKNN inference itself remains synchronous and
worker-local.

The important rule is ownership: each worker owns a full RKNN context set for
one core:

- mel spectrogram
- encoder
- encoder-KV
- decoder RKNN handle, with a short-lived `WhisperDecoder` constructed per
  window

Do not share RKNN contexts with `Arc<Mutex<_>>`. The current `rknpu2::RKNN`
handle is a thin wrapper around an RKNN context and should be treated as
thread-affine. All calls to `set_inputs`, `run`, and `get_outputs` happen on the
same worker thread that created and pinned the contexts.

---

## Pipeline stages

For batch transcription, VAD runs before fanout when a VAD model is configured
because it determines the audio windows. If VAD is not configured, the
coordinator builds fixed 30-second windows. Once windows exist, each window is
independent and can be processed by any available NPU worker.

For daemon requests, clients write 16 kHz mono s16le PCM into a shared-memory
ring buffer passed with `SCM_RIGHTS`. The server copies ready bytes out of the
ring, buffers them into 30-second windows, and dispatches a final shorter window
at end-of-stream.

| Stage | Owner | Description |
|---|---|---|
| VAD windowing | caller / coordinator | Builds fixed 30-second windows or VAD-derived windows |
| Mel spectrogram | worker NPU context | Runs `mel.rknn` and post-processes log-mel features |
| Encoder | worker NPU context | Runs `encoder.rknn` once per window |
| Encoder-KV | worker NPU context | Runs `enc_kv.rknn` once per window |
| Decoder loop | worker NPU context | Runs `decoder.rknn` once per prompt/generated token |

The decoder loop dominates runtime and varies with output length. Static
round-robin dispatch would leave cores idle when one window generates many more
tokens than the others, so the Tokio dispatcher assigns work to whichever worker
announces readiness first.

---

## Core masks

For RK3588, the implemented pool starts exactly three workers:

```rust
const NPU_WORKERS: usize = 3;

const CORE_MASKS: [u32; NPU_WORKERS] = [
    RKNN::<RuntimeAPI>::NPU_CORE_0,
    RKNN::<RuntimeAPI>::NPU_CORE_1,
    RKNN::<RuntimeAPI>::NPU_CORE_2,
];
```

Each worker calls `set_core_mask(core_mask)` immediately after constructing
each RKNN context:

```rust
let mel = RKNN::new_with_library(lib, &mut mel_model, 0)?;
mel.set_core_mask(core_mask)?;

let encoder = RKNN::new_with_library(lib, &mut encoder_model, 0)?;
encoder.set_core_mask(core_mask)?;

let enc_kv = RKNN::new_with_library(lib, &mut enc_kv_model, 0)?;
enc_kv.set_core_mask(core_mask)?;

let decoder_rknn = RKNN::new_with_library(lib, &mut decoder_model, 0)?;
decoder_rknn.set_core_mask(core_mask)?;
```

All contexts for a window stay on one NPU core. Avoid using
`NPU_CORE_0_1_2` for this design; that mode asks one model instance to use all
cores and prevents concurrent per-window scheduling.

---

## Worker-owned pipeline context

Wrap the existing model wrappers in a worker-local context. The exact model spec
remains generic over `WhisperSpec`. The current implementation stores the
decoder RKNN handle in the context and constructs a short-lived
`WhisperDecoder` for each window.

```rust
struct PipelineCtx<S: WhisperSpec> {
    mel_spec: MelSpectrogram,
    encoder: WhisperEncoder<S>,
    enc_kv: EncKvModel<S>,
    decoder_rknn: RKNN<RuntimeAPI>,
}
```

`PipelineCtx::transcribe_window(...)` delegates to the shared per-window body:
log-mel, encode, enc-KV, prompt, greedy or beam decode, token decode, and
`TranscriptSegment` construction.

---

## Tokio orchestration

Tokio owns the control plane:

- one bounded job channel per worker
- one shared ready channel
- one shared result channel
- one dispatcher future
- one collector/reorder future

The workers themselves run on dedicated OS threads. Do not use Tokio's general
worker pool for RKNN inference, because the model contexts must remain attached
to their owning thread and NPU core.

```rust
struct Job {
    window_index: usize,
    start_sample: usize,
    end_sample: usize,
    samples: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    options: Arc<TranscribeOptions>,
}

struct Ready {
    worker_id: usize,
}

struct WindowTranscription {
    window_index: usize,
    text: String,
    segments: Vec<TranscriptSegment>,
}
```

Worker startup:

```rust
fn spawn_npu_worker<S: WhisperSpec + Send + 'static>(
    worker_id: usize,
    core_mask: u32,
    lib: PathBuf,
    model_bytes: Arc<ModelBytes>,
    ready_tx: tokio::sync::mpsc::Sender<Ready>,
    result_tx: tokio::sync::mpsc::Sender<anyhow::Result<WindowTranscription>>,
) -> anyhow::Result<Worker> {
    let (job_tx, mut job_rx) = tokio::sync::mpsc::channel::<Job>(1);

    let join = std::thread::Builder::new()
        .name(format!("rkwhisper-npu-{worker_id}"))
        .spawn(move || {
            let ctx = PipelineCtx::<S>::load(&lib, &model_bytes, core_mask)?;

            ready_tx.blocking_send(Ready { worker_id })?;

            while let Some(job) = job_rx.blocking_recv() {
                let result = ctx.transcribe_window(job);
                result_tx.blocking_send(result)?;
                ready_tx.blocking_send(Ready { worker_id })?;
            }

            anyhow::Ok(())
        })
        .context("failed to spawn NPU worker")?;

    Ok(Worker {
        job_tx,
        join: Some(join),
    })
}
```

The bounded job channel is intentional. A worker should have at most one queued
window, which keeps memory bounded and keeps readiness meaningful.

---

## Dynamic dispatch

The dispatcher keeps a queue of pending windows and sends the next window to the
next ready worker.

```rust
async fn dispatch_windows(
    windows: Vec<AudioWindow>,
    worker_txs: Vec<tokio::sync::mpsc::Sender<Job>>,
    ready_rx: &mut tokio::sync::mpsc::Receiver<Ready>,
    audio: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    options: Arc<TranscribeOptions>,
) -> anyhow::Result<()> {
    let mut pending = std::collections::VecDeque::from(windows);

    while let Some(window) = pending.pop_front() {
        let ready = ready_rx.recv().await
            .ok_or_else(|| anyhow::anyhow!("all NPU workers stopped"))?;

        worker_txs[ready.worker_id]
            .send(Job {
                window_index: window.index,
                start_sample: window.start_sample,
                end_sample: window.end_sample,
                samples: Arc::<[f32]>::from(
                    audio[window.start_sample..window.end_sample].to_vec(),
                ),
                tokenizer: tokenizer.clone(),
                options: options.clone(),
            })
            .await?;
    }

    Ok(())
}
```

This gives Rayon-style load balancing without Rayon. A verbose window ties up
only its own worker; the next completed worker immediately receives the next
pending window.

---

## Result reordering

Results may complete out of order. Buffer by `window_index` and emit segments
only when the next expected index is available.

```rust
async fn collect_ordered(
    result_rx: &mut tokio::sync::mpsc::Receiver<anyhow::Result<WindowTranscription>>,
    total_windows: usize,
    vad_segments: Vec<VadSegment>,
) -> anyhow::Result<Transcription> {
    let mut pending = std::collections::BTreeMap::<usize, WindowTranscription>::new();
    let mut next = 0usize;
    let mut full_text = String::new();
    let mut segments = Vec::new();

    while next < total_windows {
        let result = result_rx.recv().await
            .ok_or_else(|| anyhow::anyhow!("result channel closed early"))??;
        pending.insert(result.window_index, result);

        while let Some(result) = pending.remove(&next) {
            full_text.push_str(&result.text);
            full_text.push(' ');
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
```

When VAD is enabled, the collector carries the precomputed `vad_segments` and
attaches them to the final `Transcription`.

---

## Shutdown and errors

Dropping all worker job senders closes the worker channels. Each worker finishes
its current window, exits its receive loop, and drops RKNN contexts on the same
thread that created them.

If a worker returns an error for a window, the worker attaches its worker ID and
window index to the error context and sends that error on the shared result
channel. The collector returns the first error it receives. The long-lived pool
remains available unless a worker channel closes or a worker thread exits.

For daemon use, workers are kept alive across requests. `rkwhisperd` loads one
three-worker pool per enabled model at startup and selects the pool by request
model id.

---

## Optional CPU affinity

NPU core masks do not pin the CPU thread. For lower latency jitter, optionally
pin each worker thread to a big CPU core before loading RKNN contexts. This is
not currently wired into `Cargo.toml`.

```rust
core_affinity::set_for_current(core_affinity::CoreId { id: 4 + worker_id });
```

This keeps CPU-side preprocessing and decoder bookkeeping from migrating while a
worker is processing a window.

---

## Dependencies

```toml
[dependencies]
anyhow = { version = "1", features = ["backtrace"] }
tokio = { version = "1", features = ["macros", "rt", "sync"] }
```

Tokio is not used to run RKNN calls concurrently on the same context. Its role is
coordination: bounded queues, readiness, cancellation, and ordered result
assembly.

---

## Tests and remaining verification

- Unit tests cover result reordering and early-close error reporting.
- Integration-test on RK3588 with long audio:
  - all three workers initialize;
  - each worker reports a distinct core mask;
  - output text and segments remain ordered by window index;
  - throughput improves versus the current serial path.
