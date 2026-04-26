# RK3588 RKNN Whisper pipeline - Tokio multi-NPU design

## Overview

The RK3588 NPU exposes three independent cores. This design runs three
independent Whisper pipelines in parallel, with one long-lived worker per NPU
core. Tokio coordinates input windows, worker readiness, backpressure, result
collection, and shutdown; RKNN inference itself remains synchronous and
worker-local.

The important rule is ownership: each worker owns a full RKNN context set for
one core:

- mel spectrogram
- encoder
- encoder-KV
- decoder RKNN handle plus `WhisperDecoder`

Do not share RKNN contexts with `Arc<Mutex<_>>`. The current `rknpu2::RKNN`
handle is a thin wrapper around an RKNN context and should be treated as
thread-affine. All calls to `set_inputs`, `run`, and `get_outputs` happen on the
same worker thread that created and pinned the contexts.

---

## Pipeline stages

VAD runs before fanout because it determines the audio windows. Once windows
exist, each window is independent and can be processed by any available NPU
worker.

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

For RK3588, start exactly three workers:

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
remains generic over `WhisperSpec`.

```rust
struct PipelineCtx<S: WhisperSpec> {
    core_id: usize,
    mel_spec: MelSpectrogram,
    encoder: WhisperEncoder<S>,
    enc_kv: EncKvModel<S>,
    decoder_rknn: RKNN<RuntimeAPI>,
    decoder: WhisperDecoder<'static, S>,
}
```

In real code, avoid forcing a fake `'static` lifetime. Prefer one of these
shapes:

- make `WhisperDecoder<S>` own its `RKNN<RuntimeAPI>` instead of borrowing it, or
- store `decoder_rknn` and construct a short-lived `WhisperDecoder` inside
  `transcribe_window`.

`PipelineCtx::transcribe_window(...)` should contain the current per-window body
from `transcribe_audio_with_options`: log-mel, encode, enc-KV, prompt, greedy or
beam decode, token decode, and `TranscriptSegment` construction.

---

## Tokio orchestration

Tokio owns the control plane:

- one bounded job channel per worker
- one shared ready channel
- one shared result channel
- one dispatcher task
- one collector/reorder task

The workers themselves run on dedicated OS threads. Do not use Tokio's general
worker pool for RKNN inference, because the model contexts must remain attached
to their owning thread and NPU core.

```rust
struct Job {
    window: AudioWindow,
    audio: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    options: Arc<TranscribeOptions>,
}

struct Ready {
    worker_id: usize,
}

struct WindowResult {
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
    model_bytes: Arc<ModelBytes>,
    ready_tx: tokio::sync::mpsc::Sender<Ready>,
    result_tx: tokio::sync::mpsc::Sender<anyhow::Result<WindowResult>>,
) -> tokio::sync::mpsc::Sender<Job> {
    let (job_tx, mut job_rx) = tokio::sync::mpsc::channel::<Job>(1);

    std::thread::Builder::new()
        .name(format!("rkwhisper-npu-{worker_id}"))
        .spawn(move || {
            let mut ctx = PipelineCtx::<S>::load(model_bytes, core_mask, worker_id)?;

            ready_tx.blocking_send(Ready { worker_id })?;

            while let Some(job) = job_rx.blocking_recv() {
                let result = ctx.transcribe_window(job);
                result_tx.blocking_send(result)?;
                ready_tx.blocking_send(Ready { worker_id })?;
            }

            anyhow::Ok(())
        })
        .expect("failed to spawn NPU worker");

    job_tx
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
    mut ready_rx: tokio::sync::mpsc::Receiver<Ready>,
    audio: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    options: Arc<TranscribeOptions>,
) -> anyhow::Result<()> {
    let mut pending = std::collections::VecDeque::from(windows);

    while let Some(window) = pending.pop_front() {
        let ready = ready_rx.recv().await
            .ok_or_else(|| anyhow::anyhow!("all NPU workers stopped"))?;

        let job = Job {
            window,
            audio: audio.clone(),
            tokenizer: tokenizer.clone(),
            options: options.clone(),
        };

        worker_txs[ready.worker_id].send(job).await?;
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
    mut result_rx: tokio::sync::mpsc::Receiver<anyhow::Result<WindowResult>>,
    total_windows: usize,
) -> anyhow::Result<Transcription> {
    let mut pending = std::collections::BTreeMap::<usize, WindowResult>::new();
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
        vad_segments: Vec::new(),
    })
}
```

When VAD is enabled, carry the precomputed `vad_segments` outside this collector
and attach them to the final `Transcription`.

---

## Shutdown and errors

Dropping all worker job senders closes the worker channels. Each worker finishes
its current window, exits its receive loop, and drops RKNN contexts on the same
thread that created them.

If any worker returns an error, the coordinator should:

1. drop all job senders to stop future work;
2. drain or close the result channel;
3. return the first error with the worker ID and window index attached.

For daemon use, keep workers alive across requests only when the model is fixed.
If requests can select different models, use a pool key of `(model_id,
model_kind)` and create one three-worker pool per loaded model, or keep the v1
daemon request-scoped and accept model reload cost.

---

## Optional CPU affinity

NPU core masks do not pin the CPU thread. For lower latency jitter, optionally
pin each worker thread to a big CPU core before loading RKNN contexts:

```rust
core_affinity::set_for_current(core_affinity::CoreId { id: 4 + worker_id });
```

This keeps CPU-side preprocessing and decoder bookkeeping from migrating while a
worker is processing a window.

---

## Dependencies

```toml
[dependencies]
anyhow = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync"] }
core_affinity = "0.8" # optional
```

Tokio is not used to run RKNN calls concurrently on the same context. Its role is
coordination: bounded queues, readiness, cancellation, and ordered result
assembly.

---

## Test plan

- Unit-test the dispatcher with fake workers that complete windows out of order.
- Unit-test result reordering for already-ordered, reversed, and delayed
  results.
- Unit-test bounded worker queues so each worker has at most one queued job.
- Integration-test on RK3588 with long audio:
  - all three workers initialize;
  - each worker reports a distinct core mask;
  - output text and segments remain ordered by window index;
  - throughput improves versus the current serial path.
