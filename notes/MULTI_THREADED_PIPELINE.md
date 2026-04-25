# RK3588 RKNN Whisper pipeline — multithreaded design

## Overview

The RK3588 NPU exposes three independent 2 TOPS cores. Rather than fusing all three into a single 6 TOPS context for one large model, this design runs **three independent Whisper pipelines in parallel**, each pinned to a different core and each processing a different audio window. Throughput scales linearly with the number of cores.

A Rayon thread pool with exactly `num_threads(3)` manages dispatch. Each worker thread owns one pair of RKNN contexts (encoder + decoder) stored in `thread_local!` storage, permanently bound to its assigned NPU core via `rknn_set_core_mask()`.

---

## Pipeline stages (per core)

Each audio window passes through three stages in sequence:

| Stage | Where | Description |
|---|---|---|
| Mel spectrogram | CPU | Log-mel filterbank, 80 bins × 3000 frames (30 s @ 16 kHz) |
| Encoder | NPU | Single forward pass → cross-attention KV cache |
| Decoder loop | NPU | Autoregressive token generation until `<EOT>` |

The encoder runs once per window. The decoder runs N times (one step per output token), so windows with verbose audio take proportionally longer — this is why work-stealing matters.

---

## Thread pool construction

```rust
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

thread_local! {
    static WHISPER_CTX: RefCell<Option<WhisperCtx>> = RefCell::new(None);
}

static CORE_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn build_whisper_pool(
    encoder_model: Arc<Vec<u8>>,
    decoder_model: Arc<Vec<u8>>,
) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .start_handler(move |_| {
            let core_id = CORE_COUNTER.fetch_add(1, Ordering::SeqCst) % 3;
            let core_mask = 1u32 << core_id;
            WHISPER_CTX.with(|cell| {
                *cell.borrow_mut() = Some(
                    WhisperCtx::new(&encoder_model, &decoder_model, core_mask, core_id)
                        .expect("RKNN init failed"),
                );
            });
        })
        .exit_handler(|_| {
            WHISPER_CTX.with(|cell| drop(cell.borrow_mut().take()));
        })
        .build()
        .unwrap()
}
```

`num_threads(3)` is load-bearing — if the pool grows larger, multiple threads would contend for the same NPU core. `start_handler` fires once per worker thread at pool creation, so the pool is fully ready (all three RKNN contexts initialized) before any work is submitted.

---

## RKNN context wrapper

```rust
pub struct WhisperCtx {
    encoder: RknnHandle,
    decoder: RknnHandle,
    pub core_id: usize,
}

// SAFETY: each WhisperCtx lives entirely on its owning Rayon worker thread.
// thread_local! guarantees no concurrent access.
unsafe impl Send for WhisperCtx {}

impl WhisperCtx {
    pub fn new(
        encoder_model: &[u8],
        decoder_model: &[u8],
        core_mask: u32,
        core_id: usize,
    ) -> anyhow::Result<Self> {
        let encoder = RknnHandle::load(encoder_model)?;
        encoder.set_core_mask(core_mask)?;

        let decoder = RknnHandle::load(decoder_model)?;
        decoder.set_core_mask(core_mask)?;

        Ok(Self { encoder, decoder, core_id })
    }

    pub fn transcribe(&mut self, window: &AudioWindow) -> String {
        let mel = compute_log_mel(&window.samples, 80, 400, 160);
        let kv_cache = self.encoder.run(&[&mel]).expect("encoder failed");

        let mut tokens: Vec<i32> = vec![SOT_TOKEN, LANG_TOKEN, TRANSCRIBE_TOKEN];
        loop {
            let logits = self.decoder
                .run(&[&kv_cache, &tokens_tensor(&tokens)])
                .expect("decoder step failed");
            let next = greedy_sample(&logits);
            if next == EOT_TOKEN || tokens.len() > 448 { break; }
            tokens.push(next);
        }

        decode_bpe(&tokens[3..])
    }
}
```

---

## Streaming dispatch

```rust
pub struct AudioWindow {
    pub id: u64,           // monotonic sequence number for reordering
    pub samples: Vec<f32>,
    pub timestamp_ms: u64,
}

pub fn run_pipeline(
    pool: &rayon::ThreadPool,
    window_rx: Receiver<AudioWindow>,
    result_tx: Sender<(u64, String)>,
) {
    pool.scope(|s| {
        for window in window_rx {
            let result_tx = result_tx.clone();
            s.spawn(move |_| {
                let transcript = WHISPER_CTX.with(|cell| {
                    cell.borrow_mut()
                        .as_mut()
                        .expect("thread has no WhisperCtx")
                        .transcribe(&window)
                });
                result_tx.send((window.id, transcript)).unwrap();
            });
        }
    });
}
```

`rayon::scope` blocks until all spawned tasks complete. Work-stealing means that if one core is running a long decoder (verbose audio), another core immediately picks up the next pending window — a round-robin dispatch would leave cores idle in this situation.

---

## Result reordering

Work-stealing may complete windows out of order. A min-heap keyed on `window.id` reconstructs the correct sequence:

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub fn collect_ordered(result_rx: Receiver<(u64, String)>, total: usize) -> Vec<String> {
    let mut heap: BinaryHeap<(Reverse<u64>, String)> = BinaryHeap::new();
    let mut out = Vec::with_capacity(total);
    let mut next_expected = 0u64;

    for (id, text) in result_rx.iter().take(total) {
        heap.push((Reverse(id), text));
        while heap.peek().map(|(Reverse(id), _)| *id) == Some(next_expected) {
            let (_, t) = heap.pop().unwrap();
            out.push(t);
            next_expected += 1;
        }
    }
    out
}
```

---

## Key design decisions

**`thread_local!` instead of `Arc<Mutex<_>>`.**
RKNN contexts are not thread-safe. Wrapping them in a mutex would serialize all NPU access, defeating the purpose. `thread_local!` gives each worker its own context with zero synchronization overhead — the context never moves or is shared.

**`AtomicUsize` for core assignment.**
Rayon starts all three threads roughly simultaneously, so thread indices may not arrive at `start_handler` in order. `fetch_add` mod 3 guarantees each thread gets a unique core ID regardless of scheduling order.

**Separate encoder/decoder models.**
Splitting Whisper into `encoder.rknn` and `decoder.rknn` (the standard RKNN port layout) means both models on a given thread share the same `core_mask`. All compute for one audio window stays on a single NPU core, avoiding cross-core cache pressure.

**`num_threads(3)` is a hard constraint.**
Increasing thread count beyond 3 would cause multiple threads to share a core mask, creating contention inside the RKNN runtime. If you need a larger Rayon pool elsewhere in your application, build this as an isolated `ThreadPool` instance rather than using the global pool.

---

## Optional: CPU core affinity

Rayon workers can be rescheduled by the OS onto any CPU core. To also pin each worker to a specific A76 or A55 cluster, call `core_affinity::set_for_current()` inside `start_handler` before the RKNN init:

```rust
.start_handler(move |_| {
    let core_id = CORE_COUNTER.fetch_add(1, Ordering::SeqCst) % 3;
    // Pin to a big core (A76 cluster starts at logical core 4 on RK3588)
    core_affinity::set_for_current(core_affinity::CoreId { id: 4 + core_id });
    // ... then init RKNN context as normal
})
```

This reduces latency jitter on the mel spectrogram computation (which runs on CPU) by preventing the scheduler from migrating the thread mid-window.

---

## Dependencies

```toml
[dependencies]
rayon            = "1.8"
crossbeam-channel = "0.5"
anyhow           = "1"
core_affinity    = "0.8"   # optional, for CPU pinning
```

RKNN bindings: use `rknn-rs` if available for your SDK version, or write thin `unsafe` FFI wrappers over `librknnrt.so` directly.
