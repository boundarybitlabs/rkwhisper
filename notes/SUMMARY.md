# rkwhisper Code Summary

`rkwhisper` is a Rust implementation of a Whisper transcription pipeline for
Rockchip RKNN/NPU models. It loads separate RKNN models for mel spectrogram
generation, the Whisper encoder, encoder cross-attention key/value generation,
and autoregressive decoder steps, then runs transcription over fixed or
VAD-derived audio windows up to 30 seconds long.

## Crate Shape

- `Cargo.toml` defines a Rust 2024 crate with a library, CLI binary, and Unix
  socket daemon binary.
- `src/lib.rs` exposes the core modules and shared Whisper audio constants.
- `src/bin/rkwhisper.rs` is the command-line entry point.
- `src/bin/rkwhisperd.rs` is the long-running Unix socket daemon.
- `src/daemon.rs` defines daemon config loading, model-file resolution,
  request framing, PCM conversion, and response serialization.
- `src/parallel.rs` owns the three-worker multi-NPU pipeline used by
  `--multi-npu` and by daemon model pools.
- The crate depends on `rknpu2` for RKNN runtime access, `tokenizers` for
  Whisper tokenization, `hound` for WAV loading, `tokio` for worker
  coordination channels/runtime, `toml` for daemon config, and `clap` for CLI
  parsing.

## Runtime Pipeline

The CLI accepts:

- model family: `tiny`, `base`, `small`, `medium`, or `large-v3-turbo`
- tokenizer path
- RKNN model paths for mel spectrogram, encoder, encoder-KV, and decoder
- input mono 16 kHz WAV file
- language/task options
- decoding controls such as `max_new_tokens`, `beam_size`, and timestamp
  suppression
- output format: text or JSON
- token suppression mode: default, none, or explicit token IDs
- optional VAD model/configuration
- optional `--multi-npu` execution across three NPU workers

At runtime, `src/bin/rkwhisper.rs`:

1. Finds the RKNN runtime library with `rknpu2::utils::find_rknn_library`.
2. Loads each `.rknn` model into an `RKNN<RuntimeAPI>` context.
3. Selects the model specification type from `src/spec.rs`.
4. Calls either the serial `whisper::transcribe_with_options` path or the
   multi-NPU `parallel::transcribe_file_parallel_with_options` path.

The serial `whisper` path processes audio in fixed 30-second windows unless a
VAD model is provided. With VAD, speech segments are split into windows no
longer than `N_SAMPLES`, where `N_SAMPLES` is 30 seconds at 16 kHz:

1. Load and validate WAV input.
2. Optionally run VAD and derive speech windows.
3. Generate log-mel features with the mel RKNN model.
4. Run the encoder RKNN model.
5. Run the encoder-KV RKNN model to compute per-layer cross-attention K/V.
6. Prime the decoder with Whisper control tokens.
7. Generate tokens using greedy decoding or beam search.
8. Decode generated token IDs back to text and timestamped segments.
9. Concatenate ordered window outputs.

The multi-NPU path loads one full RKNN context set per worker, pins each worker
to a distinct RK3588 NPU core mask, dynamically dispatches ready workers to
pending windows, and reorders results by window index.

## Daemon Runtime

`src/bin/rkwhisperd.rs` runs a Unix socket ASR daemon. It:

1. Loads `/etc/rkwhisper.toml` by default, or a path supplied with `--config`.
2. Resolves enabled model directories under `RKWHISPER_MODEL_ROOT` or
   `/usr/share/rkwhisper`.
3. Creates one persistent `ParallelTranscriberPool` per enabled model.
4. Accepts length-prefixed Protobuf `ClientHello` control messages.
5. Creates a `memfd` shared-memory ring for 16 kHz mono s16le PCM and passes the
   file descriptor to clients with `SCM_RIGHTS`.
6. Supports batch requests as a degenerate stream and live streaming requests.
7. Emits length-prefixed Protobuf `segment`, `done`, or `error` responses.

Each configured model directory must contain `tokenizer.json`, `mel.rknn`,
`encoder.rknn`, `enc_kv.rknn`, and `decoder.rknn`. If `vad.rknn` is present,
batch requests can use it for VAD-derived windows.

## Audio and Mel Frontend

`src/lib.rs` contains the audio constants and frontend wrapper:

- `SAMPLE_RATE = 16_000`
- `N_SAMPLES = 480_000`
- `N_FFT = 400`
- `HOP_LENGTH = 160`
- `N_MELS = 80`
- `N_FRAMES = 3000`

`load_audio_file` accepts only mono 16 kHz WAV files. Integer PCM is normalized
to `[-1, 1]`; floating-point WAV samples are used directly.

`MelSpectrogram::log_mel_spectrogram` pads each chunk to 30 seconds, applies
`polyphase_pre_process`, runs the mel RKNN model, converts the output to
log10 space, clamps the dynamic range to `max - 8`, normalizes with
`(x + 4) / 4`, and prints basic spectrogram statistics.

`polyphase_pre_process` reflect-pads the input by 200 samples on both sides,
adds alignment padding to a multiple of 80, and transposes the data into an
80-channel polyphase layout expected by the mel model.

## Model Specifications

`src/spec.rs` defines the `WhisperSpec` trait. It provides compile-time model
dimensions used throughout the encoder, decoder, KV cache, and logits buffers:

- mel bin count and frame count
- encoder sequence length and hidden size
- decoder layer/head/head-dimension counts
- self-attention cache length
- vocabulary size and EOT token ID

Implemented specs:

- `WhisperTiny`
- `WhisperBase`
- `WhisperSmall`
- `WhisperMedium`
- `WhisperLargeV3Turbo`

These type-level specs let the same pipeline compile for different Whisper
model sizes without runtime shape branching inside the hot paths.

## Encoder and Encoder-KV

`src/encoder.rs` contains two RKNN wrappers.

`WhisperEncoder<S>`:

- accepts a log-mel buffer
- pads or truncates it to `S::MEL_BINS * S::FRAMES`
- runs the encoder RKNN model
- returns `Encoded`, an f16 buffer shaped as `[1, ENC_SEQ, HIDDEN]`

`EncKvModel<S>`:

- accepts encoder hidden states
- runs a separate RKNN model that computes cross-attention K/V tensors
- returns per-layer encoder K and V buffers in logical
  `[1, ENC_SEQ, N_HEADS, D_HEAD]` layout
- expects one K and one V output per decoder layer

The decoder wrapper packs the encoder-KV output into the decoder RKNN graph's
native `NC1HWC2` `enc_k_l*` and `enc_v_l*` input layout.

## Decoder and KV Cache

`src/decoder.rs` implements per-token autoregressive decoding.

`WhisperDecoderState` owns the mutable self-attention cache:

- `past_k`
- `past_v`
- `pos`

The state is cloneable so beam search can branch independently. Each cache
buffer is stored per layer in native `NC1HWC2` layout
`[1, ceil(N_HEADS/8), T_CACHE, D_HEAD, 8]`.

`WhisperDecoder<S>` owns:

- decoder RKNN context reference
- per-layer encoder cross-attention K/V
- self-attention and KV update masks
- scalar token and position inputs
- reusable logits buffer

For each token step, it:

1. Builds masks for causal attention, insertion, and retention.
2. Binds token, past K/V, encoder K/V, masks, and position into the RKNN graph.
3. Runs the decoder model.
4. Reads logits and per-layer single-token present K/V.
5. Writes present K/V back into the ring-style cache slot.
6. Increments the decoder position.

Helper functions copy packed decoder present outputs back into the packed cache
slot for the next token step.

## Decoding

`src/whisper.rs` builds the Whisper control prompt:

```text
<|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]
```

It supports two token selection modes:

- greedy decoding when `beam_size == 1`
- beam search when `beam_size > 1`

Both modes suppress control tokens after the prompt. Timestamp tokens are only
suppressed when `--notimestamps` is set. EOT is suppressed for the first
generated token. A repeated 4-token pattern is used as a simple loop-break
condition.

## Beam Search

`src/beam.rs` implements a generic beam search over `WhisperSpec`.

Each `Beam` stores:

- generated tokens
- cumulative log probability
- cloned decoder state
- last logits
- finished flag

`BeamSearch::step` applies the caller's token suppression function, computes
log-softmax, keeps the top candidates, advances decoder states for unfinished
beams, and moves EOT or repeated-pattern beams into `finished_beams`.

Final ranking uses length-normalized score:

```text
log_prob / len^alpha
```

The transcription pipeline currently passes `alpha = 0.6`.

## Additional Cache Type

`src/cache.rs` defines `PackedKv`, an alternate packed K/V cache structure with
tests. It stores K and V as flat `[B, L, H, T, D]`-style buffers and supports:

- clearing
- offset calculation
- writing a single present step
- writing all layers
- left-compacting the cache when full
- exposing raw slices for RKNN input binding

This module is public but is not currently used by the main decoder path, which
uses `WhisperDecoderState` instead.

## Type Markers

`src/markers.rs` defines phantom-data marker types for units and tensor shapes,
including sample rate, channel count, log-mel dimensions, encoder input/output,
and logits. These types are not currently exported from `src/lib.rs` and are
not wired into the active pipeline.

## Tests

The test suite covers:

- packed cache length and layout
- base offset math
- writing one layer
- writing all layers
- left compaction
- slice views
- fill/compact/append behavior
- daemon Protobuf framing, PCM conversion, config parsing, and model resolution
- suppression mode parsing
- VAD segment merging, padding, and short-speech filtering
- fixed and VAD-derived transcription windowing
- timestamp token conversion
- parallel result reordering and early-close error reporting

Most of the active RKNN pipeline is integration-oriented and depends on actual
RKNN model files, tokenizer files, and Rockchip runtime availability.

## Design Note

`MULTI_THREADED_PIPELINE.md` describes the implemented Tokio-coordinated
multi-NPU pipeline. The CLI can opt into it with `--multi-npu`, while the daemon
uses persistent multi-NPU pools for all configured models.

## Important Operational Assumptions

- Input audio must be mono, 16 kHz WAV.
- Serial and parallel batch modes process fixed 30-second windows unless VAD
  produces speech-derived windows.
- Live daemon stream mode buffers incoming s16le PCM into 30-second windows and
  dispatches a final shorter window at end-of-stream.
- RKNN model I/O tensor layouts must match the documented assumptions in
  `encoder.rs` and `decoder.rs`.
- The tokenizer must contain Whisper special tokens such as
  `<|startoftranscript|>`, language tokens, task tokens, and
  `<|notimestamps|>`.
- Running the full application requires a compatible Rockchip RKNN runtime and
  model files for the selected Whisper spec.
