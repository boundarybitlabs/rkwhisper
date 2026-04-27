# rkwhisper

`rkwhisper` is a Rust Whisper transcription pipeline for Rockchip RKNN/NPU
models. It runs separate RKNN graphs for mel spectrogram generation, the Whisper
encoder, encoder cross-attention K/V generation, and autoregressive decoder
steps.

The crate provides:

- `rkwhisper`: a CLI for mono 16 kHz WAV transcription.
- `rkwhisper --multi-npu`: a three-worker RK3588 pipeline that uses the three
  NPU cores concurrently.
- `rkwhisperd`: a Unix socket daemon with persistent multi-NPU model pools.

## Requirements

- Rockchip hardware and runtime support compatible with `rknpu2`.
- RKNN model files for the selected Whisper model size.
- A Whisper `tokenizer.json`.
- Rust with edition 2024 support.
- CLI input audio as mono 16 kHz WAV.

The full transcription path requires the RKNN runtime library and model files to
be available on the target system.

## Build

```sh
cargo build --release
cargo test
```

On non-Rockchip development machines, pure Rust tests may be useful, but full
application runs require a compatible Rockchip RKNN runtime.

## Model Files

The CLI accepts model paths explicitly. The daemon resolves enabled models from a
model root, which defaults to `RKWHISPER_MODEL_ROOT` or `/usr/share/rkwhisper`.

Daemon model directories use this layout:

```text
/usr/share/rkwhisper/whisper-small-30s/
  tokenizer.json
  mel.rknn
  encoder.rknn
  enc_kv.rknn
  decoder.rknn
  vad.rknn        # optional
```

Supported model ids are:

- `whisper-tiny` or `whisper-tiny-30s`
- `whisper-base` or `whisper-base-30s`
- `whisper-small` or `whisper-small-30s`
- `whisper-medium` or `whisper-medium-30s`
- `whisper-large-v3-turbo` or `whisper-large-v3-turbo-30s`

Each directory name must match the model id used by daemon requests and config.

## CLI Usage

Basic serial transcription:

```sh
cargo run --release --bin rkwhisper -- \
  --model small \
  --tokenizer /models/whisper-small-30s/tokenizer.json \
  --mel-spec /models/whisper-small-30s/mel.rknn \
  --encoder /models/whisper-small-30s/encoder.rknn \
  --enc-kv /models/whisper-small-30s/enc_kv.rknn \
  --decoder /models/whisper-small-30s/decoder.rknn \
  input.wav
```

JSON output:

```sh
cargo run --release --bin rkwhisper -- \
  --model small \
  --output json \
  --tokenizer /models/whisper-small-30s/tokenizer.json \
  --mel-spec /models/whisper-small-30s/mel.rknn \
  --encoder /models/whisper-small-30s/encoder.rknn \
  --enc-kv /models/whisper-small-30s/enc_kv.rknn \
  --decoder /models/whisper-small-30s/decoder.rknn \
  input.wav
```

Use all three RK3588 NPU cores:

```sh
cargo run --release --bin rkwhisper -- \
  --model small \
  --multi-npu \
  --tokenizer /models/whisper-small-30s/tokenizer.json \
  --mel-spec /models/whisper-small-30s/mel.rknn \
  --encoder /models/whisper-small-30s/encoder.rknn \
  --enc-kv /models/whisper-small-30s/enc_kv.rknn \
  --decoder /models/whisper-small-30s/decoder.rknn \
  input.wav
```

Optional VAD:

```sh
cargo run --release --bin rkwhisper -- \
  --model small \
  --multi-npu \
  --vad-model /models/whisper-small-30s/vad.rknn \
  --vad-threshold 0.5 \
  --vad-min-speech-ms 250 \
  --vad-min-silence-ms 100 \
  --vad-speech-pad-ms 200 \
  --tokenizer /models/whisper-small-30s/tokenizer.json \
  --mel-spec /models/whisper-small-30s/mel.rknn \
  --encoder /models/whisper-small-30s/encoder.rknn \
  --enc-kv /models/whisper-small-30s/enc_kv.rknn \
  --decoder /models/whisper-small-30s/decoder.rknn \
  input.wav
```

Useful decoding flags:

- `--lang en`
- `--task transcribe` or `--task translate`
- `--max-new-tokens 128`
- `--beam-size 1` for greedy decoding, or higher for beam search
- `--notimestamps`
- `--suppress-tokens default`, `none`, or a comma-separated token id list

## Daemon Usage

Create a config listing enabled model ids:

```toml
# /etc/rkwhisper.toml
models = [
  "whisper-small-30s",
  "whisper-medium-30s",
]
```

Run the daemon with defaults:

```sh
cargo run --release --bin rkwhisperd
```

Defaults:

- config: `/etc/rkwhisper.toml`
- model root: `RKWHISPER_MODEL_ROOT` or `/usr/share/rkwhisper`
- socket: `/run/rkwhisper/asr.sock`

Override paths:

```sh
cargo run --release --bin rkwhisperd -- \
  --config ./rkwhisper.toml \
  --model-root /models/rkwhisper \
  --socket /tmp/rkwhisper.sock
```

`rkwhisperd` creates one persistent three-worker `ParallelTranscriberPool` per
enabled model. For packaged installs, run the service as the `rkwhisper` user
and group and let systemd create `/run/rkwhisper` with `RuntimeDirectory`.
`rkwhisperd` binds the socket and sets its mode to `0660`; it does not change
socket ownership at startup.

A typical service setup uses:

```ini
[Service]
User=rkwhisper
Group=rkwhisper
RuntimeDirectory=rkwhisper
RuntimeDirectoryMode=0750
UMask=0007
ExecStart=/usr/bin/rkwhisperd --socket /run/rkwhisper/asr.sock
```

## Daemon Protocol

Clients send a newline-terminated JSON header followed by one or more framed
s16le PCM payloads.

Batch requests send one PCM frame:

```text
{"model":"whisper-small-30s","mode":"batch","beam_size":5}
\n
<4-byte little-endian byte length><s16le PCM bytes>
```

Stream requests send multiple PCM frames and close the write side when done:

```text
{"model":"whisper-small-30s","mode":"stream"}
\n
<frame length><s16le PCM bytes>
<frame length><s16le PCM bytes>
...
```

Responses are newline-delimited JSON:

```json
{"type":"segment","text":"hello world","begin":0.0,"end":1.2}
{"type":"done","audio_s":30.0,"rtf":0.42}
```

Errors are also returned as one-line JSON:

```json
{"type":"error","error":"model not found"}
```

Request header defaults:

- `mode`: `batch`
- `lang`: `en`
- `task`: `transcribe`
- `max_new_tokens`: `128`
- `beam_size`: `5`
- `notimestamps`: `false`
- `suppress_tokens`: `default`

The header may also include VAD overrides:

- `vad_threshold`
- `vad_min_speech_ms`
- `vad_min_silence_ms`
- `vad_speech_pad_ms`
- `vad_window_samples`

## How It Works

Audio is split into fixed 30-second windows or VAD-derived speech windows. Each
window is converted to log-mel features, encoded, converted into encoder
cross-attention K/V tensors, and decoded token by token with greedy decoding or
beam search.

In multi-NPU mode, each worker owns a full RKNN context set pinned to one RK3588
NPU core. Tokio coordinates ready workers, dispatches pending windows, and
reorders completed results by window index.

For more detail:

- [Code summary](notes/SUMMARY.md)
- [Multi-NPU architecture](notes/MULTI_THREADED_PIPELINE.md)
