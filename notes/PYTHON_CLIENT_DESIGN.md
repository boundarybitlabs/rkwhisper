# RKWhisper Python Client Design

## Objective
Provide a high-performance, easy-to-use Python library for interacting with the `rkwhisperd` daemon. The client must maintain the low-latency and high-throughput benefits of the shared-memory architecture while being flexible enough for diverse environments (Web APIs, CLIs, GStreamer).

## Core Requirements

1.  **Shared-Memory Support**: Use `mmap` to attach to the daemon's memfd passed via Unix Domain Sockets (`SCM_RIGHTS`).
2.  **Dual API**:
    *   `rkwhisper.SyncClient`: For standard scripts, CLIs, and synchronous frameworks (Flask, GStreamer).
    *   `rkwhisper.AsyncClient`: For `asyncio` applications (FastAPI, TUIs like Textual).
3.  **Zero-Copy (where possible)**: Minimize PCM data copying when writing to the ring buffer.
4.  **No Heavy Dependencies**: Avoid depending on large libraries like `numpy` unless explicitly requested by the user, to keep the footprint small.

## Proposed API Interface

### 1. Connection & Configuration
```python
from rkwhisper import ClientConfig, SyncClient, AsyncClient

config = ClientConfig(
    model="whisper-small-30s",
    mode="stream",  # or "batch"
    lang="en",
    vad_threshold=0.5
)
```

### 2. Synchronous Usage (CLI/Scripts)
```python
with SyncClient.connect("/run/rkwhisper/asr.sock", config) as client:
    # Stream audio chunks (bytes or array.array)
    for chunk in audio_generator():
        client.send_audio(chunk)
    
    client.finish()
    
    # Collect results
    for segment in client.results():
        print(f"[{segment.start:.2f} -> {segment.end:.2f}]: {segment.text}")
```

### 3. Asynchronous Usage (FastAPI/TUI)
```python
async with AsyncClient.connect("/run/rkwhisper/asr.sock", config) as client:
    async def stream_audio():
        async for chunk in async_audio_source():
            await client.send_audio(chunk)
        await client.finish()

    async def collect_results():
        async for segment in client:
            print(f"Transcription: {segment.text}")

    await asyncio.gather(stream_audio(), collect_results())
```

## Integration Strategies

### REST APIs (FastAPI)
*   The `AsyncClient` should be managed as a dependency or in a connection pool.
*   Streaming responses can be implemented by yielding from the `client.results()` async generator directly to the HTTP response stream.

### GStreamer Plugin
*   A Python-based GStreamer element (`Gst.BaseTransform`) can use `SyncClient` in its `chain` function.
*   Audio buffers from GStreamer are written directly to the `SharedAudioRing` via the client.

### CLIs & TUIs
*   Use `SyncClient` for simple one-shot transcriptions.
*   Use `AsyncClient` with `Textual` or `Prompt Toolkit` to update the UI in real-time as `Segment` messages arrive.

## Technical Implementation Details

The Python client is implemented as a thin PyO3-based wrapper around the `rkwhisper-client` Rust library. This ensures that the high-performance shared-memory logic, atomic synchronization, and protocol framing are handled by the proven Rust implementation.

### 1. Project Structure
*   **Workspace Member**: `rkwhisper-python` (Rust crate).
*   **Bindings**: Uses `PyO3` to expose `SyncSession` and `AsyncSession` classes.
*   **Build System**: `maturin` is used to build and package the library as a native Python extension.

### 2. High-Performance Path
*   **Zero-Copy Handover**: Python `bytes` or `bytearray` objects are passed to Rust, where they are written directly to the shared-memory ring buffer.
*   **Shared Memory**: The Rust client handles the `SCM_RIGHTS` FD retrieval and `mmap` mapping, exposing a simple `send_audio` method to Python.
*   **Async Integration**: The `AsyncClient` uses a dedicated Tokio runtime managed within the Rust extension to bridge Python's `asyncio` with Rust's `async` ecosystem.

### 3. Distribution
*   **Maturin**: Facilitates building `aarch64` wheels for the RK3588 platform.
*   **PEP 517**: Support for standard `pip install .` via a `pyproject.toml` in the `rkwhisper-python` directory.

## Success Criteria
*   Successfully build a Python wheel using `maturin`.
*   Verify that `import rkwhisper` loads the native Rust-backed extension.
*   Achieve transcription latency parity with the native Rust client.
*   Cleanly handle `SIGNAL_CANCEL` and `BackOff` responses via Rust-to-Python error mapping.
