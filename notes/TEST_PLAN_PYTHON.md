# Python Client Test Plan (On-Device Integration)

## Objective
Verify the correctness and performance of the `rkwhisper-client` Python package by running integration tests against a live `rkwhisperd` daemon on the RK3588 platform.

## 1. Environment Requirements
*   **Hardware**: RK3588/RK3588S device.
*   **Daemon**: `rkwhisperd` must be running and listening on a known Unix socket (default: `/run/rkwhisper/asr.sock`).
*   **Models**: At least one model (e.g., `whisper-tiny` or `whisper-small-30s`) must be configured and loaded.
*   **Test Data**: Use the provided `jfkwha-001.wav` in the repository root.

## 2. Test Cases

### A. Connection & Handshake
*   **Default Connection**: Connect to the default socket; verify success.
*   **Model Selection**: Connect requesting a specific model from the `rkwhisperd.toml`; verify the daemon accepts it.
*   **Invalid Model**: Request a model name not present in the daemon's config; verify a `RuntimeError` with the daemon's error message is raised.
*   **Permissions**: Verify behavior when the socket exists but the user lacks permissions (if applicable).

### B. Inference & Accuracy (End-to-End)
*   **Basic Transcription**: Stream `jfkwha-001.wav` to the daemon; verify the returned text contains expected keywords (e.g., "And so my fellow Americans").
*   **Streaming Mode**: Use `mode="stream"` and the `split()` API. Verify that segments are received in a background thread or task while the main task continues sending audio.
*   **Batch Mode**: Use `mode="batch"` and verify the behavior matches the protocol's batch expectations.

### C. Protocol Robustness
*   **Concurrent Send/Receive**: Verify that sending audio and receiving responses can happen simultaneously without deadlock by using the `split()` API.
*   **Ring Buffer Streaming**: Stream a large amount of audio to ensure the ring buffer logic correctly handles multiple wraps and `SIGNAL_DATA_READY` notifications.
*   **Cancellation**: Start an inference and immediately call `sender.cancel()`; verify the daemon reports `Cancelled` and stops processing.
*   **Finish/EOF**: Verify that `sender.finish()` correctly triggers the final transcription and the session closes cleanly.

### D. Parameter Verification
*   **Language/Task**: Verify that passing different `lang` (e.g., "fr") or `task` ("translate") parameters works as expected.
*   **VAD Options**: Pass custom VAD thresholds and verify the daemon accepts them.

## 3. Fixture Design
```python
import pytest
import os
from rkwhisper_client import SyncSession, ClientHello

SOCKET_PATH = os.getenv("RKWHISPER_SOCKET", "/run/rkwhisper/asr.sock")
TEST_MODEL = os.getenv("RKWHISPER_TEST_MODEL", "whisper-small-30s")

@pytest.fixture
def client_hello():
    return ClientHello(model=TEST_MODEL)

@pytest.fixture
def session(client_hello):
    with SyncSession.connect(SOCKET_PATH, client_hello) as s:
        yield s
```

## 4. Execution
```bash
# Set environment variables if different from defaults
export RKWHISPER_TEST_MODEL="whisper-tiny"
pytest rkwhisper-python/tests
```

## 5. Success Criteria
*   Successful transcription of sample audio with the real daemon.
*   Accurate mapping of daemon error messages to Python exceptions.
*   No memory leaks or dangling FDs after multiple session cycles.
