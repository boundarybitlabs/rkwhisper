import pytest
import os
from rkwhisper_client import SyncSession, ClientHello

SOCKET_PATH = os.getenv("RKWHISPER_SOCKET", "/run/rkwhisper/asr.sock")


def test_connect_success(session):
    """Verify we can connect to a running daemon."""
    assert session is not None
    # Send a tiny bit of audio and finish so the daemon can exit gracefully
    session.send_audio(b"\x00\x00" * 1600)
    session.finish()
    # Drain responses
    for _ in session:
        pass


def test_connect_invalid_model(session_factory):
    """Verify that requesting an unknown model raises an error."""
    hello = ClientHello(model="whisper-invalid-model-name")
    with pytest.raises(RuntimeError) as excinfo:
        session_factory(hello)
    assert "model not found" in str(excinfo.value).lower()


def test_connect_missing_socket():
    """Verify that connecting to a non-existent socket fails."""
    hello = ClientHello(model="whisper-tiny")
    with pytest.raises(RuntimeError):
        SyncSession.connect("/tmp/non_existent_rkwhisper.sock", hello)


def test_connect_retry_on_busy(session_factory, client_hello):
    """Verify that the client automatically retries if the daemon is busy."""
    # 1. Start a session that will keep the daemon busy
    # (Default queue depth is 1, so this will block subsequent connections)
    busy_session = session_factory(client_hello)
    busy_session.send_audio(b"\x00\x00" * 16000 * 5)  # 5 seconds of silence

    # 2. Try to connect a second session.
    # This should hit a BackOff and retry automatically in the Rust layer.
    retry_session = session_factory(client_hello)
    assert retry_session is not None

    # Cleanup
    busy_session.finish()
    retry_session.finish()

    for _ in busy_session:
        pass
    for _ in retry_session:
        pass

def test_split_session(session):
    """Verify that splitting a session works and the halves are independent."""
    sender, receiver = session.split()
    
    # Verify session methods now raise an error
    with pytest.raises(RuntimeError) as excinfo:
        session.send_audio(b"\x00\x00")
    assert "split or closed" in str(excinfo.value)
    
    # Use sender and receiver
    sender.send_audio(b"\x00\x00" * 1600)
    sender.finish()
    
    # Drain receiver
    for _ in receiver:
        pass
