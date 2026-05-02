import pytest
import os
from rkwhisper_client import SyncSession, ClientHello

SOCKET_PATH = os.getenv("RKWHISPER_SOCKET", "/run/rkwhisper/asr.sock")


def test_connect_success(session):
    """Verify we can connect to a running daemon."""
    assert session is not None


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
