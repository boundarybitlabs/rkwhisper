import pytest
import os
from rkwhisper_client import SyncSession, ClientHello

SOCKET_PATH = os.getenv("RKWHISPER_SOCKET", "/run/rkwhisper/asr.sock")
TEST_MODEL = os.getenv("RKWHISPER_TEST_MODEL", "whisper-small-30s")


@pytest.fixture
def client_hello():
    """Default ClientHello for testing."""
    return ClientHello(model=TEST_MODEL, client_id="pytest-integration")

@pytest.fixture
def session_factory():
    """Factory to create a session with custom hello."""
    sessions = []

    def _create(hello=None):
        if hello is None:
            hello = ClientHello(model=TEST_MODEL, client_id="pytest-integration")
        s = SyncSession.connect(SOCKET_PATH, hello)
        sessions.append(s)
        return s


    yield _create

    for s in sessions:
        try:
            s.cancel()  # Best effort cleanup
        except:
            pass


@pytest.fixture
def session(client_hello):
    """A standard session fixture."""
    s = SyncSession.connect(SOCKET_PATH, client_hello)
    yield s
