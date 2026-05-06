import pytest
import os
from rkwhisper_client import SyncSession, AsyncSession, ClientHello

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
            # If the session was split, cancel won't work on the session object
            # but that's okay for cleanup.
            s.cancel() 
        except Exception:
            pass


@pytest.fixture
async def async_session_factory():
    """Factory to create an async session with custom hello."""
    sessions = []

    async def _create(hello=None):
        if hello is None:
            hello = ClientHello(model=TEST_MODEL, client_id="pytest-async-integration")
        s = await AsyncSession.connect(SOCKET_PATH, hello)
        sessions.append(s)
        return s

    yield _create

    for s in sessions:
        try:
            # AsyncSession methods now return Err if split, which future_into_py 
            # turns into a Python exception.
            await s.cancel()
        except Exception:
            pass


@pytest.fixture
def session(client_hello):
    """A standard sync session fixture."""
    with SyncSession.connect(SOCKET_PATH, client_hello) as s:
        yield s


@pytest.fixture
async def async_session(client_hello):
    """A standard async session fixture."""
    async with await AsyncSession.connect(SOCKET_PATH, client_hello) as s:
        yield s
