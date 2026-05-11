import pytest
import wave
import os
import asyncio
from pathlib import Path
from rkwhisper_client import ClientHello, Segment, SpeechStarted, SpeechEnded

TEST_MODEL = os.getenv("RKWHISPER_TEST_MODEL", "whisper-small-30s")
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
WAV_PATH = next(FIXTURES_DIR.glob("*.wav"), None)

@pytest.mark.asyncio
async def test_async_transcribe_audio(async_session_factory):
    """End-to-end async transcription test."""
    hello = ClientHello(
        model=TEST_MODEL, client_id="pytest-async-stream", mode="stream"
    )
    session = await async_session_factory(hello)
    sender, receiver = session.split()
    
    assert WAV_PATH is not None, f"No .wav fixtures found in {FIXTURES_DIR}"
    
    with wave.open(str(WAV_PATH), "rb") as wf:
        assert wf.getframerate() == 16000
        pcm_data = wf.readframes(wf.getnframes())

    # Truncate to 5 seconds for speed
    max_bytes = 16000 * 2 * 5
    pcm_data = pcm_data[:max_bytes]

    # Send audio in chunks
    chunk_size = 16000 * 2 # 1 second
    for i in range(0, len(pcm_data), chunk_size):
        await sender.send_audio(pcm_data[i : i + chunk_size])
    
    await sender.finish()

    results = []
    speech_started = False
    speech_ended = False

    # Use the async iterator protocol
    async for resp in receiver:
        if isinstance(resp, Segment):
            results.append(resp.text)
        elif isinstance(resp, SpeechStarted):
            speech_started = True
        elif isinstance(resp, SpeechEnded):
            speech_ended = True

    full_text = " ".join(results).lower()
    print(f"Async transcribed text: {full_text}")
    
    assert len(full_text.strip()) > 0
    assert speech_started
    assert speech_ended

@pytest.mark.asyncio
async def test_async_connect_invalid_model(async_session_factory):
    """Verify async connect raises error for unknown model."""
    from rkwhisper_client import AsyncSession
    
    hello = ClientHello(model="whisper-invalid-async")
    socket_path = os.getenv("RKWHISPER_SOCKET", "/run/rkwhisper/asr.sock")
    
    with pytest.raises(RuntimeError) as excinfo:
        await AsyncSession.connect(socket_path, hello)
    assert "model not found" in str(excinfo.value).lower()
t_path = os.getenv("RKWHISPER_SOCKET", "/run/rkwhisper/asr.sock")
    
    with pytest.raises(RuntimeError) as excinfo:
        await AsyncSession.connect(socket_path, hello)
    assert "model not found" in str(excinfo.value).lower()
