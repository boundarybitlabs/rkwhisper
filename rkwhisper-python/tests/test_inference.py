import wave
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
WAV_PATH = next(FIXTURES_DIR.glob("*.wav"), None)


def test_transcribe_audio(session):
    """End-to-end transcription test using a sample from fixtures."""
    assert WAV_PATH is not None, f"No .wav fixtures found in {FIXTURES_DIR}"
    assert WAV_PATH.exists(), f"Test audio not found at {WAV_PATH}"

    with wave.open(str(WAV_PATH), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2  # 16-bit
        pcm_data = wf.readframes(wf.getnframes())

    # Limit to 10 seconds of audio to avoid OOM on device
    # (16000 samples/sec * 2 bytes/sample * 10 seconds)
    max_bytes = 16000 * 2 * 10
    pcm_data = pcm_data[:max_bytes]

    # Send audio in chunks to simulate streaming
    chunk_size = 16000 * 2  # 1 second of audio
    for i in range(0, len(pcm_data), chunk_size):
        session.send_audio(pcm_data[i : i + chunk_size])

    session.finish()

    results = []
    speech_started = False
    speech_ended = False

    from rkwhisper_client import Segment, SpeechStarted, SpeechEnded

    for resp in session:
        if isinstance(resp, Segment):
            results.append(resp.text)
        elif isinstance(resp, SpeechStarted):
            speech_started = True
        elif isinstance(resp, SpeechEnded):
            speech_ended = True

    full_text = " ".join(results).lower()
    print(f"Transcribed text: {full_text}")

    # Check that we got some non-empty transcription
    assert len(full_text.strip()) > 0
    # verify speech events were received (assuming the audio has speech)
    assert speech_started
    assert speech_ended

