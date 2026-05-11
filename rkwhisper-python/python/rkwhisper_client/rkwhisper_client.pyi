from typing import Optional, Union
from types import TracebackType

class AudioFormat:
    """Audio format requested for a transcription session.

    The daemon currently expects 16 kHz mono signed 16-bit little-endian PCM for
    normal client use. The default constructor values describe that format:
    ``sample_rate=16000``, ``channels=1``, and ``sample_format=1``.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        sample_format: Protocol sample format identifier. ``1`` is signed
            16-bit little-endian PCM.
    """
    sample_rate: int
    channels: int
    sample_format: int
    def __init__(self, sample_rate: int = 16000, channels: int = 1, sample_format: int = 1) -> None:
        """Create an audio format description for ``ClientHello``."""
        ...

class VadOptions:
    """Per-session voice activity detection settings.

    Leave fields as ``None`` to use the daemon's defaults. Set individual fields
    when a session needs different speech detection behavior.

    Attributes:
        threshold: Speech probability threshold.
        min_speech_ms: Minimum speech duration in milliseconds.
        min_silence_ms: Minimum silence duration in milliseconds before a speech
            window is closed.
        speech_pad_ms: Padding added around detected speech in milliseconds.
        window_samples: Number of samples per VAD analysis window.
    """
    threshold: Optional[float]
    min_speech_ms: Optional[int]
    min_silence_ms: Optional[int]
    speech_pad_ms: Optional[int]
    window_samples: Optional[int]
    def __init__(
        self,
        threshold: Optional[float] = None,
        min_speech_ms: Optional[int] = None,
        min_silence_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
        window_samples: Optional[int] = None,
    ) -> None:
        """Create VAD overrides for a session."""
        ...

class ClientHello:
    """Session configuration sent to ``rkwhisperd`` during connection setup.

    ``ClientHello`` chooses the model, decoding behavior, audio format, VAD
    overrides, and an optional client id. A ``ClientHello`` is passed to
    ``SyncSession.connect`` or ``AsyncSession.connect``.

    Attributes:
        model: Model id enabled by the daemon, such as ``"whisper-small-30s"``.
        mode: Session mode. Common values are ``"batch"`` and ``"stream"``.
        lang: Language code used by the decoder, for example ``"en"``.
        task: Whisper task, typically ``"transcribe"`` or ``"translate"``.
        max_new_tokens: Maximum number of decoded tokens per window.
        beam_size: Beam size for decoding. Use ``1`` for greedy decoding.
        notimestamps: Disable timestamp tokens when true.
        suppress_tokens: Token suppression setting, such as ``"default"``.
        audio_format: Requested audio format.
        vad: VAD overrides for this session.
        client_id: Optional human-readable id for logs and diagnostics.
    """
    model: str
    mode: str
    lang: str
    task: str
    max_new_tokens: int
    beam_size: int
    notimestamps: bool
    suppress_tokens: str
    audio_format: AudioFormat
    vad: VadOptions
    client_id: str
    def __init__(
        self,
        model: str,
        mode: str = "batch",
        lang: str = "en",
        task: str = "transcribe",
        max_new_tokens: int = 128,
        beam_size: int = 5,
        notimestamps: bool = False,
        suppress_tokens: str = "default",
        audio_format: Optional[AudioFormat] = None,
        vad: Optional[VadOptions] = None,
        client_id: str = "",
    ) -> None:
        """Create a session handshake message."""
        ...

class Segment:
    """A transcribed text segment returned by the daemon.

    Attributes:
        text: Segment transcript text.
        begin: Segment start time in seconds.
        end: Segment end time in seconds.
    """
    text: str
    begin: float
    end: float

class SpeechStarted:
    """Event emitted when VAD detects the start of speech.

    Attributes:
        begin: Speech start time in seconds.
    """
    begin: float

class SpeechEnded:
    """Event emitted when VAD detects the end of speech.

    Attributes:
        end: Speech end time in seconds.
    """
    end: float

class Done:
    """Final response for a completed transcription session.

    Attributes:
        audio_s: Total processed audio duration in seconds.
        rtf: Real-time factor for the session. Values below ``1.0`` are faster
            than real time.
    """
    audio_s: float
    rtf: float

Response = Union[Segment, SpeechStarted, SpeechEnded, Done]

class SyncAudioSender:
    """Blocking audio sender returned by ``SyncSession.split()``.

    Use this half from the thread responsible for writing audio. Use the paired
    ``SyncResponseReceiver`` from another thread to consume responses while
    audio is still being sent.
    """
    def send_audio(self, pcm: Union[bytes, bytearray, memoryview]) -> None:
        """Send raw PCM bytes to the daemon.

        Args:
            pcm: C-contiguous bytes-like object containing signed 16-bit
                little-endian PCM matching the session audio format.

        Raises:
            RuntimeError: If the buffer is not C-contiguous, the connection
                fails, or the daemon reports a protocol error.
        """
        ...
    def finish(self) -> None:
        """Signal end-of-stream after the final audio chunk."""
        ...
    def cancel(self) -> None:
        """Request cancellation of the current session."""
        ...

class SyncResponseReceiver:
    """Blocking response receiver returned by ``SyncSession.split()``."""
    def recv_response(self) -> Optional[Response]:
        """Receive the next daemon response.

        Returns:
            A ``Segment``, ``SpeechStarted``, ``SpeechEnded``, or ``Done``
            response. ``None`` indicates the iterator-style stream is exhausted.

        Raises:
            RuntimeError: If the daemon returns an error, acknowledges
                cancellation, or the connection/protocol fails.
        """
        ...
    def __iter__(self) -> SyncResponseReceiver:
        """Return ``self`` for blocking iteration over responses."""
        ...
    def __next__(self) -> Response:
        """Return the next response, stopping iteration after ``Done``."""
        ...

class AsyncAudioSender:
    """Async audio sender returned by ``AsyncSession.split()``.

    Use this half from the task responsible for writing audio. Use the paired
    ``AsyncResponseReceiver`` from another task to consume responses while audio
    is still being sent.
    """
    async def send_audio(self, pcm: Union[bytes, bytearray, memoryview]) -> None:
        """Send raw PCM bytes to the daemon.

        Args:
            pcm: C-contiguous bytes-like object containing signed 16-bit
                little-endian PCM matching the session audio format.

        Raises:
            RuntimeError: If the buffer is not C-contiguous, the connection
                fails, or the daemon reports a protocol error.
        """
        ...
    async def finish(self) -> None:
        """Signal end-of-stream after the final audio chunk."""
        ...
    async def cancel(self) -> None:
        """Request cancellation of the current session."""
        ...

class AsyncResponseReceiver:
    """Async response receiver returned by ``AsyncSession.split()``."""
    async def recv_response(self) -> Response:
        """Receive the next daemon response.

        Returns:
            A ``Segment``, ``SpeechStarted``, ``SpeechEnded``, or ``Done``
            response.

        Raises:
            RuntimeError: If the daemon returns an error, acknowledges
                cancellation, or the connection/protocol fails.
        """
        ...
    def __aiter__(self) -> AsyncResponseReceiver:
        """Return ``self`` for async iteration over responses."""
        ...
    async def __anext__(self) -> Response:
        """Return the next response, stopping async iteration after ``Done``."""
        ...

class SyncSession:
    """Blocking session connected to ``rkwhisperd``.

    Use ``connect`` to perform the daemon handshake. A session can be used
    directly for simple request/response flows, or split into independent sender
    and receiver halves for streaming.

    After ``split()`` is called, direct session methods such as ``send_audio``
    and ``recv_response`` are no longer usable.
    """
    @staticmethod
    def connect(socket_path: str, hello: ClientHello) -> SyncSession:
        """Connect to the daemon and start a transcription session.

        Args:
            socket_path: Unix socket path, commonly
                ``"/run/rkwhisper/asr.sock"``.
            hello: Session configuration.

        Raises:
            RuntimeError: If the socket cannot be opened, the model is not
                available, or the handshake fails.
        """
        ...
    def split(self) -> tuple[SyncAudioSender, SyncResponseReceiver]:
        """Split the session into independent blocking sender and receiver halves."""
        ...
    def send_audio(self, pcm: Union[bytes, bytearray, memoryview]) -> None:
        """Send raw PCM bytes using the unsplit session."""
        ...
    def finish(self) -> None:
        """Signal end-of-stream using the unsplit session."""
        ...
    def cancel(self) -> None:
        """Request cancellation using the unsplit session."""
        ...
    def recv_response(self) -> Optional[Response]:
        """Receive the next response using the unsplit session."""
        ...
    def __iter__(self) -> SyncSession:
        """Return ``self`` for blocking iteration over responses."""
        ...
    def __next__(self) -> Response:
        """Return the next response, stopping iteration after ``Done``."""
        ...
    def __enter__(self) -> SyncSession:
        """Enter a context manager for the session."""
        ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit a context manager for the session."""
        ...

class AsyncSession:
    """Async session connected to ``rkwhisperd``.

    Async sessions are split-based. After connecting, call ``split()`` and use
    the returned ``AsyncAudioSender`` and ``AsyncResponseReceiver`` from
    separate tasks or sequential async code.
    """
    @staticmethod
    async def connect(socket_path: str, hello: ClientHello) -> AsyncSession:
        """Connect to the daemon and start an async transcription session.

        Args:
            socket_path: Unix socket path, commonly
                ``"/run/rkwhisper/asr.sock"``.
            hello: Session configuration.

        Raises:
            RuntimeError: If the socket cannot be opened, the model is not
                available, or the handshake fails.
        """
        ...
    def split(self) -> tuple[AsyncAudioSender, AsyncResponseReceiver]:
        """Split the session into independent async sender and receiver halves."""
        ...
    async def __aenter__(self) -> AsyncSession:
        """Enter an async context manager for the session."""
        ...
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit an async context manager for the session."""
        ...
