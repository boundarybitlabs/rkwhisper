from typing import Optional, List, Union, Iterator, AsyncIterator, Any
from types import TracebackType

class AudioFormat:
    """Audio configuration for the transcription session."""
    sample_rate: int
    channels: int
    sample_format: int
    def __init__(self, sample_rate: int = 16000, channels: int = 1, sample_format: int = 1) -> None: ...

class VadOptions:
    """Voice Activity Detection configuration."""
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
    ) -> None: ...

class ClientHello:
    """Handshake message sent to the daemon to start a session."""
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
    ) -> None: ...

class Segment:
    """A transcribed segment of text."""
    text: str
    begin: float
    end: float

class SpeechStarted:
    """Event indicating speech detection has started at the given timestamp."""
    begin: float

class SpeechEnded:
    """Event indicating speech detection has ended at the given timestamp."""
    end: float

class Done:
    """Final response indicating completion of the transcription job."""
    audio_s: float
    rtf: float

Response = Union[Segment, SpeechStarted, SpeechEnded, Done]

class SyncSession:
    """Synchronous session for communicating with the rkwhisperd daemon."""
    @staticmethod
    def connect(socket_path: str, hello: ClientHello) -> SyncSession: ...
    def send_audio(self, pcm: Union[bytes, bytearray, memoryview]) -> None: ...
    def finish(self) -> None: ...
    def cancel(self) -> None: ...
    def recv_response(self) -> Optional[Response]: ...
    def __iter__(self) -> Iterator[Response]: ...
    def __next__(self) -> Response: ...
    def __enter__(self) -> SyncSession: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...

class AsyncSession:
    """Asynchronous session for communicating with the rkwhisperd daemon."""
    @staticmethod
    async def connect(socket_path: str, hello: ClientHello) -> AsyncSession: ...
    async def send_audio(self, pcm: Union[bytes, bytearray, memoryview]) -> None: ...
    async def finish(self) -> None: ...
    async def cancel(self) -> None: ...
    async def recv_response(self) -> Response: ...
    def __aiter__(self) -> AsyncIterator[Response]: ...
    async def __anext__(self) -> Response: ...
    async def __aenter__(self) -> AsyncSession: ...
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...
