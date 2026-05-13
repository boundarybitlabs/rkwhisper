//! Rust client for `rkwhisperd`.
//!
//! The client connects to a running daemon over a Unix socket, sends 16 kHz mono
//! signed 16-bit PCM audio through the shared-memory ring established during the
//! handshake, and receives [`Response`] values as transcription progresses.
//!
//! Use [`sync`] for blocking applications and [`asynchronous`] for Tokio-based
//! applications. Both APIs expose a [`split`](sync::Session::split) pattern so
//! audio sending and response receiving can happen independently.

/// Common protocol types used to configure sessions and inspect responses.
pub use rkwhisper_protocol::{AudioFormat, ClientHello, Response, VadOptions};
use rkwhisper_protocol::{
    SIGNAL_CANCEL, SIGNAL_DATA_READY, SIGNAL_END_OF_STREAM, SharedAudioRing, decode_response,
};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;

/// Errors returned by the rkwhisper client.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Opening, cloning, reading, or writing the Unix socket failed.
    #[error("connection failed: {0}")]
    Connection(#[from] std::io::Error),
    /// Encoding, decoding, framing, or shared-memory ring handling failed.
    #[error("protocol error: {0}")]
    Protocol(#[from] rkwhisper_protocol::Error),
    /// The daemon rejected or did not complete the initial session handshake.
    #[error("handshake failed: {0}")]
    Handshake(String),
    /// The daemon returned an application-level error response.
    #[error("server error: {0}")]
    Daemon(String),
    /// The daemon acknowledged cancellation for this session.
    #[error("session cancelled")]
    Cancelled,
    /// An internal client-side failure that does not fit a narrower category.
    #[error("internal error: {0}")]
    Other(String),
}

/// Client result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Complete transcription returned by `transcribe_all`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Transcription {
    /// All segment text concatenated with spaces and trimmed.
    pub text: String,
    /// Individual transcript segments in daemon response order.
    pub segments: Vec<TranscriptSegment>,
}

/// One transcript segment with timestamps in seconds.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TranscriptSegment {
    /// Segment start time in seconds.
    pub start_sec: f32,
    /// Segment end time in seconds.
    pub end_sec: f32,
    /// Transcribed text for the segment.
    pub text: String,
}

/// Convert normalized `f32` samples to little-endian signed 16-bit PCM bytes.
///
/// Input samples are clamped to `[-1.0, 1.0]` before conversion. The daemon
/// expects 16 kHz mono PCM when using the default [`AudioFormat`].
pub fn samples_to_pcm(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let s = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    pcm
}

/// Blocking client API for `rkwhisperd`.
pub mod sync {
    use super::*;

    /// Blocking transcription session.
    ///
    /// A session owns the Unix socket and shared audio ring for one daemon
    /// request. Use it directly for simple request/response flows, or call
    /// [`split`](Self::split) to send audio and receive responses from separate
    /// threads.
    pub struct Session {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    /// Blocking audio-sending half returned by [`Session::split`].
    pub struct AudioSender {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    /// Blocking response-receiving half returned by [`Session::split`].
    pub struct ResponseReceiver {
        stream: UnixStream,
    }

    impl Session {
        /// Connect to `rkwhisperd` and complete the initial handshake.
        ///
        /// The daemon must be listening on `socket_path`, and `hello.model`
        /// must name an enabled model. If the daemon asks the client to back
        /// off during the handshake, this method retries a small fixed number
        /// of times before returning [`Error::Handshake`].
        pub fn connect(socket_path: impl AsRef<Path>, hello: ClientHello) -> Result<Self> {
            let mut retries = 0;
            let max_retries = 5;

            loop {
                let mut stream =
                    UnixStream::connect(socket_path.as_ref()).map_err(Error::Connection)?;

                // 1. Send ClientHello
                let encoded_hello = rkwhisper_protocol::encode_client_hello(&hello);
                rkwhisper_protocol::write_frame(&mut stream, &encoded_hello)?;

                // 2. Receive Response and potential FD
                let (response, fd) = rkwhisper_protocol::recv_response_with_fd(&stream)?;
                match response {
                    Response::ServerHello(sh) => {
                        let fd = fd.ok_or_else(|| {
                            Error::Handshake("no file descriptor received".to_string())
                        })?;
                        let ring = SharedAudioRing::attach(fd, sh.ring_capacity_bytes as usize)?;
                        return Ok(Self { stream, ring });
                    }
                    Response::BackOff {
                        reason,
                        retry_after_ms,
                    } => {
                        if retries >= max_retries {
                            return Err(Error::Handshake(format!(
                                "too many retries after backoff: {reason}"
                            )));
                        }
                        retries += 1;
                        std::thread::sleep(std::time::Duration::from_millis(retry_after_ms as u64));
                        continue;
                    }
                    Response::Error { error } => return Err(Error::Handshake(error)),
                    other => {
                        return Err(Error::Handshake(format!("unexpected response: {other:?}")));
                    }
                }
            }
        }

        /// Split the session into independent audio sender and response receiver halves.
        ///
        /// This is the preferred shape for live or streaming transcription,
        /// because one thread can keep feeding audio while another thread waits
        /// for daemon responses.
        pub fn split(self) -> Result<(AudioSender, ResponseReceiver)> {
            let write_stream = self.stream.try_clone().map_err(Error::Connection)?;
            let read_stream = self.stream;
            Ok((
                AudioSender {
                    stream: write_stream,
                    ring: self.ring,
                },
                ResponseReceiver {
                    stream: read_stream,
                },
            ))
        }

        /// Send PCM bytes to the daemon.
        ///
        /// `pcm` must match the session [`AudioFormat`]. With the default
        /// format this means 16 kHz mono signed 16-bit little-endian PCM.
        pub fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            AudioSender::send_audio_internal(&mut self.stream, &self.ring, pcm)
        }

        /// Tell the daemon that no more audio will be sent.
        pub fn finish(&mut self) -> Result<()> {
            AudioSender::finish_internal(&mut self.stream)
        }

        /// Request cancellation of the current transcription session.
        pub fn cancel(&mut self) -> Result<()> {
            AudioSender::cancel_internal(&mut self.stream)
        }

        /// Receive the next daemon response.
        ///
        /// Returns [`Error::Daemon`] for daemon error responses and
        /// [`Error::Cancelled`] for cancellation acknowledgements.
        pub fn recv_response(&mut self) -> Result<Response> {
            ResponseReceiver::recv_response_internal(&mut self.stream)
        }

        /// Send all samples, finish the stream, and collect transcript segments.
        ///
        /// This convenience method is intended for one-shot transcription. For
        /// live audio or long-running streams, use [`split`](Self::split).
        pub fn transcribe_all(&mut self, samples: &[f32]) -> Result<Transcription> {
            let pcm = samples_to_pcm(samples);
            self.send_audio(&pcm)?;
            self.finish()?;

            let mut text = String::new();
            let mut segments = Vec::new();

            loop {
                match self.recv_response()? {
                    Response::Segment {
                        text: t,
                        begin,
                        end,
                    } => {
                        text.push_str(&t);
                        text.push(' ');
                        segments.push(TranscriptSegment {
                            start_sec: begin,
                            end_sec: end,
                            text: t,
                        });
                    }
                    Response::Done { .. } => break,
                    _ => {}
                }
            }

            Ok(Transcription {
                text: text.trim().to_string(),
                segments,
            })
        }
    }

    impl AudioSender {
        /// Send PCM bytes to the daemon.
        ///
        /// This method may block while waiting for room in the shared audio
        /// ring. Call [`finish`](Self::finish) after the final chunk.
        pub fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            Self::send_audio_internal(&mut self.stream, &self.ring, pcm)
        }

        fn send_audio_internal(
            stream: &mut UnixStream,
            ring: &SharedAudioRing,
            pcm: &[u8],
        ) -> Result<()> {
            let mut pos = 0;
            while pos < pcm.len() {
                let n = ring.push_available(&pcm[pos..])?;
                if n > 0 {
                    pos += n;
                    stream
                        .write_all(&[SIGNAL_DATA_READY])
                        .map_err(Error::Connection)?;
                } else {
                    // Ring is full, wait a bit or backoff
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            Ok(())
        }

        /// Tell the daemon that no more audio will be sent.
        pub fn finish(&mut self) -> Result<()> {
            Self::finish_internal(&mut self.stream)
        }

        fn finish_internal(stream: &mut UnixStream) -> Result<()> {
            stream
                .write_all(&[SIGNAL_END_OF_STREAM])
                .map_err(Error::Connection)?;
            Ok(())
        }

        /// Request cancellation of the current transcription session.
        pub fn cancel(&mut self) -> Result<()> {
            Self::cancel_internal(&mut self.stream)
        }

        fn cancel_internal(stream: &mut UnixStream) -> Result<()> {
            stream
                .write_all(&[SIGNAL_CANCEL])
                .map_err(Error::Connection)?;
            Ok(())
        }
    }

    impl ResponseReceiver {
        /// Receive the next daemon response.
        ///
        /// Segment, speech-boundary, and completion responses are returned as
        /// [`Response`] values. Daemon error and cancellation responses are
        /// converted to [`Error`] variants.
        pub fn recv_response(&mut self) -> Result<Response> {
            Self::recv_response_internal(&mut self.stream)
        }

        fn recv_response_internal(stream: &mut UnixStream) -> Result<Response> {
            let frame = rkwhisper_protocol::read_frame(stream)?;
            let response = decode_response(&frame)?;
            match response {
                Response::Error { error } => Err(Error::Daemon(error)),
                Response::Cancelled { .. } => Err(Error::Cancelled),
                other => Ok(other),
            }
        }
    }
}

/// Tokio-based client API for `rkwhisperd`.
#[cfg(feature = "async")]
pub mod asynchronous {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::UnixStream;

    /// Asynchronous transcription session.
    ///
    /// A session owns the Unix socket and shared audio ring for one daemon
    /// request. Use it directly for simple async request/response flows, or call
    /// [`split`](Self::split) so separate Tokio tasks can send audio and
    /// receive responses concurrently.
    pub struct Session {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    /// Asynchronous audio-sending half returned by [`Session::split`].
    pub struct AudioSender {
        stream: tokio::net::unix::OwnedWriteHalf,
        ring: SharedAudioRing,
    }

    /// Asynchronous response-receiving half returned by [`Session::split`].
    pub struct ResponseReceiver {
        stream: tokio::net::unix::OwnedReadHalf,
    }

    impl Session {
        /// Connect to `rkwhisperd` and complete the initial handshake.
        ///
        /// The daemon must be listening on `socket_path`, and `hello.model`
        /// must name an enabled model. If the daemon asks the client to back
        /// off during the handshake, this method retries a small fixed number
        /// of times before returning [`Error::Handshake`].
        pub async fn connect(socket_path: impl AsRef<Path>, hello: ClientHello) -> Result<Self> {
            let mut retries = 0;
            let max_retries = 5;

            loop {
                let mut stream = UnixStream::connect(socket_path.as_ref())
                    .await
                    .map_err(Error::Connection)?;

                // 1. Send ClientHello
                let encoded_hello = rkwhisper_protocol::encode_client_hello(&hello);
                rkwhisper_protocol::write_frame_async(&mut stream, &encoded_hello).await?;

                // 2. Receive Response and potential FD (switch to std for recvmsg)
                let (response, fd, stream_back) = tokio::task::spawn_blocking(move || {
                    let std_stream = stream.into_std().map_err(Error::Connection)?;
                    std_stream
                        .set_nonblocking(false)
                        .map_err(Error::Connection)?;
                    let (response, fd) = rkwhisper_protocol::recv_response_with_fd(&std_stream)?;
                    std_stream
                        .set_nonblocking(true)
                        .map_err(Error::Connection)?;
                    let stream = UnixStream::from_std(std_stream).map_err(Error::Connection)?;
                    Ok::<(Response, Option<std::os::fd::OwnedFd>, UnixStream), Error>((
                        response, fd, stream,
                    ))
                })
                .await
                .map_err(|e| Error::Other(e.to_string()))??;

                stream = stream_back;

                match response {
                    Response::ServerHello(sh) => {
                        let fd = fd.ok_or_else(|| {
                            Error::Handshake("no file descriptor received".to_string())
                        })?;
                        let ring = SharedAudioRing::attach(fd, sh.ring_capacity_bytes as usize)?;
                        return Ok(Self { stream, ring });
                    }
                    Response::BackOff {
                        reason,
                        retry_after_ms,
                    } => {
                        if retries >= max_retries {
                            return Err(Error::Handshake(format!(
                                "too many retries after backoff: {reason}"
                            )));
                        }
                        retries += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(retry_after_ms as u64))
                            .await;
                        continue;
                    }
                    Response::Error { error } => return Err(Error::Handshake(error)),
                    other => {
                        return Err(Error::Handshake(format!("unexpected response: {other:?}")));
                    }
                }
            }
        }

        /// Split the session into independent audio sender and response receiver halves.
        ///
        /// This is the preferred shape for live or streaming transcription,
        /// because one task can keep feeding audio while another task waits for
        /// daemon responses.
        pub fn split(self) -> (AudioSender, ResponseReceiver) {
            let (read, write) = self.stream.into_split();
            (
                AudioSender {
                    stream: write,
                    ring: self.ring,
                },
                ResponseReceiver { stream: read },
            )
        }

        /// Send PCM bytes to the daemon.
        ///
        /// `pcm` must match the session [`AudioFormat`]. With the default
        /// format this means 16 kHz mono signed 16-bit little-endian PCM.
        pub async fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            AudioSender::send_audio_internal(&mut self.stream, &self.ring, pcm).await
        }

        /// Tell the daemon that no more audio will be sent.
        pub async fn finish(&mut self) -> Result<()> {
            AudioSender::finish_internal(&mut self.stream).await
        }

        /// Request cancellation of the current transcription session.
        pub async fn cancel(&mut self) -> Result<()> {
            AudioSender::cancel_internal(&mut self.stream).await
        }

        /// Receive the next daemon response.
        ///
        /// Returns [`Error::Daemon`] for daemon error responses and
        /// [`Error::Cancelled`] for cancellation acknowledgements.
        pub async fn recv_response(&mut self) -> Result<Response> {
            ResponseReceiver::recv_response_internal(&mut self.stream).await
        }

        /// Send all samples, finish the stream, and collect transcript segments.
        ///
        /// This convenience method is intended for one-shot transcription. For
        /// live audio or long-running streams, use [`split`](Self::split).
        pub async fn transcribe_all(&mut self, samples: &[f32]) -> Result<Transcription> {
            let pcm = samples_to_pcm(samples);
            self.send_audio(&pcm).await?;
            self.finish().await?;

            let mut text = String::new();
            let mut segments = Vec::new();

            loop {
                match self.recv_response().await? {
                    Response::Segment {
                        text: t,
                        begin,
                        end,
                    } => {
                        text.push_str(&t);
                        text.push(' ');
                        segments.push(TranscriptSegment {
                            start_sec: begin,
                            end_sec: end,
                            text: t,
                        });
                    }
                    Response::Done { .. } => break,
                    _ => {}
                }
            }

            Ok(Transcription {
                text: text.trim().to_string(),
                segments,
            })
        }
    }

    impl AudioSender {
        /// Send PCM bytes to the daemon.
        ///
        /// This method waits asynchronously for room in the shared audio ring
        /// when the daemon has not yet consumed earlier audio. Call
        /// [`finish`](Self::finish) after the final chunk.
        pub async fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            Self::send_audio_internal(&mut self.stream, &self.ring, pcm).await
        }

        async fn send_audio_internal(
            stream: &mut (impl tokio::io::AsyncWrite + Unpin),
            ring: &SharedAudioRing,
            pcm: &[u8],
        ) -> Result<()> {
            let mut pos = 0;
            while pos < pcm.len() {
                let n = ring.push_available(&pcm[pos..])?;
                if n > 0 {
                    pos += n;
                    stream
                        .write_all(&[SIGNAL_DATA_READY])
                        .await
                        .map_err(Error::Connection)?;
                } else {
                    // Ring is full, wait a bit
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
            Ok(())
        }

        /// Tell the daemon that no more audio will be sent.
        pub async fn finish(&mut self) -> Result<()> {
            Self::finish_internal(&mut self.stream).await
        }

        async fn finish_internal(stream: &mut (impl tokio::io::AsyncWrite + Unpin)) -> Result<()> {
            stream
                .write_all(&[SIGNAL_END_OF_STREAM])
                .await
                .map_err(Error::Connection)?;
            Ok(())
        }

        /// Request cancellation of the current transcription session.
        pub async fn cancel(&mut self) -> Result<()> {
            Self::cancel_internal(&mut self.stream).await
        }

        async fn cancel_internal(stream: &mut (impl tokio::io::AsyncWrite + Unpin)) -> Result<()> {
            stream
                .write_all(&[SIGNAL_CANCEL])
                .await
                .map_err(Error::Connection)?;
            Ok(())
        }
    }

    impl ResponseReceiver {
        /// Receive the next daemon response.
        ///
        /// Segment, speech-boundary, and completion responses are returned as
        /// [`Response`] values. Daemon error and cancellation responses are
        /// converted to [`Error`] variants.
        pub async fn recv_response(&mut self) -> Result<Response> {
            Self::recv_response_internal(&mut self.stream).await
        }

        async fn recv_response_internal(
            stream: &mut (impl tokio::io::AsyncRead + Unpin),
        ) -> Result<Response> {
            let frame = rkwhisper_protocol::read_frame_async(stream).await?;
            let response = decode_response(&frame)?;
            match response {
                Response::Error { error } => Err(Error::Daemon(error)),
                Response::Cancelled { .. } => Err(Error::Cancelled),
                other => Ok(other),
            }
        }
    }
}
