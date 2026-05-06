pub use rkwhisper_protocol::{AudioFormat, ClientHello, Response, VadOptions};
use rkwhisper_protocol::{
    SIGNAL_CANCEL, SIGNAL_DATA_READY, SIGNAL_END_OF_STREAM, SharedAudioRing, decode_response,
};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("connection failed: {0}")]
    Connection(#[from] std::io::Error),
    #[error("protocol error: {0}")]
    Protocol(#[from] rkwhisper_protocol::Error),
    #[error("handshake failed: {0}")]
    Handshake(String),
    #[error("server error: {0}")]
    Daemon(String),
    #[error("session cancelled")]
    Cancelled,
    #[error("internal error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TranscriptSegment {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
}

pub fn samples_to_pcm(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let s = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    pcm
}

pub mod sync {
    use super::*;

    pub struct Session {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    pub struct AudioSender {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    pub struct ResponseReceiver {
        stream: UnixStream,
    }

    impl Session {
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

        pub fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            AudioSender::send_audio_internal(&mut self.stream, &self.ring, pcm)
        }

        pub fn finish(&mut self) -> Result<()> {
            AudioSender::finish_internal(&mut self.stream)
        }

        pub fn cancel(&mut self) -> Result<()> {
            AudioSender::cancel_internal(&mut self.stream)
        }

        pub fn recv_response(&mut self) -> Result<Response> {
            ResponseReceiver::recv_response_internal(&mut self.stream)
        }

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

        pub fn finish(&mut self) -> Result<()> {
            Self::finish_internal(&mut self.stream)
        }

        fn finish_internal(stream: &mut UnixStream) -> Result<()> {
            stream
                .write_all(&[SIGNAL_END_OF_STREAM])
                .map_err(Error::Connection)?;
            Ok(())
        }

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

#[cfg(feature = "async")]
pub mod asynchronous {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::UnixStream;

    pub struct Session {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    pub struct AudioSender {
        stream: tokio::net::unix::OwnedWriteHalf,
        ring: SharedAudioRing,
    }

    pub struct ResponseReceiver {
        stream: tokio::net::unix::OwnedReadHalf,
    }

    impl Session {
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

        pub async fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            AudioSender::send_audio_internal(&mut self.stream, &self.ring, pcm).await
        }

        pub async fn finish(&mut self) -> Result<()> {
            AudioSender::finish_internal(&mut self.stream).await
        }

        pub async fn cancel(&mut self) -> Result<()> {
            AudioSender::cancel_internal(&mut self.stream).await
        }

        pub async fn recv_response(&mut self) -> Result<Response> {
            ResponseReceiver::recv_response_internal(&mut self.stream).await
        }

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
