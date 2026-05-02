use anyhow::{Result, bail};
use rkwhisper_protocol::{
    decode_response, encode_client_hello, ClientHello, Response, SharedAudioRing,
    SIGNAL_CANCEL, SIGNAL_DATA_READY, SIGNAL_END_OF_STREAM,
};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;

pub mod sync {
    use super::*;

    pub struct Session {
        stream: UnixStream,
        ring: SharedAudioRing,
    }

    impl Session {
        pub fn connect(socket_path: impl AsRef<Path>, hello: ClientHello) -> Result<Self> {
            let mut stream = UnixStream::connect(socket_path)?;

            // 1. Send ClientHello
            let encoded_hello = encode_client_hello(&hello);
            rkwhisper_protocol::write_frame(&mut stream, &encoded_hello)?;

            // 2. Receive FD
            let fd = SharedAudioRing::recv_fd(&stream)?;

            // 3. Receive ServerHello
            let frame = rkwhisper_protocol::read_frame(&mut stream)?;
            let response = decode_response(&frame)?;
            let server_hello = match response {
                Response::ServerHello(sh) => sh,
                Response::Error { error } => bail!("server error during handshake: {error}"),
                other => bail!("unexpected response during handshake: {other:?}"),
            };

            // 4. Attach to ring
            let ring = SharedAudioRing::attach(fd, server_hello.ring_capacity_bytes as usize)?;

            Ok(Self { stream, ring })
        }

        pub fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            let mut pos = 0;
            while pos < pcm.len() {
                let n = self.ring.push_available(&pcm[pos..])?;
                if n > 0 {
                    pos += n;
                    self.stream.write_all(&[SIGNAL_DATA_READY])?;
                } else {
                    // Ring is full, wait a bit or backoff
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            Ok(())
        }

        pub fn finish(&mut self) -> Result<()> {
            self.stream.write_all(&[SIGNAL_END_OF_STREAM])?;
            Ok(())
        }

        pub fn cancel(&mut self) -> Result<()> {
            self.stream.write_all(&[SIGNAL_CANCEL])?;
            Ok(())
        }

        pub fn recv_response(&mut self) -> Result<Response> {
            let frame = rkwhisper_protocol::read_frame(&mut self.stream)?;
            decode_response(&frame)
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

    impl Session {
        pub async fn connect(socket_path: impl AsRef<Path>, hello: ClientHello) -> Result<Self> {
            let mut stream = UnixStream::connect(socket_path).await?;

            // 1. Send ClientHello
            let encoded_hello = encode_client_hello(&hello);
            rkwhisper_protocol::write_frame_async(&mut stream, &encoded_hello).await?;

            // 2. Receive FD
            let (fd, mut stream) = recv_fd_async(stream).await?;

            // 3. Receive ServerHello
            let frame = rkwhisper_protocol::read_frame_async(&mut stream).await?;
            let response = decode_response(&frame)?;
            let server_hello = match response {
                Response::ServerHello(sh) => sh,
                Response::Error { error } => bail!("server error during handshake: {error}"),
                other => bail!("unexpected response during handshake: {other:?}"),
            };

            // 4. Attach to ring
            let ring = SharedAudioRing::attach(fd, server_hello.ring_capacity_bytes as usize)?;

            Ok(Self { stream, ring })
        }

        pub async fn send_audio(&mut self, pcm: &[u8]) -> Result<()> {
            let mut pos = 0;
            while pos < pcm.len() {
                let n = self.ring.push_available(&pcm[pos..])?;
                if n > 0 {
                    pos += n;
                    self.stream.write_all(&[SIGNAL_DATA_READY]).await?;
                } else {
                    // Ring is full, wait a bit
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
            Ok(())
        }

        pub async fn finish(&mut self) -> Result<()> {
            self.stream.write_all(&[SIGNAL_END_OF_STREAM]).await?;
            Ok(())
        }

        pub async fn cancel(&mut self) -> Result<()> {
            self.stream.write_all(&[SIGNAL_CANCEL]).await?;
            Ok(())
        }

        pub async fn recv_response(&mut self) -> Result<Response> {
            let frame = rkwhisper_protocol::read_frame_async(&mut self.stream).await?;
            decode_response(&frame)
        }
    }

    async fn recv_fd_async(stream: UnixStream) -> Result<(std::os::fd::OwnedFd, UnixStream)> {
        let std_stream = stream.into_std()?;
        std_stream.set_nonblocking(false)?;
        let fd = SharedAudioRing::recv_fd(&std_stream)?;
        std_stream.set_nonblocking(true)?;
        let stream = UnixStream::from_std(std_stream)?;
        Ok((fd, stream))
    }
}
