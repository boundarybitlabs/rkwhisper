pub const SAMPLE_RATE: u32 = 16_000;
use anyhow::{Context, Result, anyhow, bail};
use std::io::{IoSlice, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
use rustix::fs::{MemfdFlags, ftruncate, memfd_create};
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, SendAncillaryBuffer, SendAncillaryMessage, SendFlags,
    recvmsg, sendmsg,
};

pub const FRAME_MAX_BYTES: usize = 1024 * 1024;
pub const RING_HEADER_BYTES: usize = 32;
pub const RING_DATA_BYTES: usize = SAMPLE_RATE as usize * 2 * 120;
pub const RING_MAGIC: u32 = 0x5257_4853; // RWHS
pub const RING_VERSION: u32 = 1;

pub const SIGNAL_DATA_READY: u8 = 0x01;
pub const SIGNAL_END_OF_STREAM: u8 = 0x02;
pub const SIGNAL_CANCEL: u8 = 0x03;

const SAMPLE_FORMAT_S16LE: i32 = 1;

#[derive(Clone, Debug, Default)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_format: i32,
}

#[derive(Clone, Debug, Default)]
pub struct VadOptions {
    pub threshold: Option<f32>,
    pub min_speech_ms: Option<u32>,
    pub min_silence_ms: Option<u32>,
    pub speech_pad_ms: Option<u32>,
    pub window_samples: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct ClientHello {
    pub model: String,
    pub mode: String,
    pub lang: String,
    pub task: String,
    pub max_new_tokens: usize,
    pub beam_size: usize,
    pub notimestamps: bool,
    pub suppress_tokens: String,
    pub audio_format: AudioFormat,
    pub vad: VadOptions,
}

impl Default for ClientHello {
    fn default() -> Self {
        Self {
            model: String::new(),
            mode: "batch".to_string(),
            lang: "en".to_string(),
            task: "transcribe".to_string(),
            max_new_tokens: 128,
            beam_size: 5,
            notimestamps: false,
            suppress_tokens: "default".to_string(),
            audio_format: supported_audio_format(),
            vad: VadOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServerHello {
    pub audio_format: AudioFormat,
    pub ring_capacity_bytes: u64,
    pub ring_header_bytes: u32,
}

#[derive(Clone, Debug)]
pub enum Response {
    ServerHello(ServerHello),
    Segment {
        text: String,
        begin: f32,
        end: f32,
    },
    Done {
        audio_s: f32,
        rtf: f32,
    },
    Error {
        error: String,
    },
    Cancelled {
        audio_s: f32,
        rtf: f32,
        windows_dispatched: u64,
        windows_completed: u64,
    },
    BackOff {
        reason: String,
        retry_after_ms: u32,
    },
}

pub fn supported_audio_format() -> AudioFormat {
    AudioFormat {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        sample_format: SAMPLE_FORMAT_S16LE,
    }
}

pub fn validate_client_hello(hello: &ClientHello) -> Result<()> {
    if hello.mode != "batch" && hello.mode != "stream" {
        bail!("unsupported mode {:?}", hello.mode);
    }
    if hello.beam_size == 0 {
        bail!("beam_size must be at least 1");
    }
    if hello.audio_format.sample_rate != SAMPLE_RATE
        || hello.audio_format.channels != 1
        || hello.audio_format.sample_format != SAMPLE_FORMAT_S16LE
    {
        bail!("unsupported audio format; expected 16 kHz mono s16le");
    }
    Ok(())
}

pub fn read_client_hello(reader: &mut impl Read) -> Result<ClientHello> {
    let frame = read_frame(reader)?;
    decode_client_hello(&frame)
}

pub fn write_response(writer: &mut impl Write, response: Response) -> Result<()> {
    write_frame(writer, &encode_response(response))
}

pub fn read_frame(reader: &mut impl Read) -> Result<Vec<u8>> {
    let mut len = [0u8; 4];
    reader
        .read_exact(&mut len)
        .context("failed to read protobuf frame length")?;
    let len = u32::from_le_bytes(len) as usize;
    if len > FRAME_MAX_BYTES {
        bail!("protobuf frame exceeds {FRAME_MAX_BYTES} bytes");
    }
    let mut frame = vec![0u8; len];
    reader
        .read_exact(&mut frame)
        .context("failed to read protobuf frame body")?;
    Ok(frame)
}

pub fn write_frame(writer: &mut impl Write, frame: &[u8]) -> Result<()> {
    if frame.len() > FRAME_MAX_BYTES {
        bail!("protobuf frame exceeds {FRAME_MAX_BYTES} bytes");
    }
    writer.write_all(&(frame.len() as u32).to_le_bytes())?;
    writer.write_all(frame)?;
    writer.flush()?;
    Ok(())
}

pub fn send_response_with_fd(
    stream: &UnixStream,
    response: Response,
    fd: Option<&OwnedFd>,
) -> Result<()> {
    let frame = encode_response(response);
    if frame.len() > FRAME_MAX_BYTES {
        bail!("protobuf frame exceeds {FRAME_MAX_BYTES} bytes");
    }

    let len_bytes = (frame.len() as u32).to_le_bytes();
    let iov = [IoSlice::new(&len_bytes), IoSlice::new(&frame)];

    if let Some(fd) = fd {
        let fds = [fd.as_fd()];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        control.push(SendAncillaryMessage::ScmRights(&fds));
        sendmsg(stream, &iov, &mut control, SendFlags::empty())
            .context("failed to send response with fd")?;
    } else {
        let mut control = SendAncillaryBuffer::new(&mut []);
        sendmsg(stream, &iov, &mut control, SendFlags::empty())
            .context("failed to send response without fd")?;
    }

    Ok(())
}

pub fn recv_response_with_fd(stream: &UnixStream) -> Result<(Response, Option<OwnedFd>)> {
    let mut len_bytes = [0u8; 4];
    let mut iov = [rustix::io::IoSliceMut::new(&mut len_bytes)];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut control = RecvAncillaryBuffer::new(&mut space);

    let msg = recvmsg(stream, &mut iov, &mut control, rustix::net::RecvFlags::empty())
        .context("failed to receive response header and ancillary data")?;

    if msg.bytes != 4 {
        bail!("failed to receive full response frame length");
    }

    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > FRAME_MAX_BYTES {
        bail!("protobuf frame exceeds {FRAME_MAX_BYTES} bytes");
    }

    let mut body = vec![0u8; len];
    let mut body_reader = stream;
    body_reader
        .read_exact(&mut body)
        .context("failed to read response frame body")?;

    let response = decode_response(&body)?;
    let mut received_fd = None;

    for cmsg in control.drain() {
        if let RecvAncillaryMessage::ScmRights(fds) = cmsg {
            received_fd = fds.map(|fd| fd.try_clone()).next().transpose()?;
        }
    }

    Ok((response, received_fd))
}

#[cfg(feature = "async")]
pub async fn read_frame_async(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
) -> Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;
    let mut len = [0u8; 4];
    reader
        .read_exact(&mut len)
        .await
        .context("failed to read protobuf frame length")?;
    let len = u32::from_le_bytes(len) as usize;
    if len > FRAME_MAX_BYTES {
        bail!("protobuf frame exceeds {FRAME_MAX_BYTES} bytes");
    }
    let mut frame = vec![0u8; len];
    reader
        .read_exact(&mut frame)
        .await
        .context("failed to read protobuf frame body")?;
    Ok(frame)
}

#[cfg(feature = "async")]
pub async fn write_frame_async(
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    frame: &[u8],
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    if frame.len() > FRAME_MAX_BYTES {
        bail!("protobuf frame exceeds {FRAME_MAX_BYTES} bytes");
    }
    writer.write_all(&(frame.len() as u32).to_le_bytes()).await?;
    writer.write_all(frame).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
pub struct SharedAudioRing {
    fd: OwnedFd,
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
unsafe impl Send for SharedAudioRing {}
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
unsafe impl Sync for SharedAudioRing {}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
impl SharedAudioRing {
    pub fn create(capacity: usize) -> Result<Self> {
        let len = RING_HEADER_BYTES
            .checked_add(capacity)
            .ok_or_else(|| anyhow!("ring size overflow"))?;
        let fd = memfd_create("rkwhisper-audio", MemfdFlags::CLOEXEC)
            .context("failed to create audio memfd")?;
        ftruncate(&fd, len as u64).context("failed to size audio memfd")?;
        let ring = Self::attach(fd, capacity)?;
        ring.init_header();
        Ok(ring)
    }

    pub fn attach(fd: OwnedFd, capacity: usize) -> Result<Self> {
        let len = RING_HEADER_BYTES
            .checked_add(capacity)
            .ok_or_else(|| anyhow!("ring size overflow"))?;
        let ptr = unsafe {
            mmap(
                null_mut(),
                len,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )
        }
        .context("failed to map audio memfd")? as *mut u8;

        Ok(Self {
            fd,
            ptr,
            len,
            capacity,
        })
    }

    pub fn init_header(&self) {
        self.write_u32(0, RING_MAGIC);
        self.write_u32(4, RING_VERSION);
        self.capacity_atomic()
            .store(self.capacity as u64, Ordering::Release);
        self.write_atomic().store(0, Ordering::Release);
        self.read_atomic().store(0, Ordering::Release);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn fd(&self) -> &OwnedFd {
        &self.fd
    }

    pub fn send_fd(&self, stream: &UnixStream) -> Result<()> {
        let byte = [0u8];
        let iov = [IoSlice::new(&byte)];
        let fds = [self.fd.as_fd()];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        if !control.push(SendAncillaryMessage::ScmRights(&fds)) {
            bail!("failed to prepare memfd ancillary data");
        }
        let sent = sendmsg(stream, &iov, &mut control, SendFlags::empty())
            .context("failed to send audio memfd")?;
        if sent != 1 {
            bail!("failed to send audio memfd marker byte");
        }
        Ok(())
    }

    pub fn recv_fd(stream: &UnixStream) -> Result<OwnedFd> {

        let mut byte = [0u8; 1];
        let mut iov = [rustix::io::IoSliceMut::new(&mut byte)];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = RecvAncillaryBuffer::new(&mut space);

        let msg = recvmsg(stream, &mut iov, &mut control, rustix::net::RecvFlags::empty())
            .context("failed to receive audio memfd")?;

        if msg.bytes != 1 {
            bail!("failed to receive audio memfd marker byte");
        }

        for cmsg in control.drain() {
            if let RecvAncillaryMessage::ScmRights(fds) = cmsg {
                return fds
                    .map(|fd| fd.try_clone())
                    .next()
                    .transpose()?
                    .ok_or_else(|| anyhow!("no file descriptor received"));
            }
        }

        bail!("no file descriptor received in ancillary data");
    }

    pub fn drain_available(&self, out: &mut Vec<u8>) -> Result<()> {
        let read = self.read_offset();
        let write = self.write_offset();
        if write.saturating_sub(read) > self.capacity as u64 {
            bail!("shared-memory ring overrun");
        }
        if write <= read {
            return Ok(());
        }
        let available = (write - read) as usize;
        let start = (read as usize) % self.capacity;
        let first = available.min(self.capacity - start);
        out.extend_from_slice(self.data_slice(start, first));
        if available > first {
            out.extend_from_slice(self.data_slice(0, available - first));
        }
        self.set_read_offset(write);
        Ok(())
    }

    pub fn push_available(&self, data: &[u8]) -> Result<usize> {
        let read = self.read_offset();
        let write = self.write_offset();
        if write.saturating_sub(read) > self.capacity as u64 {
            bail!("shared-memory ring overrun");
        }
        let available = self.capacity - (write - read) as usize;
        let to_write = data.len().min(available);
        if to_write == 0 {
            return Ok(0);
        }

        let start = (write as usize) % self.capacity;
        let first = to_write.min(self.capacity - start);
        self.write_data_slice(start, &data[..first]);
        if to_write > first {
            self.write_data_slice(0, &data[first..to_write]);
        }
        self.set_write_offset(write + to_write as u64);
        Ok(to_write)
    }

    fn write_data_slice(&self, offset: usize, data: &[u8]) {
        unsafe {
            self.ptr
                .add(RING_HEADER_BYTES + offset)
                .copy_from_nonoverlapping(data.as_ptr(), data.len())
        }
    }

    fn read_offset(&self) -> u64 {
        self.read_atomic().load(Ordering::Acquire)
    }

    fn set_read_offset(&self, value: u64) {
        self.read_atomic().store(value, Ordering::Release);
    }

    fn write_offset(&self) -> u64 {
        self.write_atomic().load(Ordering::Acquire)
    }

    fn set_write_offset(&self, value: u64) {
        self.write_atomic().store(value, Ordering::Release);
    }

    fn capacity_atomic(&self) -> &AtomicU64 {
        unsafe { &*(self.ptr.add(8) as *const AtomicU64) }
    }

    fn write_atomic(&self) -> &AtomicU64 {
        unsafe { &*(self.ptr.add(16) as *const AtomicU64) }
    }

    fn read_atomic(&self) -> &AtomicU64 {
        unsafe { &*(self.ptr.add(24) as *const AtomicU64) }
    }

    fn data_slice(&self, start: usize, len: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.add(RING_HEADER_BYTES + start), len) }
    }

    fn write_u32(&self, offset: usize, value: u32) {
        let bytes = value.to_le_bytes();
        unsafe { self.ptr.add(offset).copy_from(bytes.as_ptr(), bytes.len()) }
    }

    #[cfg(test)]
    fn set_test_offsets(&self, read: u64, write: u64) {
        self.read_atomic().store(read, Ordering::Release);
        self.write_atomic().store(write, Ordering::Release);
    }
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
impl Drop for SharedAudioRing {
    fn drop(&mut self) {
        let _ = unsafe { munmap(self.ptr.cast(), self.len) };
    }
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
pub struct SharedAudioRing;

#[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
impl SharedAudioRing {
    pub fn create(_capacity: usize) -> Result<Self> {
        bail!("shared-memory daemon protocol requires Linux or FreeBSD memfd support")
    }

    pub fn capacity(&self) -> usize {
        0
    }

    pub fn send_fd(&self, _stream: &UnixStream) -> Result<()> {
        bail!("shared-memory daemon protocol requires Linux or FreeBSD memfd support")
    }

    pub fn drain_available(&self, _out: &mut Vec<u8>) -> Result<()> {
        Ok(())
    }
}

pub fn decode_client_hello(bytes: &[u8]) -> Result<ClientHello> {
    let mut hello = ClientHello::default();
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 2) => hello.model = input.string()?,
            (2, 2) => hello.mode = input.string()?,
            (3, 2) => hello.lang = input.string()?,
            (4, 2) => hello.task = input.string()?,
            (5, 0) => hello.max_new_tokens = input.varint()? as usize,
            (6, 0) => hello.beam_size = input.varint()? as usize,
            (7, 0) => hello.notimestamps = input.varint()? != 0,
            (8, 2) => hello.suppress_tokens = input.string()?,
            (9, 2) => hello.audio_format = decode_audio_format(input.bytes()?)?,
            (10, 2) => hello.vad = decode_vad_options(input.bytes()?)?,
            (_, _) => input.skip(wire)?,
        }
    }
    if hello.model.is_empty() {
        bail!("client hello missing model");
    }
    Ok(hello)
}

fn decode_audio_format(bytes: &[u8]) -> Result<AudioFormat> {
    let mut format = AudioFormat::default();
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 0) => format.sample_rate = input.varint()? as u32,
            (2, 0) => format.channels = input.varint()? as u32,
            (3, 0) => format.sample_format = input.varint()? as i32,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(format)
}

fn decode_vad_options(bytes: &[u8]) -> Result<VadOptions> {
    let mut vad = VadOptions::default();
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 5) => vad.threshold = Some(input.fixed32()?),
            (2, 0) => vad.min_speech_ms = Some(input.varint()? as u32),
            (3, 0) => vad.min_silence_ms = Some(input.varint()? as u32),
            (4, 0) => vad.speech_pad_ms = Some(input.varint()? as u32),
            (5, 0) => vad.window_samples = Some(input.varint()? as usize),
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(vad)
}

fn encode_response(response: Response) -> Vec<u8> {
    let mut out = Vec::new();
    match response {
        Response::ServerHello(server) => field_message(&mut out, 1, &encode_server_hello(server)),
        Response::Segment { text, begin, end } => {
            let mut msg = Vec::new();
            field_string(&mut msg, 1, &text);
            field_fixed32(&mut msg, 2, begin);
            field_fixed32(&mut msg, 3, end);
            field_message(&mut out, 2, &msg);
        }
        Response::Done { audio_s, rtf } => {
            let mut msg = Vec::new();
            field_fixed32(&mut msg, 1, audio_s);
            field_fixed32(&mut msg, 2, rtf);
            field_message(&mut out, 3, &msg);
        }
        Response::Error { error } => {
            let mut msg = Vec::new();
            field_string(&mut msg, 1, &error);
            field_message(&mut out, 4, &msg);
        }
        Response::Cancelled {
            audio_s,
            rtf,
            windows_dispatched,
            windows_completed,
        } => {
            let mut msg = Vec::new();
            field_fixed32(&mut msg, 1, audio_s);
            field_fixed32(&mut msg, 2, rtf);
            field_varint(&mut msg, 3, windows_dispatched);
            field_varint(&mut msg, 4, windows_completed);
            field_message(&mut out, 5, &msg);
        }
        Response::BackOff {
            reason,
            retry_after_ms,
        } => {
            let mut msg = Vec::new();
            field_string(&mut msg, 1, &reason);
            field_varint(&mut msg, 2, retry_after_ms as u64);
            field_message(&mut out, 6, &msg);
        }
    }
    out
}

fn encode_server_hello(server: ServerHello) -> Vec<u8> {
    let mut out = Vec::new();
    field_message(&mut out, 1, &encode_audio_format(&server.audio_format));
    field_varint(&mut out, 2, server.ring_capacity_bytes);
    field_varint(&mut out, 3, server.ring_header_bytes as u64);
    out
}

fn encode_audio_format(format: &AudioFormat) -> Vec<u8> {
    let mut out = Vec::new();
    field_varint(&mut out, 1, format.sample_rate as u64);
    field_varint(&mut out, 2, format.channels as u64);
    field_varint(&mut out, 3, format.sample_format as u64);
    out
}

pub fn encode_client_hello(hello: &ClientHello) -> Vec<u8> {
    let mut out = Vec::new();
    field_string(&mut out, 1, &hello.model);
    field_string(&mut out, 2, &hello.mode);
    field_string(&mut out, 3, &hello.lang);
    field_string(&mut out, 4, &hello.task);
    field_varint(&mut out, 5, hello.max_new_tokens as u64);
    field_varint(&mut out, 6, hello.beam_size as u64);
    field_varint(&mut out, 7, if hello.notimestamps { 1 } else { 0 });
    field_string(&mut out, 8, &hello.suppress_tokens);
    field_message(&mut out, 9, &encode_audio_format(&hello.audio_format));
    field_message(&mut out, 10, &encode_vad_options(&hello.vad));
    out
}

fn encode_vad_options(vad: &VadOptions) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(threshold) = vad.threshold {
        field_fixed32(&mut out, 1, threshold);
    }
    if let Some(min_speech_ms) = vad.min_speech_ms {
        field_varint(&mut out, 2, min_speech_ms as u64);
    }
    if let Some(min_silence_ms) = vad.min_silence_ms {
        field_varint(&mut out, 3, min_silence_ms as u64);
    }
    if let Some(speech_pad_ms) = vad.speech_pad_ms {
        field_varint(&mut out, 4, speech_pad_ms as u64);
    }
    if let Some(window_samples) = vad.window_samples {
        field_varint(&mut out, 5, window_samples as u64);
    }
    out
}

pub fn decode_response(bytes: &[u8]) -> Result<Response> {
    let mut input = ProtoInput::new(bytes);
    let (field, wire) = input.key()?;
    if wire != 2 {
        bail!("invalid response wire type {wire}");
    }
    let body = input.bytes()?;
    match field {
        1 => Ok(Response::ServerHello(decode_server_hello(body)?)),
        2 => decode_segment(body),
        3 => decode_done(body),
        4 => Ok(Response::Error {
            error: decode_error(body)?,
        }),
        5 => decode_cancelled(body),
        6 => decode_back_off(body),
        _ => bail!("unknown response field {field}"),
    }
}

fn decode_server_hello(bytes: &[u8]) -> Result<ServerHello> {
    let mut hello = ServerHello {
        audio_format: AudioFormat::default(),
        ring_capacity_bytes: 0,
        ring_header_bytes: 0,
    };
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 2) => hello.audio_format = decode_audio_format(input.bytes()?)?,
            (2, 0) => hello.ring_capacity_bytes = input.varint()?,
            (3, 0) => hello.ring_header_bytes = input.varint()? as u32,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(hello)
}

fn decode_segment(bytes: &[u8]) -> Result<Response> {
    let mut text = String::new();
    let mut begin = 0.0;
    let mut end = 0.0;
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 2) => text = input.string()?,
            (2, 5) => begin = input.fixed32()?,
            (3, 5) => end = input.fixed32()?,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(Response::Segment { text, begin, end })
}

fn decode_done(bytes: &[u8]) -> Result<Response> {
    let mut audio_s = 0.0;
    let mut rtf = 0.0;
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 5) => audio_s = input.fixed32()?,
            (2, 5) => rtf = input.fixed32()?,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(Response::Done { audio_s, rtf })
}

fn decode_error(bytes: &[u8]) -> Result<String> {
    let mut input = ProtoInput::new(bytes);
    let mut error = String::new();
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 2) => error = input.string()?,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(error)
}

fn decode_cancelled(bytes: &[u8]) -> Result<Response> {
    let mut audio_s = 0.0;
    let mut rtf = 0.0;
    let mut windows_dispatched = 0;
    let mut windows_completed = 0;
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 5) => audio_s = input.fixed32()?,
            (2, 5) => rtf = input.fixed32()?,
            (3, 0) => windows_dispatched = input.varint()?,
            (4, 0) => windows_completed = input.varint()?,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(Response::Cancelled {
        audio_s,
        rtf,
        windows_dispatched,
        windows_completed,
    })
}

fn decode_back_off(bytes: &[u8]) -> Result<Response> {
    let mut reason = String::new();
    let mut retry_after_ms = 0;
    let mut input = ProtoInput::new(bytes);
    while !input.is_empty() {
        let (field, wire) = input.key()?;
        match (field, wire) {
            (1, 2) => reason = input.string()?,
            (2, 0) => retry_after_ms = input.varint()? as u32,
            (_, _) => input.skip(wire)?,
        }
    }
    Ok(Response::BackOff {
        reason,
        retry_after_ms,
    })
}

fn field_varint(out: &mut Vec<u8>, field: u32, value: u64) {
    varint(out, ((field as u64) << 3) | 0);
    varint(out, value);
}

fn field_fixed32(out: &mut Vec<u8>, field: u32, value: f32) {
    varint(out, ((field as u64) << 3) | 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn field_string(out: &mut Vec<u8>, field: u32, value: &str) {
    field_message(out, field, value.as_bytes());
}

fn field_message(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    varint(out, ((field as u64) << 3) | 2);
    varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

struct ProtoInput<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ProtoInput<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn key(&mut self) -> Result<(u32, u8)> {
        let key = self.varint()?;
        Ok(((key >> 3) as u32, (key & 0x7) as u8))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| anyhow!("truncated protobuf varint"))?;
            self.pos += 1;
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("protobuf varint too long");
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow!("protobuf length overflow"))?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| anyhow!("truncated protobuf length-delimited field"))?;
        self.pos = end;
        Ok(bytes)
    }

    fn string(&mut self) -> Result<String> {
        Ok(std::str::from_utf8(self.bytes()?)
            .context("protobuf string is not valid UTF-8")?
            .to_string())
    }

    fn fixed32(&mut self) -> Result<f32> {
        let end = self.pos + 4;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| anyhow!("truncated protobuf fixed32"))?;
        self.pos = end;
        Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.pos += 4,
            _ => bail!("unsupported protobuf wire type {wire}"),
        }
        if self.pos > self.bytes.len() {
            bail!("truncated protobuf field");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_client_hello() {
        let mut audio = Vec::new();
        field_varint(&mut audio, 1, 16_000);
        field_varint(&mut audio, 2, 1);
        field_varint(&mut audio, 3, SAMPLE_FORMAT_S16LE as u64);

        let mut vad = Vec::new();
        field_fixed32(&mut vad, 1, 0.6);
        field_varint(&mut vad, 2, 300);

        let mut msg = Vec::new();
        field_string(&mut msg, 1, "whisper-small-30s");
        field_string(&mut msg, 2, "stream");
        field_varint(&mut msg, 6, 2);
        field_message(&mut msg, 9, &audio);
        field_message(&mut msg, 10, &vad);

        let hello = decode_client_hello(&msg).unwrap();
        assert_eq!(hello.model, "whisper-small-30s");
        assert_eq!(hello.mode, "stream");
        assert_eq!(hello.beam_size, 2);
        assert_eq!(hello.audio_format.sample_rate, 16_000);
        assert_eq!(hello.vad.min_speech_ms, Some(300));
    }

    #[test]
    fn validates_audio_format() {
        let mut hello = ClientHello {
            model: "whisper-small-30s".to_string(),
            ..ClientHello::default()
        };
        validate_client_hello(&hello).unwrap();
        hello.audio_format.sample_rate = 48_000;
        assert!(validate_client_hello(&hello).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn ring_reports_overrun() {
        let ring = SharedAudioRing::create(8).unwrap();
        ring.set_test_offsets(0, 9);
        let mut out = Vec::new();
        let error = ring.drain_available(&mut out).unwrap_err();
        assert!(error.to_string().contains("shared-memory ring overrun"));
    }
}
