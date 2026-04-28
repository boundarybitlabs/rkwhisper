#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///

import argparse
import array
import mmap
import os
import socket
import struct
import sys
import threading
import time
import wave
from pathlib import Path


DEFAULT_SOCKET = "/run/rkwhisper/asr.sock"
SAMPLE_FORMAT_S16LE = 1
SIGNAL_DATA_READY = b"\x01"
SIGNAL_END_OF_STREAM = b"\x02"
RING_MAGIC = 0x5257_4853
RING_VERSION = 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Client for rkwhisperd")
    parser.add_argument("wav", type=Path, help="mono 16 kHz WAV file")
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="rkwhisperd Unix socket")
    parser.add_argument("--mode", choices=("batch", "stream"), default="batch")
    parser.add_argument("--model", default="whisper-tiny-30s")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--notimestamps", action="store_true")
    parser.add_argument("--suppress-tokens", default="default")
    parser.add_argument("--vad-threshold", type=float)
    parser.add_argument("--vad-min-speech-ms", type=int)
    parser.add_argument("--vad-min-silence-ms", type=int)
    parser.add_argument("--vad-speech-pad-ms", type=int)
    parser.add_argument("--vad-window-samples", type=int)
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=1000,
        help="stream mode PCM chunk duration in milliseconds",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="print final transcript text instead of daemon response dictionaries",
    )
    args = parser.parse_args()

    if args.mode == "batch":
        return run_batch(args)
    return run_stream(args)


def run_batch(args: argparse.Namespace) -> int:
    pcm = read_wav_s16le(args.wav)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(args.socket)
        stream = sock.makefile("rb")
        ring = setup_shared_ring(sock, stream, hello_from_args(args))
        write_ring_stream(sock, ring, [pcm])
        return read_responses(stream, text_only=args.text)


def run_stream(args: argparse.Namespace) -> int:
    transcript_parts: list[str] = []
    done = threading.Event()
    status = {"code": 0}

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(args.socket)
        stream = sock.makefile("rb")
        ring = setup_shared_ring(sock, stream, hello_from_args(args))

        reader = threading.Thread(
            target=read_responses_thread,
            args=(stream, args.text, transcript_parts, done, status),
            daemon=True,
        )
        reader.start()

        try:
            for pcm in iter_wav_s16le_chunks(args.wav, args.frame_ms):
                write_ring_chunk(sock, ring, pcm)
            sock.sendall(SIGNAL_END_OF_STREAM)
        except BrokenPipeError:
            done.set()
            status["code"] = 1

        reader.join()
        if args.text and transcript_parts:
            print("".join(transcript_parts).strip())
        return status["code"]


def hello_from_args(args: argparse.Namespace) -> dict:
    hello = {
        "model": args.model,
        "mode": args.mode,
        "lang": args.lang,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "beam_size": args.beam_size,
        "notimestamps": args.notimestamps,
        "suppress_tokens": args.suppress_tokens,
        "audio_format": {
            "sample_rate": 16_000,
            "channels": 1,
            "sample_format": SAMPLE_FORMAT_S16LE,
        },
    }
    vad = {
        "threshold": args.vad_threshold,
        "min_speech_ms": args.vad_min_speech_ms,
        "min_silence_ms": args.vad_min_silence_ms,
        "speech_pad_ms": args.vad_speech_pad_ms,
        "window_samples": args.vad_window_samples,
    }
    vad = {key: value for key, value in vad.items() if value is not None}
    if vad:
        hello["vad"] = vad
    return hello


def setup_shared_ring(sock: socket.socket, stream, hello: dict) -> dict:
    send_frame(sock, encode_client_hello(hello))
    fd = recv_fd(sock)
    response = read_response(stream)
    if response["type"] == "error":
        raise SystemExit(response["error"])
    if response["type"] != "server_hello":
        raise SystemExit(f"unexpected setup response: {response}")

    total_len = response["ring_header_bytes"] + response["ring_capacity_bytes"]
    mapping = mmap.mmap(fd, total_len)
    os.close(fd)
    magic, version = struct.unpack_from("<II", mapping, 0)
    if magic != RING_MAGIC or version != RING_VERSION:
        raise SystemExit("server returned an invalid shared-memory ring")
    return {
        "mmap": mapping,
        "capacity": response["ring_capacity_bytes"],
        "header_bytes": response["ring_header_bytes"],
    }


def recv_fd(sock: socket.socket) -> int:
    fds = array.array("i")
    marker, ancdata, _flags, _addr = sock.recvmsg(1, socket.CMSG_SPACE(fds.itemsize))
    if not marker:
        raise SystemExit("daemon closed before sending shared-memory fd")
    for level, cmsg_type, data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(data[: fds.itemsize])
            return fds[0]
    raise SystemExit("daemon did not send shared-memory fd")


def write_ring_stream(sock: socket.socket, ring: dict, chunks) -> None:
    for chunk in chunks:
        offset = 0
        while offset < len(chunk):
            step = min(64 * 1024, len(chunk) - offset)
            write_ring_chunk(sock, ring, chunk[offset : offset + step])
            offset += step
    sock.sendall(SIGNAL_END_OF_STREAM)


def write_ring_chunk(sock: socket.socket, ring: dict, pcm: bytes) -> None:
    if not pcm:
        return
    if len(pcm) % 2 != 0:
        raise SystemExit("PCM writes must contain whole s16le samples")

    mapping = ring["mmap"]
    capacity = ring["capacity"]
    header_bytes = ring["header_bytes"]
    written = 0
    while written < len(pcm):
        read = struct.unpack_from("<Q", mapping, 24)[0]
        write = struct.unpack_from("<Q", mapping, 16)[0]
        used = write - read
        if used >= capacity:
            time.sleep(0.005)
            continue

        free = capacity - used
        count = min(free, len(pcm) - written)
        pos = write % capacity
        first = min(count, capacity - pos)
        mapping[header_bytes + pos : header_bytes + pos + first] = pcm[written : written + first]
        if count > first:
            mapping[header_bytes : header_bytes + count - first] = pcm[
                written + first : written + count
            ]
        struct.pack_into("<Q", mapping, 16, write + count)
        written += count
        sock.sendall(SIGNAL_DATA_READY)


def send_frame(sock: socket.socket, frame: bytes) -> None:
    sock.sendall(struct.pack("<I", len(frame)))
    sock.sendall(frame)


def read_responses(stream, text_only: bool) -> int:
    transcript_parts: list[str] = []
    status = {"code": 0}
    read_responses_file(stream, text_only, transcript_parts, status)
    if text_only and transcript_parts:
        print("".join(transcript_parts).strip())
    return status["code"]


def read_responses_thread(
    stream,
    text_only: bool,
    transcript_parts: list[str],
    done: threading.Event,
    status: dict,
) -> None:
    try:
        read_responses_file(stream, text_only, transcript_parts, status)
    finally:
        done.set()


def read_responses_file(stream, text_only: bool, transcript_parts: list[str], status: dict) -> None:
    while True:
        try:
            message = read_response(stream)
        except EOFError:
            status["code"] = 1
            return

        msg_type = message.get("type")
        if msg_type == "segment":
            if text_only:
                transcript_parts.append(message.get("text", ""))
                transcript_parts.append(" ")
            else:
                print(message, flush=True)
        elif msg_type == "done":
            if not text_only:
                print(message, flush=True)
            return
        elif msg_type == "error":
            print(message.get("error", "daemon error"), file=sys.stderr)
            status["code"] = 1
            return
        else:
            print(message, flush=True)


def read_response(stream) -> dict:
    length = stream.read(4)
    if len(length) == 0:
        raise EOFError
    if len(length) != 4:
        raise SystemExit("truncated protobuf response length")
    size = struct.unpack("<I", length)[0]
    body = stream.read(size)
    if len(body) != size:
        raise SystemExit("truncated protobuf response body")
    return decode_response(body)


def encode_client_hello(hello: dict) -> bytes:
    out = bytearray()
    field_string(out, 1, hello["model"])
    field_string(out, 2, hello["mode"])
    field_string(out, 3, hello["lang"])
    field_string(out, 4, hello["task"])
    field_varint(out, 5, hello["max_new_tokens"])
    field_varint(out, 6, hello["beam_size"])
    field_varint(out, 7, 1 if hello["notimestamps"] else 0)
    field_string(out, 8, hello["suppress_tokens"])
    field_message(out, 9, encode_audio_format(hello["audio_format"]))
    if "vad" in hello:
        field_message(out, 10, encode_vad(hello["vad"]))
    return bytes(out)


def encode_audio_format(audio_format: dict) -> bytes:
    out = bytearray()
    field_varint(out, 1, audio_format["sample_rate"])
    field_varint(out, 2, audio_format["channels"])
    field_varint(out, 3, audio_format["sample_format"])
    return bytes(out)


def encode_vad(vad: dict) -> bytes:
    out = bytearray()
    if "threshold" in vad:
        field_fixed32(out, 1, vad["threshold"])
    if "min_speech_ms" in vad:
        field_varint(out, 2, vad["min_speech_ms"])
    if "min_silence_ms" in vad:
        field_varint(out, 3, vad["min_silence_ms"])
    if "speech_pad_ms" in vad:
        field_varint(out, 4, vad["speech_pad_ms"])
    if "window_samples" in vad:
        field_varint(out, 5, vad["window_samples"])
    return bytes(out)


def decode_response(body: bytes) -> dict:
    for field, wire, value in iter_fields(body):
        if field == 1 and wire == 2:
            return decode_server_hello(value)
        if field == 2 and wire == 2:
            return decode_segment(value)
        if field == 3 and wire == 2:
            return decode_done(value)
        if field == 4 and wire == 2:
            return decode_error(value)
    return {"type": "unknown"}


def decode_server_hello(body: bytes) -> dict:
    response = {"type": "server_hello", "ring_capacity_bytes": 0, "ring_header_bytes": 0}
    for field, wire, value in iter_fields(body):
        if field == 2 and wire == 0:
            response["ring_capacity_bytes"] = value
        elif field == 3 and wire == 0:
            response["ring_header_bytes"] = value
    return response


def decode_segment(body: bytes) -> dict:
    response = {"type": "segment", "text": "", "begin": 0.0, "end": 0.0}
    for field, wire, value in iter_fields(body):
        if field == 1 and wire == 2:
            response["text"] = value.decode("utf-8")
        elif field == 2 and wire == 5:
            response["begin"] = value
        elif field == 3 and wire == 5:
            response["end"] = value
    return response


def decode_done(body: bytes) -> dict:
    response = {"type": "done", "audio_s": 0.0, "rtf": 0.0}
    for field, wire, value in iter_fields(body):
        if field == 1 and wire == 5:
            response["audio_s"] = value
        elif field == 2 and wire == 5:
            response["rtf"] = value
    return response


def decode_error(body: bytes) -> dict:
    response = {"type": "error", "error": "daemon error"}
    for field, wire, value in iter_fields(body):
        if field == 1 and wire == 2:
            response["error"] = value.decode("utf-8")
    return response


def field_varint(out: bytearray, field: int, value: int) -> None:
    varint(out, (field << 3) | 0)
    varint(out, value)


def field_fixed32(out: bytearray, field: int, value: float) -> None:
    varint(out, (field << 3) | 5)
    out.extend(struct.pack("<f", value))


def field_string(out: bytearray, field: int, value: str) -> None:
    field_message(out, field, value.encode("utf-8"))


def field_message(out: bytearray, field: int, value: bytes) -> None:
    varint(out, (field << 3) | 2)
    varint(out, len(value))
    out.extend(value)


def varint(out: bytearray, value: int) -> None:
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def iter_fields(body: bytes):
    pos = 0
    while pos < len(body):
        key, pos = read_varint(body, pos)
        field = key >> 3
        wire = key & 0x7
        if wire == 0:
            value, pos = read_varint(body, pos)
        elif wire == 2:
            size, pos = read_varint(body, pos)
            value = body[pos : pos + size]
            pos += size
        elif wire == 5:
            value = struct.unpack_from("<f", body, pos)[0]
            pos += 4
        else:
            raise SystemExit(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def read_varint(body: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(body):
            raise SystemExit("truncated protobuf varint")
        byte = body[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, pos
        shift += 7
        if shift >= 64:
            raise SystemExit("protobuf varint too long")


def read_wav_s16le(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        validate_wav(wav, path)
        return wav_to_s16le(wav, wav.getnframes())


def iter_wav_s16le_chunks(path: Path, frame_ms: int):
    if frame_ms <= 0:
        raise SystemExit("--frame-ms must be positive")

    with wave.open(str(path), "rb") as wav:
        validate_wav(wav, path)
        frames_per_chunk = max(1, wav.getframerate() * frame_ms // 1000)
        while True:
            pcm = wav_to_s16le(wav, frames_per_chunk)
            if not pcm:
                break
            yield pcm


def validate_wav(wav: wave.Wave_read, path: Path) -> None:
    if wav.getnchannels() != 1:
        raise SystemExit(f"{path}: expected mono WAV, got {wav.getnchannels()} channels")
    if wav.getframerate() != 16_000:
        raise SystemExit(f"{path}: expected 16 kHz WAV, got {wav.getframerate()} Hz")
    if wav.getcomptype() != "NONE":
        raise SystemExit(f"{path}: compressed WAV is not supported")
    if wav.getsampwidth() not in (1, 2, 3, 4):
        raise SystemExit(f"{path}: unsupported sample width {wav.getsampwidth()} bytes")


def wav_to_s16le(wav: wave.Wave_read, frames: int) -> bytes:
    data = wav.readframes(frames)
    width = wav.getsampwidth()
    if width == 2:
        return data
    if width == 1:
        return pcm_u8_to_s16le(data)
    if width == 3:
        return pcm_s24le_to_s16le(data)
    if width == 4:
        return pcm_s32le_to_s16le(data)
    raise AssertionError("validated sample width should be handled")


def pcm_u8_to_s16le(data: bytes) -> bytes:
    out = bytearray(len(data) * 2)
    for i, sample in enumerate(data):
        value = (sample - 128) << 8
        struct.pack_into("<h", out, i * 2, clamp_i16(value))
    return bytes(out)


def pcm_s24le_to_s16le(data: bytes) -> bytes:
    out = bytearray((len(data) // 3) * 2)
    for i in range(0, len(data), 3):
        raw = data[i : i + 3]
        value = int.from_bytes(raw + (b"\xff" if raw[2] & 0x80 else b"\x00"), "little", signed=True)
        struct.pack_into("<h", out, (i // 3) * 2, clamp_i16(value >> 8))
    return bytes(out)


def pcm_s32le_to_s16le(data: bytes) -> bytes:
    out = bytearray((len(data) // 4) * 2)
    for i in range(0, len(data), 4):
        value = struct.unpack_from("<i", data, i)[0]
        struct.pack_into("<h", out, (i // 4) * 2, clamp_i16(value >> 16))
    return bytes(out)


def clamp_i16(value: int) -> int:
    return max(-32768, min(32767, value))


if __name__ == "__main__":
    raise SystemExit(main())
