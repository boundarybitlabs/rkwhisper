#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///

import argparse
import json
import socket
import struct
import sys
import threading
import wave
from pathlib import Path


DEFAULT_SOCKET = "/var/rkwhisper/asr.sock"


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
        help="stream mode PCM frame duration in milliseconds",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="print final transcript text instead of daemon JSONL",
    )
    args = parser.parse_args()

    if args.mode == "batch":
        return run_batch(args)
    return run_stream(args)


def header_from_args(args: argparse.Namespace) -> dict:
    header = {
        "model": args.model,
        "mode": args.mode,
        "lang": args.lang,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "beam_size": args.beam_size,
        "notimestamps": args.notimestamps,
        "suppress_tokens": args.suppress_tokens,
    }
    optional_fields = {
        "vad_threshold": args.vad_threshold,
        "vad_min_speech_ms": args.vad_min_speech_ms,
        "vad_min_silence_ms": args.vad_min_silence_ms,
        "vad_speech_pad_ms": args.vad_speech_pad_ms,
        "vad_window_samples": args.vad_window_samples,
    }
    header.update({key: value for key, value in optional_fields.items() if value is not None})
    return header


def run_batch(args: argparse.Namespace) -> int:
    pcm = read_wav_s16le(args.wav)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(args.socket)
        send_header(sock, header_from_args(args))
        send_frame(sock, pcm)
        return read_responses(sock, text_only=args.text)


def run_stream(args: argparse.Namespace) -> int:
    transcript_parts: list[str] = []
    done = threading.Event()
    status = {"code": 0}

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(args.socket)
        send_header(sock, header_from_args(args))

        reader = threading.Thread(
            target=read_responses_thread,
            args=(sock.makefile("rb"), args.text, transcript_parts, done, status),
            daemon=True,
        )
        reader.start()

        try:
            for pcm in iter_wav_s16le_chunks(args.wav, args.frame_ms):
                send_frame(sock, pcm)
            send_frame(sock, b"")
        except BrokenPipeError:
            done.set()
            status["code"] = 1

        reader.join()
        if args.text and transcript_parts:
            print("".join(transcript_parts).strip())
        return status["code"]


def send_header(sock: socket.socket, header: dict) -> None:
    sock.sendall(json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n")


def send_frame(sock: socket.socket, pcm: bytes) -> None:
    sock.sendall(struct.pack("<i", len(pcm)))
    if pcm:
        sock.sendall(pcm)


def read_responses(sock: socket.socket, text_only: bool) -> int:
    transcript_parts: list[str] = []
    status = {"code": 0}
    read_responses_file(sock.makefile("rb"), text_only, transcript_parts, status)
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
    for raw_line in stream:
        line = raw_line.decode("utf-8").rstrip("\n")
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            print(line, file=sys.stderr)
            status["code"] = 1
            continue

        msg_type = message.get("type")
        if msg_type == "segment":
            if text_only:
                transcript_parts.append(message.get("text", ""))
                transcript_parts.append(" ")
            else:
                print(json.dumps(message, ensure_ascii=False), flush=True)
        elif msg_type == "done":
            if not text_only:
                print(json.dumps(message, ensure_ascii=False), flush=True)
            return
        elif msg_type == "error":
            print(message.get("error", "daemon error"), file=sys.stderr)
            status["code"] = 1
            return
        else:
            print(json.dumps(message, ensure_ascii=False), flush=True)


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
