#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rkwhisper-client",
# ]
# ///

import argparse
import sys
import wave
import threading
from pathlib import Path
from rkwhisper_client import SyncSession, ClientHello, VadOptions, Segment, Done, SpeechStarted, SpeechEnded


def main() -> int:
    parser = argparse.ArgumentParser(description="Client for rkwhisperd")
    parser.add_argument("wav", type=Path, help="mono 16 kHz WAV file")
    parser.add_argument("--socket", default="/run/rkwhisper/asr.sock", help="rkwhisperd Unix socket")
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

    hello = ClientHello(
        model=args.model,
        mode=args.mode,
        lang=args.lang,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        beam_size=args.beam_size,
        notimestamps=args.notimestamps,
        suppress_tokens=args.suppress_tokens,
        vad=VadOptions(
            threshold=args.vad_threshold,
            min_speech_ms=args.vad_min_speech_ms,
            min_silence_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
            window_samples=args.vad_window_samples,
        ),
        client_id="pywhisper-client-cli",
    )

    try:
        with SyncSession.connect(args.socket, hello) as session:
            pcm_data = read_wav_s16le(args.wav)
            
            if args.mode == "batch":
                session.send_audio(pcm_data)
                session.finish()

                # 2. Consume results
                transcript_parts = []
                for resp in session:
                    if isinstance(resp, Segment):
                        if args.text:
                            transcript_parts.append(resp.text)
                        else:
                            print({
                                "type": "segment",
                                "text": resp.text,
                                "begin": resp.begin,
                                "end": resp.end
                            }, flush=True)
                    elif isinstance(resp, SpeechStarted):
                        if not args.text:
                            print({"type": "speech_started", "begin": resp.begin}, flush=True)
                    elif isinstance(resp, SpeechEnded):
                        if not args.text:
                            print({"type": "speech_ended", "end": resp.end}, flush=True)
                    elif isinstance(resp, Done):
                        if not args.text:
                            print({
                                "type": "done",
                                "audio_s": resp.audio_s,
                                "rtf": resp.rtf
                            }, flush=True)

                if args.text and transcript_parts:
                    print(" ".join(transcript_parts).strip())
            else:
                # In stream mode, split and use a background thread for responses
                sender, receiver = session.split()
                transcript_parts = []

                def receiver_thread():
                    for resp in receiver:
                        if isinstance(resp, Segment):
                            if args.text:
                                transcript_parts.append(resp.text)
                            else:
                                print({
                                    "type": "segment",
                                    "text": resp.text,
                                    "begin": resp.begin,
                                    "end": resp.end
                                }, flush=True)
                        elif isinstance(resp, SpeechStarted):
                            if not args.text:
                                print({"type": "speech_started", "begin": resp.begin}, flush=True)
                        elif isinstance(resp, SpeechEnded):
                            if not args.text:
                                print({"type": "speech_ended", "end": resp.end}, flush=True)
                        elif isinstance(resp, Done):
                            if not args.text:
                                print({
                                    "type": "done",
                                    "audio_s": resp.audio_s,
                                    "rtf": resp.rtf
                                }, flush=True)

                t = threading.Thread(target=receiver_thread, daemon=True)
                t.start()

                # In stream mode, send in chunks
                chunk_size = (16000 * 2 * args.frame_ms) // 1000
                for i in range(0, len(pcm_data), chunk_size):
                    sender.send_audio(pcm_data[i:i+chunk_size])
                
                sender.finish()
                
                # Wait for receiver thread to finish
                t.join()

                if args.text and transcript_parts:
                    print(" ".join(transcript_parts).strip())
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def read_wav_s16le(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise SystemExit(f"{path}: expected mono WAV, got {wav.getnchannels()} channels")
        if wav.getframerate() != 16_000:
            raise SystemExit(f"{path}: expected 16 kHz WAV, got {wav.getframerate()} Hz")
        if wav.getsampwidth() != 2:
            # For simplicity in this CLI tool, we only handle s16le directly now.
            # The previous version had complex PCM conversion which we can restore if needed,
            # but usually users provide 16-bit mono.
            raise SystemExit(f"{path}: expected 16-bit samples, got {wav.getsampwidth()*8}-bit")
        return wav.readframes(wav.getnframes())


if __name__ == "__main__":
    raise SystemExit(main())
