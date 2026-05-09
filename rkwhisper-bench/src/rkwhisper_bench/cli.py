"""rkwhisper-bench: WER and RTF benchmark for rkwhisperd against LibriSpeech."""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from rkwhisper_client import ClientHello, Done, Segment, SyncSession, VadOptions

from .librispeech import Utterance, load_utterances

TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

# No silence between concatenated utterances.  Even a short gap causes Whisper
# to emit <|endoftext|> and stop generating for the rest of the window.
_SILENCE_PCM: bytes = b""

# Token budget per 30-second Whisper window.  The protocol default of 128 is
# fine for a single short utterance but far too small when multiple utterances
# are packed into one window.
_MAX_NEW_TOKENS = 1000


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_pcm(path: Path) -> tuple[bytes, float]:
    """Decode a FLAC file to 16-bit LE PCM bytes and return (pcm, duration_s)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr} Hz")
    if data.ndim == 2:
        data = data.mean(axis=1)
    duration = len(data) / 16000.0
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, duration


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_group(
    socket_path: str,
    hello: ClientHello,
    pcm_list: list[bytes],
) -> tuple[str, float, float]:
    """Transcribe one or more utterances concatenated as a single request.

    A short silence gap is inserted between utterances so the server VAD
    treats each as a separate speech segment.

    Returns:
        hypothesis      – full batch transcript (all segments joined)
        server_rtf      – RTF reported by the daemon in Done
        wall_elapsed_s  – wall-clock seconds from first byte to Done
    """
    pieces: list[bytes] = []
    for i, pcm in enumerate(pcm_list):
        pieces.append(pcm)
        if i < len(pcm_list) - 1:
            pieces.append(_SILENCE_PCM)
    combined = b"".join(pieces)

    session = SyncSession.connect(socket_path, hello)
    sender, receiver = session.split()
    send_exc: Exception | None = None

    def _send() -> None:
        nonlocal send_exc
        try:
            sender.send_audio(combined)
            sender.finish()
        except Exception as e:
            send_exc = e

    t = threading.Thread(target=_send, daemon=True)
    wall_start = time.monotonic()
    t.start()

    parts: list[str] = []
    server_rtf = 0.0
    for resp in receiver:
        if isinstance(resp, Segment):
            parts.append(resp.text.strip())
        elif isinstance(resp, Done):
            server_rtf = resp.rtf
            break

    wall_elapsed = time.monotonic() - wall_start
    t.join()
    if send_exc is not None:
        raise send_exc

    return " ".join(parts), server_rtf, wall_elapsed


# ---------------------------------------------------------------------------
# Per-split stats accumulator
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    name: str
    utterances: int = 0
    errors: int = 0
    # WER edit-distance components (accumulated for corpus-level WER)
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    # Timing
    total_audio_s: float = 0.0
    total_wall_s: float = 0.0
    total_server_processing_s: float = 0.0  # sum of (audio_s * server_rtf)

    def add(self, other: "SplitStats") -> None:
        self.utterances += other.utterances
        self.errors += other.errors
        self.hits += other.hits
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.total_audio_s += other.total_audio_s
        self.total_wall_s += other.total_wall_s
        self.total_server_processing_s += other.total_server_processing_s

    @property
    def ref_words(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return 0.0
        return (self.substitutions + self.deletions + self.insertions) / self.ref_words

    @property
    def server_rtf(self) -> float:
        if self.total_audio_s == 0:
            return 0.0
        return self.total_server_processing_s / self.total_audio_s

    @property
    def wall_rtf(self) -> float:
        if self.total_audio_s == 0:
            return 0.0
        return self.total_wall_s / self.total_audio_s


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def bench_split(
    split_dir: Path,
    socket_path: str,
    hello: ClientHello,
    limit: int | None,
    verbose: bool,
    batch_size: int,
) -> SplitStats:
    utterances = list(load_utterances(split_dir))
    if limit is not None:
        utterances = utterances[:limit]

    stats = SplitStats(name=split_dir.name, utterances=len(utterances))
    total = len(utterances)
    width = len(str(total))

    print(
        f"\nBenchmarking {split_dir.name} ({total} utterances, batch_size={batch_size})..."
    )

    batch_num = 0
    utt_cursor = 0
    while utt_cursor < total:
        batch_utts = utterances[utt_cursor : utt_cursor + batch_size]
        utt_cursor += len(batch_utts)
        batch_num += 1
        first_id = batch_utts[0].id
        last_id = batch_utts[-1].id
        label = first_id if len(batch_utts) == 1 else f"{first_id}…{last_id}"

        # Load audio for every utterance in this batch.
        loaded: list[tuple[Utterance, bytes, float]] = []
        for utt in batch_utts:
            try:
                pcm, duration = load_audio_pcm(utt.audio_path)
                loaded.append((utt, pcm, duration))
            except Exception as e:
                print(f"  {label} ERROR (audio) {utt.id}: {e}", file=sys.stderr)
                stats.errors += 1

        if not loaded:
            continue

        pcm_list = [pcm for _, pcm, _ in loaded]
        dur_list = [dur for _, _, dur in loaded]
        batch_audio_s = sum(dur_list)

        try:
            hyp, server_rtf, wall_elapsed = transcribe_group(
                socket_path, hello, pcm_list
            )
        except Exception as e:
            print(f"  {label} ERROR (transcribe): {e}", file=sys.stderr)
            stats.errors += len(loaded)
            continue

        # WER is computed at the batch level: concatenated reference vs full
        # batch hypothesis.  This gives the same corpus-level metric as
        # per-utterance processing, and avoids the need to attribute Whisper
        # segments back to individual utterances.
        ref = " ".join(utt.reference for utt, _, _ in loaded)
        result = jiwer.process_words(
            ref,
            hyp,
            reference_transform=TRANSFORM,
            hypothesis_transform=TRANSFORM,
        )

        batch_ref_words = result.hits + result.substitutions + result.deletions
        batch_wer = (
            (result.substitutions + result.deletions + result.insertions)
            / batch_ref_words
            if batch_ref_words > 0
            else 0.0
        )

        stats.hits += result.hits
        stats.substitutions += result.substitutions
        stats.deletions += result.deletions
        stats.insertions += result.insertions
        stats.total_audio_s += batch_audio_s
        stats.total_wall_s += wall_elapsed
        stats.total_server_processing_s += server_rtf * batch_audio_s

        n = len(loaded)
        ids_label = f"{first_id}" if n == 1 else f"{first_id}…{last_id} ({n} utts)"
        print(
            f"  [batch {batch_num:>{width}}] {ids_label}"
            f" | WER: {batch_wer:5.1%} | RTF: {server_rtf:.3f} | {batch_audio_s:.1f}s"
        )
        if verbose:
            ref_norm = " ".join(TRANSFORM([ref])[0])
            hyp_norm = " ".join(TRANSFORM([hyp])[0]) if hyp.strip() else "(empty)"
            print(f"    REF: {ref_norm}")
            print(f"    HYP: {hyp_norm}")

    return stats


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary(stats: SplitStats) -> None:
    hours = stats.total_audio_s / 3600.0
    ok = stats.utterances - stats.errors
    print()
    print(f"{'=' * 52}")
    print(f"  {stats.name}")
    print(f"{'=' * 52}")
    print(f"  Utterances:    {ok}/{stats.utterances}")
    print(f"  Ref words:     {stats.ref_words:,}")
    print(f"  WER:           {stats.wer:.2%}")
    print(f"    Substitutions: {stats.substitutions:,}")
    print(f"    Deletions:     {stats.deletions:,}")
    print(f"    Insertions:    {stats.insertions:,}")
    print(f"  RTF (server):  {stats.server_rtf:.4f}")
    print(f"  RTF (wall):    {stats.wall_rtf:.4f}")
    print(f"  Total audio:   {hours:.2f}h")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rkwhisper-bench",
        description="Benchmark rkwhisperd against LibriSpeech — reports WER and RTF.",
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="DIR",
        help="LibriSpeech split directory (e.g. /data/LibriSpeech/test-clean)",
    )
    parser.add_argument(
        "--socket",
        default="/run/rkwhisper/asr.sock",
        metavar="PATH",
        help="rkwhisperd Unix socket (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="whisper-small-30s",
        metavar="NAME",
        help="Model name (default: %(default)s)",
    )
    parser.add_argument("--lang", default="en", metavar="LANG")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N utterances per split",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help=(
            "Disable server-side VAD by setting threshold=2.0, forcing fixed 30 s windows. "
            "Recommended when batching utterances, as the VAD may drop short utterances "
            "in concatenated audio."
        ),
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="Beam search width (default: %(default)s; 1 = greedy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=9,
        metavar="N",
        help=(
            "Utterances per request (default: %(default)s). "
            "At ~10 s/utterance, 9 utterances ≈ 90 s ≈ 3 windows, "
            "fully utilising all three NPU cores on RK3588."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print normalised REF and HYP for each batch",
    )
    args = parser.parse_args()

    hello = ClientHello(
        model=args.model,
        mode="batch",
        lang=args.lang,
        max_new_tokens=_MAX_NEW_TOKENS,
        beam_size=args.beam_size,
        notimestamps=True,
        vad=VadOptions(threshold=2.0) if args.no_vad else VadOptions(),
        client_id="rkwhisper-bench",
    )

    all_stats: list[SplitStats] = []

    for dataset_path in args.datasets:
        path = Path(dataset_path)
        if not path.is_dir():
            print(f"ERROR: {path} is not a directory", file=sys.stderr)
            sys.exit(1)

        stats = bench_split(
            path, args.socket, hello, args.limit, args.verbose, args.batch_size
        )
        print_summary(stats)
        all_stats.append(stats)

    if len(all_stats) > 1:
        total = SplitStats(name="TOTAL")
        for s in all_stats:
            total.add(s)
        print_summary(total)
