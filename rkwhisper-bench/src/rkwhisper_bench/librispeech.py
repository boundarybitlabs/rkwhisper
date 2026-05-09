from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Utterance:
    id: str
    audio_path: Path
    reference: str


def load_utterances(split_dir: Path) -> Iterator[Utterance]:
    """Walk a LibriSpeech split directory and yield utterances in sorted order.

    Expected layout:
      <split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
      <split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
    Each line in trans.txt: "<UTT_ID> REFERENCE TEXT IN UPPERCASE"
    """
    for trans_file in sorted(split_dir.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, reference = line.partition(" ")
                if not reference:
                    continue
                audio_path = chapter_dir / f"{utt_id}.flac"
                if audio_path.exists():
                    yield Utterance(id=utt_id, audio_path=audio_path, reference=reference)
