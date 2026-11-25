from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

from src.contracts.enums import VoiceIntent

ALLOWED_INTENTS = [item.value for item in VoiceIntent]
REQUIRED_COLUMNS = {"text", "intent", "speaker_id"}


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - cols)
        if missing:
            raise ValueError(f"missing csv columns: {missing}")
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]
    bad = [
        f"row {idx}: intent={r['intent']!r}"
        for idx, r in enumerate(rows, 2)
        if r["intent"] not in ALLOWED_INTENTS or not r["text"] or not r["speaker_id"]
    ]
    if bad:
        raise ValueError(f"invalid intent csv rows: {'; '.join(bad)}")
    return rows


def prepare_intent_dataset(
    csv_path: Path, root: Path, min_per_intent: int = 20, seed: int = 42
) -> tuple[Path, Path]:
    """validate team csv and build speaker grouped train and eval splits"""
    rows = _load_rows(csv_path)
    counts = Counter(r["intent"] for r in rows)
    short = [f"{name}={counts[name]}" for name in ALLOWED_INTENTS if counts[name] < min_per_intent]
    if short:
        raise ValueError(f"need at least {min_per_intent} rows per intent: {', '.join(short)}")

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker_id"]].append(r)
    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        raise ValueError("need at least 2 distinct speakers to split without leakage")
    random.Random(seed).shuffle(speakers)
    eval_speakers = set(speakers[: max(1, round(len(speakers) * 0.2))])

    train_rows = [r for s in speakers if s not in eval_speakers for r in by_speaker[s]]
    eval_rows = [r for s in speakers if s in eval_speakers for r in by_speaker[s]]

    train_jsonl = root / "data" / "intent" / "train.jsonl"
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with train_jsonl.open("w", encoding="utf-8") as handle:
        for r in train_rows:
            handle.write(json.dumps({"text": r["text"], "intent": r["intent"], "speaker_id": r["speaker_id"]}, ensure_ascii=False) + "\n")

    eval_dir = root / "data" / "eval" / "voice_intent_8"
    eval_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for idx, r in enumerate(eval_rows, 1):
        fname = f"intent_{idx:03d}.txt"
        (eval_dir / fname).write_text(r["text"] + "\n", encoding="utf-8")
        samples.append({"id": f"intent_{idx:03d}", "file": fname, "reference": r["intent"]})
    manifest = {"task": "intent_classification", "samples": samples}
    (eval_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return eval_dir, train_jsonl
