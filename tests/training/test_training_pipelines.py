import json
from pathlib import Path

import pytest

from src.contracts.enums import VoiceIntent
from src.training.currency_data import DENOMINATIONS, split_train_val
from src.training.intent_data import prepare_intent_dataset


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    lines = ["text,intent,speaker_id,source,written_by"]
    lines.extend(f"{txt},{intent},{spk},team,ali" for txt, intent, spk in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_currency_label_order_and_group_split():
    assert DENOMINATIONS == ["5", "10", "20", "50", "100", "200"]
    items = [(Path(f"{i}.jpg"), i % 6, f"g{i // 2}") for i in range(40)]
    train, val = split_train_val(items, val_ratio=0.1, seed=42)
    train_groups = {g for _, _, g in train}
    val_groups = {g for _, _, g in val}
    assert train and val and train_groups.isdisjoint(val_groups)


def test_intent_data_rejects_bad_label(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "in.csv", [("sample one", "alarm_set", "s1")])
    with pytest.raises(ValueError, match="alarm_set"):
        prepare_intent_dataset(csv_path, tmp_path, min_per_intent=1)


def test_intent_data_enforces_minimum_count(tmp_path: Path):
    rows = [(f"txt {v.value} {i}", v.value, f"s{i % 3}") for v in VoiceIntent for i in range(2)]
    csv_path = _write_csv(tmp_path / "in.csv", rows)
    with pytest.raises(ValueError, match="at least 3"):
        prepare_intent_dataset(csv_path, tmp_path, min_per_intent=3)


def test_intent_data_speaker_split_and_manifest(tmp_path: Path):
    rows = []
    for spk_idx in range(5):
        for v in VoiceIntent:
            for j in range(5):
                rows.append((f"phrase {v.value} {spk_idx} {j}", v.value, f"spk_{spk_idx}"))
    csv_path = _write_csv(tmp_path / "in.csv", rows)
    eval_dir, train_jsonl = prepare_intent_dataset(csv_path, tmp_path, min_per_intent=20, seed=42)

    train_records = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    train_speakers = {r["speaker_id"] for r in train_records}

    assert manifest["task"] == "intent_classification"
    assert len(train_records) + len(manifest["samples"]) == len(rows)
    assert len(train_speakers) < 5
    for sample in manifest["samples"]:
        assert (eval_dir / sample["file"]).is_file()
        assert sample["reference"] in {v.value for v in VoiceIntent}
