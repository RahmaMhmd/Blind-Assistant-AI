import json

import pytest

from src.evaluation.testset import MANIFEST_NAME, load_evalset

SET_NAME = "demo"
TASK = "speech_recognition"
SAMPLES = [
    {"id": "a", "file": "a.wav", "reference": "one"},
    {"id": "b", "file": "b.wav", "reference": "two"},
]


def _build(root, samples=SAMPLES, create_files=True):
    folder = root / SET_NAME
    folder.mkdir()
    manifest = {"task": TASK, "samples": samples}
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    if create_files:
        for item in samples:
            (folder / item["file"]).touch()


def test_loads_samples(tmp_path):
    _build(tmp_path)
    evalset = load_evalset(SET_NAME, root=tmp_path)

    assert evalset.task == TASK
    assert [sample.id for sample in evalset.samples] == ["a", "b"]
    assert evalset.samples[0].reference == "one"


def test_duplicate_ids_rejected(tmp_path):
    _build(tmp_path, samples=[SAMPLES[0], SAMPLES[0]])

    with pytest.raises(ValueError):
        load_evalset(SET_NAME, root=tmp_path)


def test_missing_file_rejected(tmp_path):
    _build(tmp_path, create_files=False)

    with pytest.raises(FileNotFoundError):
        load_evalset(SET_NAME, root=tmp_path)


def test_empty_samples_rejected(tmp_path):
    _build(tmp_path, samples=[])

    with pytest.raises(ValueError):
        load_evalset(SET_NAME, root=tmp_path)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evalset(SET_NAME, root=tmp_path)
