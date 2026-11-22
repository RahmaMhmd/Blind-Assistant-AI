import json

from PIL import Image

from src.evaluation.prepare_ocr_words import prepare
from src.evaluation.testset import load_evalset

SOURCE, TARGET = "whole", "words"
IMAGE_SIZE = (200, 100)
WORD_POLYGON = [10, 10, 110, 10, 110, 50, 10, 50]  
TINY_POLYGON = [0, 0, 5, 0, 5, 5, 0, 5]
WORD_SIZE = (100, 40)


def _instance(text, polygon):
    return {"coordinates": polygon, "language": "Arabic", "text": text}


def _build_source(root, instances):
    (root / SOURCE / "images").mkdir(parents=True)
    Image.new("RGB", IMAGE_SIZE, "white").save(root / SOURCE / "images" / "a.png")
    sample = {
        "id": "img",
        "file": "images/a.png",
        "reference": "ignored",
        "instances": json.dumps(instances),  # stored as a string, like the real file
    }
    manifest = {"task": "ocr", "samples": [sample]}
    (root / SOURCE / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_words_are_cut_and_unreadable_or_tiny_ones_are_skipped(tmp_path):
    _build_source(tmp_path, [
        _instance("كلمة", WORD_POLYGON),
        _instance("###", WORD_POLYGON),
        _instance("صغيرة", TINY_POLYGON),
    ])

    assert prepare(SOURCE, TARGET, root=tmp_path) == 1

    evalset = load_evalset(TARGET, root=tmp_path)
    assert evalset.task == "ocr"
    assert [sample.reference for sample in evalset.samples] == ["كلمة"]
    assert Image.open(evalset.samples[0].path).size == WORD_SIZE
