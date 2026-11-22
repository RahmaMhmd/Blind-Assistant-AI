import json
from pathlib import Path

import pytest

from src.placement.build import SOURCE_DIR, TEXT_ENCODING, PlacementInvalid
from src.placement.cli import main

ROOT = Path(__file__).resolve().parents[2]


def _source(name):
    return json.loads(
        (ROOT / SOURCE_DIR / f"{name}.json").read_text(encoding=TEXT_ENCODING)
    )


def test_stale_output_fails_the_check(tmp_path):
    for name in ("ocr",):
        target = tmp_path / SOURCE_DIR
        target.mkdir(parents=True)
        (target / f"{name}.json").write_text(json.dumps(_source(name)))

    assert main(["check"], tmp_path) == 1
    assert main(["build"], tmp_path) == 0
    assert main(["check"], tmp_path) == 0


def test_invalid_source_stops_the_build(tmp_path):
    target = tmp_path / SOURCE_DIR
    target.mkdir(parents=True)
    raw = _source("ocr")
    raw["primary"]["name"] = "same"
    raw["fallback"]["name"] = "same"
    (target / "ocr.json").write_text(json.dumps(raw))

    with pytest.raises(PlacementInvalid):
        main(["build"], tmp_path)
