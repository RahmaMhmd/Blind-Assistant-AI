import json
from pathlib import Path

import pytest
from jsonschema import validate

from src.placement.build import (
    BUILT_FILE,
    SCHEMA_FILE,
    SOURCE_DIR,
    TEXT_ENCODING,
    load_sources,
)
from src.placement.checks import entry_problems
from src.placement.cli import main
from src.placement.models import FeatureEntry

ROOT = Path(__file__).resolve().parents[2]


def _source(name):
    return json.loads(
        (ROOT / SOURCE_DIR / f"{name}.json").read_text(encoding=TEXT_ENCODING)
    )


def _entry(**changes):
    raw = _source("ocr")
    raw.update(changes)

    return FeatureEntry.model_validate(raw)


def test_committed_outputs_are_in_sync():
    assert main(["check"], ROOT) == 0


def test_built_file_matches_the_committed_schema():
    built = json.loads((ROOT / BUILT_FILE).read_text(encoding=TEXT_ENCODING))
    schema = json.loads((ROOT / SCHEMA_FILE).read_text(encoding=TEXT_ENCODING))

    validate(built, schema)


def test_sources_cover_the_three_features():
    assert set(load_sources(ROOT).features) == {"ocr", "image_quality", "qr_barcode"}


def test_tentative_entry_has_no_problems():
    assert entry_problems("ocr", _entry()) == []


def test_confirmed_entry_needs_a_pinned_model():
    problems = entry_problems("ocr", _entry(status="confirmed"))

    assert any("sha256" in problem for problem in problems)
    assert any("version" in problem for problem in problems)


def test_unknown_feature_is_reported():
    assert entry_problems("teleport", _entry()) == ["teleport unknown feature"]


def test_same_primary_and_fallback_is_reported():
    raw = _source("ocr")
    raw["fallback"] = raw["primary"]

    problems = entry_problems("ocr", FeatureEntry.model_validate(raw))

    assert "ocr primary and fallback are the same" in problems


def test_blocked_license_cannot_be_production():
    raw = _source("ocr")
    raw["primary"]["license"]["commercialUse"] = "blocked"

    assert entry_problems("ocr", FeatureEntry.model_validate(raw))


def test_bad_sha_is_rejected_by_the_model():
    raw = _source("ocr")
    raw["primary"]["sha256"] = "abc"

    with pytest.raises(ValueError):
        FeatureEntry.model_validate(raw)
