from pathlib import Path

import pytest

pytest.importorskip("jiwer")

import inspect  # noqa: E402

from src.evaluation.__main__ import main  # noqa: E402
from src.evaluation.config import load_config  # noqa: E402
from src.evaluation.registry import ADAPTERS, build_adapter  # noqa: E402
from src.evaluation.report import format_rows, row_records, summarize  # noqa: E402
from src.evaluation.runner import Row, RunResult, run  # noqa: E402
from src.evaluation.testset import EvalSet, Sample  # noqa: E402

FAKE_MODEL = "fake-speech"
SPEECH_TASK = "speech_recognition"
OTHER_TASK = "image_classification"
UNUSED_PATH = Path("unused")
HALF_WRONG = 0.5
SAMPLES = [
    Sample("s01", UNUSED_PATH, "افتح الكاميرا"),
    Sample("s02", UNUSED_PATH, "اقرأ النص"),
]


def test_fake_model_run_gives_expected_numbers():
    summary = summarize(run(build_adapter(FAKE_MODEL), EvalSet("t", SPEECH_TASK, SAMPLES)))

    assert summary["samples"] == len(SAMPLES)
    assert summary["wer"] == HALF_WRONG
    assert summary["mean_seconds"] >= 0
    assert summary["max_seconds"] >= summary["mean_seconds"]


def test_task_mismatch_is_rejected():
    with pytest.raises(ValueError):
        run(build_adapter(FAKE_MODEL), EvalSet("t", OTHER_TASK, SAMPLES))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_adapter("no-such-model")


def test_command_reports_error_instead_of_crashing():
    assert main(["--model", "no-such-model", "--testset", "x"]) == 1


def test_every_configured_model_matches_its_adapter():
    for name, spec in load_config()["models"].items():
        accepted = set(inspect.signature(ADAPTERS[spec["adapter"]]).parameters)
        assert set(spec.get("params", {})) <= accepted, name


def test_rows_show_reference_and_output():
    result = run(build_adapter(FAKE_MODEL), EvalSet("t", SPEECH_TASK, SAMPLES))
    text = format_rows(result)

    assert "s01" in text and "reference" in text and "output" in text


LONG_REPEAT = 1000


def test_rows_show_scores_and_cut_long_output_in_the_printout():
    row = Row("x", "كلمة " * LONG_REPEAT, "كلمة", 0.0)
    text = format_rows(RunResult("m", "t", SPEECH_TASK, [row]))

    assert "wer=" in text and "cer=" in text
    assert len(text) < LONG_REPEAT


def test_model_variants_share_the_base_settings():
    models = load_config()["models"]
    base = models["whisper-small-eg"]["params"]
    beam = models["whisper-small-eg-beam"]["params"]

    assert beam["model_id"] == base["model_id"]
    assert beam["chunk_length_s"] is None and beam["generate_kwargs"]


def test_row_records_hold_each_sample_with_its_own_scores():
    records = row_records(run(build_adapter(FAKE_MODEL), EvalSet("t", SPEECH_TASK, SAMPLES)))

    assert [record["id"] for record in records] == ["s01", "s02"]
    assert records[0]["wer"] == 0 and records[1]["wer"] > 0
