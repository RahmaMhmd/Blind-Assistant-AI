import pytest

from src.evaluation.metrics.detection import detection_metrics


def test_exact_box_match_gives_perfect_map():
    refs = [{"boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]], "labels": [0, 1]}]
    hyps = [{"boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]], "scores": [0.95, 0.9], "labels": [0, 1]}]
    scores = detection_metrics(refs, hyps)
    assert scores["map50"] == pytest.approx(1.0)
    assert scores["map50_95"] == pytest.approx(1.0)


def test_missed_detection_reduces_map():
    refs = [
        {"boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": [0]},
        {"boxes": [[0.2, 0.2, 0.6, 0.6]], "labels": [0]},
    ]
    hyps = [
        {"boxes": [[0.1, 0.1, 0.5, 0.5]], "scores": [0.9], "labels": [0]},
        {"boxes": [[0.0, 0.0, 0.05, 0.05]], "scores": [0.8], "labels": [0]},
    ]
    scores = detection_metrics(refs, hyps)
    assert scores["map50"] == pytest.approx(0.5)
    assert scores["map50_95"] == pytest.approx(0.5)


def test_empty_predictions_and_empty_ground_truth():
    assert detection_metrics([{"boxes": [], "labels": []}], [{"boxes": [], "scores": [], "labels": []}]) == {
        "map50": 0.0,
        "map50_95": 0.0,
    }
    assert detection_metrics(
        [{"boxes": [[0.1, 0.1, 0.4, 0.4]], "labels": [0]}],
        [{"boxes": [], "scores": [], "labels": []}],
    ) == {"map50": 0.0, "map50_95": 0.0}


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        detection_metrics([{"boxes": [], "labels": []}], [])
