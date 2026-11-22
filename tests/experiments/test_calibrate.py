import pytest

from data_tools.split import Split
from experiments.quality.calibrate import calibrate
from experiments.quality.report import report
from experiments.quality.samples import load_samples
from src.imagequality.assess import Quality
from tests.experiments.support import DATASET, build_dataset

MINIMUM_SCORE = 0.9


@pytest.fixture
def dataset(tmp_path):
    return build_dataset(tmp_path)


def test_splits_do_not_overlap(dataset):
    layout, spec = dataset
    train = load_samples(layout, spec, Split.TRAIN)
    test = load_samples(layout, spec, Split.TEST)

    assert train and test
    assert len(train) + len(test) == 72


def test_calibration_separates_classes_on_test(dataset):
    layout, spec = dataset
    thresholds = calibrate(load_samples(layout, spec, Split.TRAIN), DATASET)
    scores = report(load_samples(layout, spec, Split.TEST), thresholds)

    assert thresholds.calibrated_on == DATASET
    assert all(score >= MINIMUM_SCORE for score in scores.values())
    expected = {f"{q.value}_{k}" for q in Quality for k in ("precision", "recall")}
    assert set(scores) == expected


def test_empty_split_is_rejected(tmp_path):
    layout, spec = build_dataset(tmp_path)
    spec = spec.model_copy(update={"test_percent": 1})
    layout.manifest(DATASET).write_text("file,group,batch,quality\n")

    with pytest.raises(ValueError):
        load_samples(layout, spec, Split.TEST)
