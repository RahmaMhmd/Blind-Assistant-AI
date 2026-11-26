import numpy as np
import pytest

from experiments.quality.search import best_cutoff

BETA = 1.0


def test_finds_midpoint_between_classes():
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    positives = np.array([True, True, True, False, False, False])

    assert best_cutoff(values, positives, BETA) == pytest.approx(6.5)


def test_ignores_order_of_input():
    values = np.array([12.0, 1.0, 11.0, 3.0, 10.0, 2.0])
    positives = np.array([False, True, False, True, False, True])

    assert best_cutoff(values, positives, BETA) == pytest.approx(6.5)


def test_overlap_picks_best_f_score():
    values = np.array([1.0, 2.0, 6.0, 5.0, 11.0, 12.0])
    positives = np.array([True, True, True, False, False, False])

    assert best_cutoff(values, positives, BETA) == pytest.approx(8.5)


def test_higher_beta_favors_recall():
    values = np.array([1.0, 2.0, 5.0, 6.0, 7.0, 8.0])
    positives = np.array([True, True, False, True, False, False])
    strict = best_cutoff(values, positives, 0.5)
    loose = best_cutoff(values, positives, 3.0)

    assert loose > strict


@pytest.mark.parametrize(
    ("values", "positives"),
    [
        (np.array([1.0, 2.0]), np.array([False, False])),
        (np.array([1.0]), np.array([True])),
        (np.array([3.0, 3.0]), np.array([True, False])),
    ],
)
def test_rejects_unusable_input(values, positives):
    with pytest.raises(ValueError):
        best_cutoff(values, positives, BETA)
