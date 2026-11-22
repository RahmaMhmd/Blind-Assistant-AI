import numpy as np


def _f_beta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    weight = beta**2
    top = (1 + weight) * precision * recall
    bottom = weight * precision + recall

    return np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)


def best_cutoff(values: np.ndarray, positives: np.ndarray, beta: float) -> float:
    """cutoff that predicts positive below it and maximizes f beta"""
    if positives.sum() == 0 or len(values) < 2:
        raise ValueError("need at least one positive and two samples")

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    hits = np.cumsum(positives[order])
    gaps = np.flatnonzero(ranked[:-1] < ranked[1:])

    if gaps.size == 0:
        raise ValueError("all values are equal")

    taken = gaps + 1
    precision = hits[gaps] / taken
    recall = hits[gaps] / positives.sum()
    best = gaps[np.argmax(_f_beta(precision, recall, beta))]

    return float((ranked[best] + ranked[best + 1]) / 2)
