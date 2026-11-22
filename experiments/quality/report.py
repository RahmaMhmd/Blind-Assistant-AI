from src.imagequality.assess import Quality, classify
from src.imagequality.thresholds import Thresholds

from .samples import Sample


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def report(samples: list[Sample], thresholds: Thresholds) -> dict[str, float]:
    """precision and recall per class so run it once on the test split"""
    predicted = [classify(s.reading, thresholds) for s in samples]
    scores = {}

    for quality in Quality:
        hits = sum(
            1 for p, s in zip(predicted, samples) if p is quality and s.label is quality
        )
        scores[f"{quality.value}_precision"] = _ratio(hits, predicted.count(quality))
        scores[f"{quality.value}_recall"] = _ratio(
            hits, sum(1 for s in samples if s.label is quality)
        )

    return scores
