from dataclasses import dataclass

from data_tools.layout import Layout
from data_tools.registry import DatasetSpec
from data_tools.split import Split
from src.imagequality.assess import Quality, Reading, measure
from src.imagequality.thresholds import DEFAULT_LONG_SIDE

from ..rows import load_items
from .constants import QUALITY_COLUMN


@dataclass(frozen=True)
class Sample:
    reading: Reading
    label: Quality


def load_samples(
    layout: Layout,
    spec: DatasetSpec,
    split: Split,
    long_side: int = DEFAULT_LONG_SIDE,
) -> list[Sample]:
    items = load_items(layout, spec, split)

    if not items:
        raise ValueError(f"no {split.value} samples in {spec.name}")

    return [
        Sample(
            measure(item.path.read_bytes(), long_side),
            Quality(item.row[QUALITY_COLUMN]),
        )
        for item in items
    ]
