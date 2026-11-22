from dataclasses import dataclass

import numpy as np

from data_tools.layout import Layout
from data_tools.registry import DatasetSpec
from data_tools.split import Split
from src.imagequality.metrics import decode_gray

from ..rows import load_items

KIND_COLUMN = "code_kind"
VALUE_COLUMN = "value"


@dataclass(frozen=True)
class Case:
    gray: np.ndarray
    kind: str
    value: str


def load_cases(layout: Layout, spec: DatasetSpec, split: Split) -> list[Case]:
    items = load_items(layout, spec, split)

    if not items:
        raise ValueError(f"no {split.value} cases in {spec.name}")

    return [
        Case(
            decode_gray(item.path.read_bytes()),
            item.row[KIND_COLUMN],
            item.row[VALUE_COLUMN],
        )
        for item in items
    ]
