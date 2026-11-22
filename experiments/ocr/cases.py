from dataclasses import dataclass

from data_tools.layout import Layout
from data_tools.registry import DatasetSpec
from data_tools.split import Split

from ..rows import load_items

LANGUAGE_COLUMN = "language"
TEXT_COLUMN = "text"


@dataclass(frozen=True)
class OcrCase:
    image: bytes
    language: str
    text: str


def load_ocr_cases(
    layout: Layout, spec: DatasetSpec, split: Split
) -> list[OcrCase]:
    items = load_items(layout, spec, split)

    if not items:
        raise ValueError(f"no {split.value} cases in {spec.name}")

    return [
        OcrCase(
            item.path.read_bytes(),
            item.row[LANGUAGE_COLUMN],
            item.row[TEXT_COLUMN],
        )
        for item in items
    ]
