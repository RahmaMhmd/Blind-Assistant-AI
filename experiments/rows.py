from dataclasses import dataclass
from pathlib import Path

from data_tools.constants import FILE_COLUMN, GROUP_COLUMN
from data_tools.layout import Layout
from data_tools.manifest import Row, read_manifest
from data_tools.registry import DatasetSpec
from data_tools.split import Split, split_of


@dataclass(frozen=True)
class Item:
    path: Path
    row: Row


def load_items(layout: Layout, spec: DatasetSpec, split: Split) -> list[Item]:
    """manifest rows that belong to one split with their file paths"""
    folder = layout.dataset_dir(spec.name)
    rows = read_manifest(layout.manifest(spec.name))

    return [
        Item(folder / row[FILE_COLUMN], row)
        for row in rows
        if split_of(row[GROUP_COLUMN], spec.test_percent) is split
    ]
