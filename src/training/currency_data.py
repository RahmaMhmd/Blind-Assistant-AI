import random
from pathlib import Path

from data_tools.layout import Layout
from data_tools.registry import get_spec
from data_tools.split import Split
from experiments.rows import load_items

DENOMINATIONS = ["5", "10", "20", "50", "100", "200"]


def split_train_val(
    items: list[tuple[Path, int, str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[tuple[Path, int, str]], list[tuple[Path, int, str]]]:
    """hold out validation groups with no group leakage"""
    groups = sorted({group for _, _, group in items})
    if len(groups) < 2 or val_ratio <= 0:
        return list(items), []
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, round(len(groups) * val_ratio))])
    train = [row for row in items if row[2] not in val_groups]
    val = [row for row in items if row[2] in val_groups]
    return train, val


def load_currency_splits(
    root: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    layout = Layout(root)
    spec = get_spec(layout, "currency")
    label_of = {name: idx for idx, name in enumerate(DENOMINATIONS)}
    train_raw = [
        (item.path, label_of[item.row["denomination"]], item.row["group"])
        for item in load_items(layout, spec, Split.TRAIN)
    ]
    test_rows = [
        (item.path, label_of[item.row["denomination"]], item.row["group"])
        for item in load_items(layout, spec, Split.TEST)
    ]
    train_rows, val_rows = split_train_val(train_raw, val_ratio, seed)
    return train_rows, val_rows, test_rows
