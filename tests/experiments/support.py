from pathlib import Path

import cv2
import numpy as np

from data_tools.constants import BASE_COLUMNS
from data_tools.layout import Layout
from data_tools.manifest import write_manifest
from data_tools.registry import DatasetSpec, Kind
from src.imagequality.assess import Quality

SIDE = 160
PER_CLASS = 24
TEST_PERCENT = 50
DIM_FACTOR = 0.15
BLUR_KERNEL = (25, 25)
JPEG_EXT = ".jpg"
DATASET = "image-quality"


def make_image(quality: Quality, seed: int) -> np.ndarray:
    base = np.random.default_rng(seed).integers(0, 256, (SIDE, SIDE), dtype=np.uint8)

    if quality is Quality.BLUR:
        return cv2.GaussianBlur(base, BLUR_KERNEL, 0)
    if quality is Quality.DARK:
        return (base * DIM_FACTOR).astype(np.uint8)

    return base


def build_dataset(root: Path) -> tuple[Layout, DatasetSpec]:
    layout = Layout(root)
    spec = DatasetSpec(
        name=DATASET,
        kind=Kind.IMAGE,
        labels={"quality": [q.value for q in Quality]},
        test_percent=TEST_PERCENT,
    )
    folder = layout.dataset_dir(DATASET)
    folder.mkdir(parents=True)
    rows = []

    for index, quality in enumerate(Quality):
        for count in range(PER_CLASS):
            seed = index * PER_CLASS + count
            name = f"{DATASET}_b01_g{seed:03d}_001{JPEG_EXT}"
            cv2.imwrite(str(folder / name), make_image(quality, seed))
            row = {"file": name, "group": f"g{seed:03d}", "batch": "b01"}
            rows.append({**row, "quality": quality.value})

    write_manifest(layout.manifest(DATASET), rows, [*BASE_COLUMNS, "quality"])

    return layout, spec
