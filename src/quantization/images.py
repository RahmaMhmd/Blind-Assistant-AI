from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def load_tensor(path: Path, size: int) -> np.ndarray:
    """load image as nchw float tensor"""
    with Image.open(path) as img:
        resized = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def list_images(folder: Path, limit: int | None = None) -> list[Path]:
    if not folder.is_dir():
        return []
    paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return paths[:limit] if limit else paths
