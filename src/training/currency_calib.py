import random
import shutil
from pathlib import Path

from data_tools.layout import Layout
from data_tools.registry import get_spec
from data_tools.split import Split
from experiments.rows import load_items


def _reset_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def prepare_currency_calib(
    root: Path,
    calib_count: int = 200,
    bench_count: int = 100,
    seed: int = 42,
) -> tuple[Path, Path]:
    """copy deterministic train images for currency int8 calibration and benchmark"""
    layout = Layout(root)
    spec = get_spec(layout, "currency")
    items = sorted(load_items(layout, spec, Split.TRAIN), key=lambda x: x.path.name)
    needed = calib_count + bench_count
    if len(items) < needed:
        raise ValueError(f"need {needed} train images, found {len(items)}")
    rng = random.Random(seed)
    rng.shuffle(items)

    calib_dir = root / "data" / "calib_currency" / "calib"
    bench_dir = root / "data" / "calib_currency" / "bench"
    _reset_dir(calib_dir)
    _reset_dir(bench_dir)

    for idx, item in enumerate(items[:calib_count]):
        shutil.copy2(item.path, calib_dir / f"calib_{idx:04d}{item.path.suffix.lower()}")
    for idx, item in enumerate(items[calib_count:needed]):
        shutil.copy2(item.path, bench_dir / f"bench_{idx:04d}{item.path.suffix.lower()}")
    return calib_dir, bench_dir
