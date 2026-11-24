from pathlib import Path

from .config import load_config, resolve_path


def prepare_calibration(
    output_dir: Path | None = None,
    count: int | None = None,
) -> list[Path]:
    """save real photos for model calibration"""
    from datasets import load_dataset

    cfg = load_config()["calibration"]
    paths_cfg = load_config()["paths"]
    target_dir = Path(output_dir) if output_dir else resolve_path(paths_cfg["calibration_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    limit = count if count is not None else int(cfg["count"])
    stream = load_dataset(cfg["dataset"], split=cfg["split"], streaming=True)
    saved: list[Path] = []

    for index, item in enumerate(stream):
        if index >= limit:
            break
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out_path = target_dir / f"calib_{index:04d}.jpg"
        img.save(out_path, format="JPEG", quality=int(cfg["quality"]))
        saved.append(out_path)

    return saved
