from pathlib import Path

from src.contracts.models import StrictModel

TEXT_ENCODING = "utf-8"
DEFAULT_LONG_SIDE = 640


class ThresholdsNotCalibrated(RuntimeError):
    pass


class Thresholds(StrictModel):
    long_side: int = DEFAULT_LONG_SIDE
    min_sharpness: float | None = None
    min_brightness: float | None = None
    calibrated_on: str | None = None

    def require_ready(self) -> None:
        if self.min_sharpness is None or self.min_brightness is None:
            raise ThresholdsNotCalibrated(
                "run calibration on the image-quality dataset"
            )


def load_thresholds(path: Path) -> Thresholds:
    return Thresholds.model_validate_json(path.read_text(encoding=TEXT_ENCODING))


def save_thresholds(path: Path, thresholds: Thresholds) -> None:
    path.write_text(thresholds.model_dump_json(indent=1) + "\n", encoding=TEXT_ENCODING)
