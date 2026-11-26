import json
from pathlib import Path

from PIL import Image

from data_tools.layout import Layout
from data_tools.naming import ParsedName, build_name
from data_tools.registry import SPEC_ROOT_KEY

DATASET = "demo"
BATCH = "b01"
EXIF_ORIENTATION = 0x0112
EXIF_DESCRIPTION = 0x010E
ROTATE_QUARTER = 6
IMAGE_SIZE = (64, 48)
SPECS = {
    DATASET: {"kind": "image", "labels": {"quality": ["sharp", "blur"], "note": None}}
}


def name_of(group: int, seq: int = 1, batch: str = BATCH, ext: str = ".jpg") -> str:
    parts = ParsedName(DATASET, batch, f"g{group:03d}", f"{seq:03d}", ext)

    return build_name(parts)


def make_layout(root: Path, specs: dict = SPECS) -> Layout:
    layout = Layout(root)
    layout.data.mkdir(parents=True, exist_ok=True)
    layout.spec_file.write_text(json.dumps({SPEC_ROOT_KEY: specs}))

    return layout


def make_image(path: Path, shade: int = 100, tagged: bool = False) -> Path:
    image = Image.new("RGB", IMAGE_SIZE, (shade, shade, shade))
    options = {}

    if tagged:
        exif = Image.Exif()
        exif[EXIF_ORIENTATION] = ROTATE_QUARTER
        exif[EXIF_DESCRIPTION] = "private place"
        options["exif"] = exif

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **options)

    return path


class FakeDetector:
    def __init__(self, flagged: set[str] = frozenset()):
        self.flagged = flagged

    def count_faces(self, path: Path) -> int:
        return 1 if path.name in self.flagged else 0
