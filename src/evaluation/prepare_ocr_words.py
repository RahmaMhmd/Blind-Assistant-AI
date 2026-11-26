import argparse
import json
from math import dist
from pathlib import Path

from PIL import Image

from .config import REPO_ROOT, load_config
from .testset import MANIFEST_NAME

SECTION = "prepare_ocr_words"
WORDS_DIR = "words"
IMAGE_SUFFIX = ".png"

def _upright_crop(image: Image.Image, points: list[float]) -> Image.Image | None:
    """Cut one word out of its four-point polygon and straighten it."""
    x1, y1, x2, y2, x3, y3, x4, y4 = points  
    width = round(max(dist((x1, y1), (x2, y2)), dist((x4, y4), (x3, y3))))
    height = round(max(dist((x1, y1), (x4, y4)), dist((x2, y2), (x3, y3))))

    if width < 1 or height < 1:
        return None

    corners = (x1, y1, x4, y4, x3, y3, x2, y2)  
    return image.transform((width, height), Image.Transform.QUAD, corners, Image.Resampling.BICUBIC)


def _instances(item: dict) -> list[dict]:
    instances = item["instances"]
    return json.loads(instances) if isinstance(instances, str) else instances


def prepare(source: str, target: str, root: Path | None = None) -> int:
    """Turn a test set of whole images into a test set of single-word images."""
    settings = load_config()[SECTION]
    root = Path(root) if root else REPO_ROOT / load_config()["paths"]["testsets_dir"]
    source_dir, target_dir = root / source, root / target
    manifest = json.loads((source_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    (target_dir / WORDS_DIR).mkdir(parents=True, exist_ok=True)
    samples = []

    for item in manifest["samples"]:
        with Image.open(source_dir / item["file"]) as opened:
            image = opened.convert("RGB")

        for number, instance in enumerate(_instances(item), start=1):
            text = instance["text"].strip()

            if not text or settings["ignore_text"] in text:
                continue

            crop = _upright_crop(image, instance["coordinates"])

            if crop is None or min(crop.size) < settings["min_side_px"]:
                continue

            sample_id = f"{item['id']}_w{number:02d}"
            file = f"{WORDS_DIR}/{sample_id}{IMAGE_SUFFIX}"
            crop.save(target_dir / file)
            samples.append({"id": sample_id, "file": file, "reference": text})

    output = {"task": manifest["task"], "samples": samples}
    (target_dir / MANIFEST_NAME).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return len(samples)
def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.prepare_ocr_words")
    parser.add_argument("--source", required=True, help="test set of whole images")
    parser.add_argument("--target", required=True, help="name of the new word-level test set")
    args = parser.parse_args()
    print(f"saved {prepare(args.source, args.target)} word images in test set '{args.target}'")

if __name__ == "__main__":
    main()
