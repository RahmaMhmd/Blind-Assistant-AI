from pathlib import Path

from PIL import Image, ImageOps

from .constants import (
    IMAGE_FORMATS,
    JPEG_QUALITY,
    JPEG_SUBSAMPLING,
    METADATA_KEYS,
)

JPEG_FORMAT = "JPEG"
SAVE_OPTIONS = {JPEG_FORMAT: {"quality": JPEG_QUALITY, "subsampling": JPEG_SUBSAMPLING}}


class UnsupportedImage(Exception):
    pass


def clean_image(source: Path, target: Path) -> None:
    """rotate pixels upright and save without any metadata"""
    with Image.open(source) as image:
        image_format = image.format

        if image_format not in IMAGE_FORMATS:
            raise UnsupportedImage(source.name)

        upright = ImageOps.exif_transpose(image)
        upright.info.clear()
        upright.getexif().clear()
        upright.save(target, format=image_format, **SAVE_OPTIONS.get(image_format, {}))


def has_metadata(path: Path) -> bool:
    with Image.open(path) as image:
        text = getattr(image, "text", {})
        extra = any(key in image.info for key in METADATA_KEYS)

        return bool(image.getexif()) or bool(text) or extra
