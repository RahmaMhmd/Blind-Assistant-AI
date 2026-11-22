import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from data_tools.image_io import UnsupportedImage, clean_image, has_metadata
from tests.datatools.support import IMAGE_SIZE, make_image


def test_tagged_jpeg_reports_metadata(tmp_path):
    assert has_metadata(make_image(tmp_path / "a.jpg", tagged=True))


def test_plain_jpeg_reports_none(tmp_path):
    assert not has_metadata(make_image(tmp_path / "a.jpg"))


def test_clean_removes_metadata_and_rotates(tmp_path):
    source = make_image(tmp_path / "a.jpg", tagged=True)
    target = tmp_path / "clean.jpg"

    clean_image(source, target)

    with Image.open(target) as image:
        assert image.size == IMAGE_SIZE[::-1]
    assert not has_metadata(target)


def test_clean_removes_png_text(tmp_path):
    source = tmp_path / "a.png"
    meta = PngInfo()
    meta.add_text("Author", "someone")
    Image.new("RGB", IMAGE_SIZE).save(source, pnginfo=meta)
    target = tmp_path / "clean.png"

    assert has_metadata(source)
    clean_image(source, target)
    assert not has_metadata(target)


def test_clean_rejects_other_formats(tmp_path):
    source = tmp_path / "a.gif"
    Image.new("P", IMAGE_SIZE).save(source, format="GIF")

    with pytest.raises(UnsupportedImage):
        clean_image(source, tmp_path / "out.gif")


def test_clean_fails_on_non_image(tmp_path):
    source = tmp_path / "a.jpg"
    source.write_text("not an image")

    with pytest.raises(OSError):
        clean_image(source, tmp_path / "out.jpg")
