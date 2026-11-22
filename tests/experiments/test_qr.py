import cv2
import numpy as np
import pytest

from experiments.qr.cases import Case
from experiments.qr.compare import compare
from experiments.qr.decoders import (
    BARCODE_KIND,
    QR_KIND,
    CvBarcodeDecoder,
    CvQrDecoder,
    ZxingDecoder,
)

VALUE = "https://example.com/item/42"
SCALE = 6
BORDER = 40
WHITE = 255


class FakeDetector:
    def __init__(self, result):
        self.result = result

    def detectAndDecode(self, gray):
        return self.result


def qr_image(value: str = VALUE) -> np.ndarray:
    raw = cv2.QRCodeEncoder.create().encode(value)
    big = cv2.resize(raw, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

    return cv2.copyMakeBorder(big, *[BORDER] * 4, cv2.BORDER_CONSTANT, value=WHITE)


def test_opencv_qr_reads_generated_code():
    assert CvQrDecoder().decode(qr_image()) == [VALUE]


def test_opencv_qr_returns_nothing_on_blank_image():
    blank = np.full((200, 200), WHITE, dtype=np.uint8)

    assert CvQrDecoder().decode(blank) == []


@pytest.mark.parametrize(
    "result",
    [
        ("123456", None, None),
        (("123456",), ("EAN_13",), None),
        (True, ("123456",), ("EAN_13",), None),
    ],
)
def test_barcode_adapter_handles_every_opencv_layout(result):
    decoder = CvBarcodeDecoder(FakeDetector(result))

    assert decoder.decode(qr_image()) == ["123456"]


def test_barcode_adapter_handles_missing_info():
    assert CvBarcodeDecoder(FakeDetector((None, None, None))).decode(qr_image()) == []
    assert CvBarcodeDecoder(FakeDetector(("", None, None))).decode(qr_image()) == []


def test_compare_counts_correct_wrong_and_missed():
    cases = [
        Case(qr_image(), QR_KIND, VALUE),
        Case(qr_image(), QR_KIND, "other value"),
        Case(np.full((200, 200), WHITE, dtype=np.uint8), QR_KIND, VALUE),
        Case(qr_image(), BARCODE_KIND, VALUE),
    ]
    stats = compare(CvQrDecoder(), cases)

    assert (stats.total, stats.correct, stats.wrong) == (3, 1, 1)
    assert stats.median_ms <= stats.p95_ms


def test_zxing_and_opencv_read_generated_codes():
    zxingcpp = pytest.importorskip("zxingcpp")
    qr = zxingcpp.create_barcode(VALUE, zxingcpp.BarcodeFormat.QRCode)
    qr_gray = np.array(zxingcpp.write_barcode_to_image(qr, scale=4))
    ean = zxingcpp.create_barcode("4006381333931", zxingcpp.BarcodeFormat.EAN13)
    ean_raw = np.array(zxingcpp.write_barcode_to_image(ean, scale=2))
    ean_gray = cv2.copyMakeBorder(
        ean_raw, *[BORDER] * 4, cv2.BORDER_CONSTANT, value=WHITE
    )

    assert ZxingDecoder().decode(qr_gray) == [VALUE]
    assert ZxingDecoder().decode(ean_gray) == ["4006381333931"]
    assert CvBarcodeDecoder().decode(ean_gray) == ["4006381333931"]
