from typing import Protocol

import cv2
import numpy as np

QR_KIND = "qr"
BARCODE_KIND = "barcode"
BARCODE_INFO_INDEX = -3


class Decoder(Protocol):
    name: str
    kinds: frozenset[str]

    def decode(self, gray: np.ndarray) -> list[str]: ...


class CvQrDecoder:
    name = "opencv-qrcode-detector"
    kinds = frozenset({QR_KIND})

    def __init__(self, detector=None):
        self._detector = detector or cv2.QRCodeDetector()

    def decode(self, gray: np.ndarray) -> list[str]:
        try:
            value = self._detector.detectAndDecode(gray)[0]
        except UnicodeDecodeError:
            return []

        return [value] if value else []


class CvBarcodeDecoder:
    name = "opencv-barcode-detector"
    kinds = frozenset({BARCODE_KIND})

    def __init__(self, detector=None):
        self._detector = detector or cv2.barcode_BarcodeDetector()

    def decode(self, gray: np.ndarray) -> list[str]:
        """the info slot is third from the end in every opencv layout"""
        info = self._detector.detectAndDecode(gray)[BARCODE_INFO_INDEX]
        items = (info,) if isinstance(info, str) else info or ()

        return [value for value in items if value]


class ZxingDecoder:
    name = "zxing-cpp"
    kinds = frozenset({QR_KIND, BARCODE_KIND})

    def __init__(self):
        import zxingcpp

        self._read = zxingcpp.read_barcodes

    def decode(self, gray: np.ndarray) -> list[str]:
        return [result.text for result in self._read(gray) if result.text]


def default_decoders() -> list[Decoder]:
    return [CvQrDecoder(), CvBarcodeDecoder(), ZxingDecoder()]
