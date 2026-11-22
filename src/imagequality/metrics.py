import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def decode_gray(data: bytes) -> np.ndarray:
    gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ImageDecodeError("image bytes could not be decoded")

    return gray


def normalize(gray: np.ndarray, long_side: int) -> np.ndarray:
    """shrink to a fixed long side so thresholds do not depend on camera size"""
    scale = long_side / max(gray.shape)

    if scale >= 1:
        return gray

    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())
