import cv2
import numpy as np

SIDE = 400
SEED = 7
BLUR_KERNEL = (31, 31)
DIM_FACTOR = 0.1
JPEG_EXT = ".jpg"


def noisy_image() -> np.ndarray:
    rng = np.random.default_rng(SEED)

    return rng.integers(0, 256, (SIDE, SIDE), dtype=np.uint8)


def blurred_image() -> np.ndarray:
    return cv2.GaussianBlur(noisy_image(), BLUR_KERNEL, 0)


def dark_image() -> np.ndarray:
    return (noisy_image() * DIM_FACTOR).astype(np.uint8)


def encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(JPEG_EXT, image)
    assert ok

    return buffer.tobytes()
