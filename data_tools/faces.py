import os
from pathlib import Path
from typing import Protocol

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2

from .constants import (
    FACE_INITIAL_SIZE,
    FACE_NMS_THRESHOLD,
    FACE_SCORE_THRESHOLD,
    FACE_TOP_K,
)


class FaceDetector(Protocol):
    def count_faces(self, path: Path) -> int: ...


class YuNetDetector:
    """count faces with the yunet onnx model"""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            FACE_INITIAL_SIZE,
            FACE_SCORE_THRESHOLD,
            FACE_NMS_THRESHOLD,
            FACE_TOP_K,
        )

    def count_faces(self, path: Path) -> int:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)

        return 0 if faces is None else len(faces)
