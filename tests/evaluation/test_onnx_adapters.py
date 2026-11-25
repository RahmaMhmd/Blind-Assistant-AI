from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.adapters.onnx_classifier import OnnxClassifierAdapter
from src.evaluation.adapters.onnx_detector import OnnxDetectorAdapter


class FakeDetectorSession:
    def run(self, _outputs, _inputs):
        logits = np.array([[[6.0, -5.0], [-5.0, -5.0]]], dtype=np.float32)
        boxes = np.array([[[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]]], dtype=np.float32)
        return [logits, boxes]


class FakeClassifierSession:
    def run(self, _outputs, _inputs):
        return [np.array([[0.1, 4.2, 0.3]], dtype=np.float32)]


def test_onnx_detector_adapter_parses_boxes(tmp_path: Path):
    img = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img)
    adapter = OnnxDetectorAdapter("d", "object_detection", "unused.onnx", session=FakeDetectorSession())
    out = adapter.predict(img)
    assert len(out["boxes"]) == 1 and out["labels"] == [0]
    assert out["scores"][0] > 0.99


def test_onnx_classifier_adapter_maps_label(tmp_path: Path):
    img = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img)
    adapter = OnnxClassifierAdapter(
        "m", "image_classification", "unused.onnx", labels=["blur", "sharp", "dark"], session=FakeClassifierSession()
    )
    assert adapter.predict(img) == "sharp"
