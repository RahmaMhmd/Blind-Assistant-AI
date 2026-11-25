from pathlib import Path

from src.quantization.config import resolve_path
from src.quantization.images import load_tensor
from src.quantization.metrics import decode_detections


class OnnxDetectorAdapter:
    """run an onnx object detector on one image"""

    def __init__(
        self,
        name: str,
        task: str,
        model_path: str,
        input_size: int = 640,
        input_name: str = "pixel_values",
        conf_threshold: float = 0.5,
        session=None,
    ):
        self.name = name
        self.task = task
        self._size = int(input_size)
        self._input_name = input_name
        self._conf = float(conf_threshold)
        if session is not None:
            self._session = session
        else:
            import onnxruntime as ort

            path = resolve_path(model_path)
            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def predict(self, sample_path: Path) -> dict:
        tensor = load_tensor(sample_path, self._size)
        logits, boxes = self._session.run(None, {self._input_name: tensor})[:2]
        xyxy, scores, labels = decode_detections(logits, boxes, self._conf)
        return {
            "boxes": xyxy.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }
