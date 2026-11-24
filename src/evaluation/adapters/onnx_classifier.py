from pathlib import Path

import numpy as np

from src.quantization.config import resolve_path
from src.quantization.images import load_tensor


class OnnxClassifierAdapter:
    """run an onnx classifier on one image"""

    def __init__(
        self,
        name: str,
        task: str,
        model_path: str,
        input_size: int = 224,
        input_name: str = "pixel_values",
        labels: list[str] | None = None,
        session=None,
    ):
        self.name = name
        self.task = task
        self._size = int(input_size)
        self._input_name = input_name
        self._labels = labels
        if session is not None:
            self._session = session
        else:
            import onnxruntime as ort

            path = resolve_path(model_path)
            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def predict(self, sample_path: Path) -> str:
        tensor = load_tensor(sample_path, self._size)
        logits = self._session.run(None, {self._input_name: tensor})[0]
        idx = int(np.argmax(logits[0]))
        return self._labels[idx] if self._labels and idx < len(self._labels) else str(idx)
