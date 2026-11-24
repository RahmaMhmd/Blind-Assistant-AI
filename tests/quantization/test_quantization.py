from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.quantization.__main__ import main
from src.quantization.benchmark import log_benchmark
from src.quantization.config import load_config, model_spec
from src.quantization.images import list_images, load_tensor
from src.quantization.metrics import (
    classification_agreement,
    cxcywh_to_xyxy,
    decode_detections,
    detection_agreement,
    iou_matrix,
)
from src.quantization.nodes import select_conv_nodes


class StubMlflow:
    def __init__(self):
        self.metrics = None

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self, run_name):
        yield SimpleNamespace(info=SimpleNamespace(run_id="r1"))

    def set_tag(self, key, value):
        self.tag = (key, value)

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics):
        self.metrics = metrics


def _node(name: str, group: int = 1, kernel: list[int] | None = None, op: str = "Conv"):
    attrs = [SimpleNamespace(name="group", i=group)]
    if kernel is not None:
        attrs.append(SimpleNamespace(name="kernel_shape", ints=kernel))
    return SimpleNamespace(name=name, op_type=op, attribute=attrs)


def test_config_contains_all_quantized_models():
    cfg = load_config()
    assert set(cfg["models"]) == {"rtdetrv2_r18", "dfine_small", "mobilenetv3_large"}
    assert model_spec("dfine_small", cfg)["strategy"] == "dfine_mixed"


def test_load_tensor_shape_and_range(tmp_path: Path):
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color=(128, 64, 255)).save(img_path)
    tensor = load_tensor(img_path, 224)
    assert tensor.shape == (1, 3, 224, 224) and tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    norm = load_tensor(img_path, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert -1.0 <= float(norm.min()) <= float(norm.max()) <= 1.0
    assert list_images(tmp_path) == [img_path]


def test_select_conv_nodes_strategies():
    graph = SimpleNamespace(
        node=[
            _node("/backbone/stages.0/conv1", 1, [3, 3]),
            _node("/backbone/stages.2/pw", 1, [1, 1]),
            _node("/backbone/stages.2/dw", 4, [1, 1]),
            _node("/backbone/stages.3/conv1", 1, [3, 3]),
            _node("/backbone/stages.3/conv2", 1, [3, 3]),
            _node("/decoder/attn", 1, [1, 1], op="MatMul"),
        ]
    )
    assert len(select_conv_nodes(graph, "conv_all")) == 5
    assert select_conv_nodes(graph, "backbone_tail", 2) == ["/backbone/stages.3/conv1", "/backbone/stages.3/conv2"]
    assert select_conv_nodes(graph, "dfine_mixed") == ["/backbone/stages.3/conv1", "/backbone/stages.2/pw"]
def test_detection_and_classification_metrics():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    logits = np.array([[[5.0, -5.0]]], dtype=np.float32)
    xyxy, scores, labels = decode_detections(logits, np.expand_dims(boxes, 0), 0.5)
    assert xyxy.shape == (1, 4) and labels.tolist() == [0] and scores[0] > 0.99
    assert iou_matrix(cxcywh_to_xyxy(boxes), xyxy)[0, 0] == pytest.approx(1.0)
    assert detection_agreement([(xyxy, scores, labels)], [(xyxy, scores, labels)], 0.5) == 1.0
    assert classification_agreement([1, 2, 3], [1, 0, 3]) == pytest.approx(2 / 3)


def test_log_benchmark_and_cli_error():
    stub = StubMlflow()
    assert log_benchmark({"model": "dfine_small", "task": "detection", "samples": 2, "agreement": 0.99}, mlflow=stub) == "r1"
    assert stub.metrics == {"agreement": 0.99}
    assert main(["benchmark", "--model", "missing_model"]) == 1
