from .adapters.base import Adapter
from .adapters.camelbert import CamelBertAdapter
from .adapters.fake import FakeAdapter
from .adapters.keyword_spotter import KeywordSpotterAdapter
from .adapters.mobilenet import MobileNetAdapter
from .adapters.onnx_classifier import OnnxClassifierAdapter
from .adapters.onnx_detector import OnnxDetectorAdapter
from .adapters.paddle_recognizer import PaddleRecognizerAdapter
from .adapters.scene_vlm import SceneVlmAdapter
from .adapters.whisper import WhisperAdapter
from .config import load_config

ADAPTERS = {
    "fake": FakeAdapter,
    "whisper": WhisperAdapter,
    "paddle_recognizer": PaddleRecognizerAdapter,
    "mobilenet": MobileNetAdapter,
    "camelbert": CamelBertAdapter,
    "onnx_detector": OnnxDetectorAdapter,
    "onnx_classifier": OnnxClassifierAdapter,
    "keyword_spotter": KeywordSpotterAdapter,
    "scene_vlm": SceneVlmAdapter,
}


def build_adapter(model_name: str) -> Adapter:
    models = load_config()["models"]

    if model_name not in models:
        raise ValueError(f"unknown model '{model_name}', available: {sorted(models)}")

    spec = models[model_name]

    if spec["adapter"] not in ADAPTERS:
        raise ValueError(f"unknown adapter '{spec['adapter']}' for '{model_name}'")

    adapter_class = ADAPTERS[spec["adapter"]]
    return adapter_class(model_name, spec["task"], **spec.get("params", {}))
