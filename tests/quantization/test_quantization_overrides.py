from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.quantization import benchmark as bm_mod
from src.quantization import export as ex_mod
from src.quantization import quantize as qz_mod


def test_quantize_uses_spec_calibration_dir(tmp_path: Path, monkeypatch):
    spec_calib = tmp_path / "custom_calib"
    spec_calib.mkdir()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(spec_calib / "img.jpg")
    fp32 = tmp_path / "m_fp32.onnx"
    fp32.write_bytes(b"fake")
    seen = {}

    class DummyQFormat:
        QDQ = "QDQ"
        QOperator = "QOperator"

    class DummyCMethod:
        MinMax = "MinMax"
        Entropy = "Entropy"
        Percentile = "Percentile"

    monkeypatch.setattr(qz_mod, "load_config", lambda: {
        "paths": {"fp32_dir": str(tmp_path), "int8_dir": str(tmp_path), "calibration_dir": str(tmp_path / "unused")},
        "calibration": {"count": 10},
        "models": {"m": {"task": "classification", "strategy": "conv_all", "quant_format": "qdq", "calibrate_method": "percentile", "calibration_dir": str(spec_calib), "input_size": 16, "input_name": "pixel_values"}},
    })
    monkeypatch.setattr(qz_mod, "select_conv_nodes", lambda g, s, t: [])
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=lambda p: SimpleNamespace(graph=None)))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", SimpleNamespace(
        CalibrationDataReader=object, CalibrationMethod=DummyCMethod, QuantFormat=DummyQFormat,
        QuantType=SimpleNamespace(QUInt8=1, QInt8=2),
        quantize_static=lambda **kw: seen.update(kw),
    ))
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization.shape_inference", SimpleNamespace(quant_pre_process=lambda s, p, **kw: None))
    qz_mod.quantize_int8("m", fp32_path=fp32, int8_path=tmp_path / "out.onnx")
    assert seen["calibration_data_reader"].get_next()["pixel_values"].shape == (1, 3, 16, 16)


def test_benchmark_uses_spec_benchmark_dir(tmp_path: Path, monkeypatch):
    bench_dir = tmp_path / "custom_bench"
    bench_dir.mkdir()
    Image.new("RGB", (16, 16)).save(bench_dir / "a.jpg")
    (tmp_path / "m_fp32.onnx").write_bytes(b"a")
    (tmp_path / "m_int8.onnx").write_bytes(b"b")
    monkeypatch.setattr(bm_mod, "load_config", lambda: {
        "paths": {"fp32_dir": str(tmp_path), "int8_dir": str(tmp_path), "calibration_dir": str(tmp_path / "unused")},
        "benchmark": {"max_images": 5, "warmup_runs": 0, "conf_threshold": 0.5, "iou_threshold": 0.5},
        "models": {"m": {"task": "classification", "benchmark_dir": str(bench_dir), "input_size": 16, "input_name": "pixel_values"}},
    })
    monkeypatch.setattr(bm_mod, "_run_session", lambda path, tensors, name, warmup: ([[np.array([[1.0, 0.0]])] for _ in tensors], [1.0]))
    summary = bm_mod.benchmark_model("m")
    assert summary["samples"] == 1 and summary["agreement"] == 1.0


def test_export_loads_num_classes_and_checkpoint(tmp_path: Path, monkeypatch):
    calls = {}

    class DummyModel:
        def eval(self):
            calls["eval"] = True

        def load_state_dict(self, state):
            calls["loaded"] = state

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=lambda src, pretrained, num_classes=1000: calls.update({"num_classes": num_classes}) or DummyModel()))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda p, map_location="cpu": {"w": 1}))
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(b"x")
    ex_mod._load_source_model({"task": "classification", "source": "s", "num_classes": 6, "checkpoint": str(ckpt)})
    assert calls["num_classes"] == 6 and calls["loaded"] == {"w": 1} and calls["eval"]
