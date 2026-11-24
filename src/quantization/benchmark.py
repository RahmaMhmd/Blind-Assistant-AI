from pathlib import Path
from statistics import mean
import time

import numpy as np

from .config import load_config, model_spec, resolve_path
from .images import list_images, load_tensor
from .metrics import classification_agreement, decode_detections, detection_agreement


def _run_session(model_path: Path, tensors: list[np.ndarray], input_name: str, warmup: int) -> tuple[list, list[float]]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    for sample in tensors[:warmup]:
        sess.run(None, {input_name: sample})
    outputs, times_ms = [], []
    for sample in tensors:
        t0 = time.perf_counter()
        outputs.append(sess.run(None, {input_name: sample}))
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return outputs, times_ms


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def log_benchmark(summary: dict, mlflow=None) -> str:
    """save quantization results in mlflow"""
    if mlflow is None:
        import mlflow

    settings = load_config()["tracking"]
    db_uri = f"sqlite:///{resolve_path(settings['database']).as_posix()}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment(settings["experiment"])
    with mlflow.start_run(run_name=f"quantize-{summary['model']}") as run:
        mlflow.set_tag("run_type", settings["run_type"])
        mlflow.log_params({"model": summary["model"], "task": summary["task"], "samples": summary["samples"]})
        metrics = {k: float(v) for k, v in summary.items() if isinstance(v, (int, float)) and k != "samples"}
        mlflow.log_metrics(metrics)
        return run.info.run_id


def benchmark_model(model_name: str, calib_dir: Path | None = None, track: bool = False) -> dict:
    cfg = load_config()
    spec = model_spec(model_name, cfg)
    bcfg, pcfg = cfg["benchmark"], cfg["paths"]

    fp32_path = resolve_path(pcfg["fp32_dir"]) / f"{model_name}_fp32.onnx"
    int8_path = resolve_path(pcfg["int8_dir"]) / f"{model_name}_int8.onnx"
    img_dir = Path(calib_dir) if calib_dir else resolve_path(pcfg["calibration_dir"])
    img_paths = list_images(img_dir, int(bcfg["max_images"]))
    if not img_paths:
        raise FileNotFoundError(f"no benchmark images in {img_dir}")

    tensors = [load_tensor(p, int(spec["input_size"])) for p in img_paths]
    fp32_out, fp32_ms = _run_session(fp32_path, tensors, spec["input_name"], int(bcfg["warmup_runs"]))
    int8_out, int8_ms = _run_session(int8_path, tensors, spec["input_name"], int(bcfg["warmup_runs"]))

    if spec["task"] == "detection":
        conf = float(bcfg["conf_threshold"])
        fp32_dec = [decode_detections(o[0], o[1], conf) for o in fp32_out]
        int8_dec = [decode_detections(o[0], o[1], conf) for o in int8_out]
        agreement = detection_agreement(fp32_dec, int8_dec, float(bcfg["iou_threshold"]))
    else:
        fp32_cls = [int(np.argmax(o[0][0])) for o in fp32_out]
        int8_cls = [int(np.argmax(o[0][0])) for o in int8_out]
        agreement = classification_agreement(fp32_cls, int8_cls)

    fp32_mb, int8_mb = _size_mb(fp32_path), _size_mb(int8_path)
    mean_fp32, mean_int8 = mean(fp32_ms), mean(int8_ms)
    summary = {
        "model": model_name,
        "task": spec["task"],
        "samples": len(tensors),
        "fp32_size_mb": fp32_mb,
        "int8_size_mb": int8_mb,
        "size_reduction_pct": round((1.0 - int8_mb / fp32_mb) * 100.0, 2) if fp32_mb else 0.0,
        "fp32_mean_ms": round(mean_fp32, 2),
        "int8_mean_ms": round(mean_int8, 2),
        "speedup": round(mean_fp32 / mean_int8, 2) if mean_int8 else 0.0,
        "agreement": round(agreement, 4),
    }
    if track:
        summary["mlflow_run_id"] = log_benchmark(summary)
    return summary
