from pathlib import Path

from .config import load_config, model_spec, resolve_path
from .images import list_images, load_tensor
from .nodes import select_conv_nodes


def _build_reader(
    image_paths: list[Path],
    size: int,
    input_name: str,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._idx = 0

        def get_next(self):
            if self._idx >= len(image_paths):
                return None
            tensor = load_tensor(image_paths[self._idx], size, mean=mean, std=std)
            self._idx += 1
            return {input_name: tensor}

    return Reader()


def quantize_int8(
    model_name: str,
    fp32_path: Path | None = None,
    int8_path: Path | None = None,
    calib_dir: Path | None = None,
) -> Path:
    """run static mixed precision int8 quantization"""
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg = load_config()
    spec = model_spec(model_name, cfg)
    paths = cfg["paths"]

    src = Path(fp32_path) if fp32_path else resolve_path(paths["fp32_dir"]) / f"{model_name}_fp32.onnx"
    prep = src.with_name(f"{model_name}_prep.onnx")
    out = Path(int8_path) if int8_path else resolve_path(paths["int8_dir"]) / f"{model_name}_int8.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    cdir = Path(calib_dir) if calib_dir else resolve_path(paths["calibration_dir"])

    quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
    graph = onnx.load(str(prep)).graph
    nodes = select_conv_nodes(graph, spec["strategy"], int(spec.get("tail_count", 13)))
    imgs = list_images(cdir, int(cfg["calibration"]["count"]))
    if not imgs:
        raise FileNotFoundError(f"no calibration images in {cdir}")

    fmt_map = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}
    cal_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    fmt = fmt_map[spec.get("quant_format", "qoperator").lower()]
    cal = cal_map[spec.get("calibrate_method", "minmax").lower()]
    reader = _build_reader(imgs, int(spec["input_size"]), spec["input_name"], spec.get("mean"), spec.get("std"))

    quantize_static(
        model_input=str(prep),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=fmt,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None if fmt == QuantFormat.QDQ else nodes,
        per_channel=True,
        calibrate_method=cal,
    )
    prep.unlink(missing_ok=True)
    return out
