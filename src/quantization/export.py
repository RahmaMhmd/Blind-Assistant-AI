from pathlib import Path

from .config import load_config, model_spec, resolve_path


def _load_source_model(spec: dict):
    if spec["task"] == "detection":
        from transformers import AutoModelForObjectDetection

        model = AutoModelForObjectDetection.from_pretrained(spec["source"])
    elif spec["task"] == "classification":
        import timm
        import torch

        model = timm.create_model(
            spec["source"], pretrained=True, num_classes=int(spec.get("num_classes", 1000))
        )
        if spec.get("checkpoint"):
            state = torch.load(resolve_path(spec["checkpoint"]), map_location="cpu")
            model.load_state_dict(state)
    else:
        raise ValueError(f"unsupported task: {spec['task']}")
    model.eval()
    return model


def export_fp32(model_name: str, output_dir: Path | None = None) -> Path:
    """export model weights to fp32 onnx file"""
    import torch

    cfg = load_config()
    spec = model_spec(model_name, cfg)
    target_dir = Path(output_dir) if output_dir else resolve_path(cfg["paths"]["fp32_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{model_name}_fp32.onnx"

    size = int(spec["input_size"])
    in_name = spec["input_name"]
    out_names = list(spec["outputs"])
    axes = {name: {0: "batch_size"} for name in [in_name, *out_names]}
    model = _load_source_model(spec)
    dummy = torch.randn(1, 3, size, size)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(spec["opset"]),
        do_constant_folding=True,
        input_names=[in_name],
        output_names=out_names,
        dynamic_axes=axes,
        dynamo=False,
    )
    return out_path
