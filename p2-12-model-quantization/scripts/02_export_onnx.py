import os
import torch
from transformers import AutoModelForObjectDetection

# 1. Setup model ID and output path (Apache 2.0 licensed D-FINE small)
MODEL_ID = "ustc-community/dfine-small-coco"
OUTPUT_DIR = "models/onnx_fp32"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ONNX_PATH = os.path.join(OUTPUT_DIR, "dfine_small_fp32.onnx")

# 2. Load pretrained D-FINE small in evaluation mode
print(f"Loading {MODEL_ID} from Hugging Face...")
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
model.eval()

# 3. Create dummy input matching D-FINE standard resolution (Batch=1, Channels=3, H=640, W=640)
dummy_input = torch.randn(1, 3, 640, 640)

# 4. Export to a single self-contained ONNX file (FP32)
print(f"Exporting model to ONNX at: {ONNX_PATH} ...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"},
    },
    dynamo=False,
)

# 5. Print exported file size
size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"\nExport completed successfully! ONNX (FP32) file size: {size_mb:.2f} MB")