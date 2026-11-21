import os
import glob
import onnx
import numpy as np
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# 1. Configuration: switch between "dfine_small" and "rtdetrv2_r18"
MODEL_NAME = "rtdetrv2_r18"
FP32_MODEL_PATH = f"models/onnx_fp32/{MODEL_NAME}_fp32.onnx"
PREPROCESSED_MODEL_PATH = f"models/onnx_fp32/{MODEL_NAME}_prep.onnx"

INT8_DIR = os.environ.get("OUT_DIR", "models/onnx_int8")
os.makedirs(INT8_DIR, exist_ok=True)
INT8_MODEL_PATH = f"{INT8_DIR}/{MODEL_NAME}_int8.onnx"
CALIB_DIR = "data/calibration"


# 2. Calibration Data Reader (640x640, [0, 1] normalized RGB)
class ObjectDetectorCalibrationReader(CalibrationDataReader):
    def __init__(self, image_folder, input_name="pixel_values"):
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        self.input_name = input_name
        self.index = 0

    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((640, 640), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        img_path = self.image_paths[self.index]
        self.index += 1
        return {self.input_name: self.preprocess_image(img_path)}


# 3. Pre-process FP32 ONNX graph
print(f"Pre-processing FP32 model: {FP32_MODEL_PATH} ...")
quant_pre_process(FP32_MODEL_PATH, PREPROCESSED_MODEL_PATH, skip_symbolic_shape=True)

# 4. Architecture-aware Mixed-Precision Node Selection
model_onnx = onnx.load(PREPROCESSED_MODEL_PATH)
conv_nodes_to_quantize = []

if MODEL_NAME == "dfine_small":
    # Select deep stages.3 standard Convs ([:-1]) + fast 1x1 pointwise Convs in stages.2 (14 nodes)
    stage3_convs = []
    stage2_1x1_convs = []
    for node in model_onnx.graph.node:
        if node.op_type != "Conv":
            continue
        name_lower = node.name.lower()
        attrs = {a.name: a for a in node.attribute}
        group = attrs["group"].i if "group" in attrs else 1
        kernel = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else []

        if "/backbone/" in name_lower and group == 1:
            if "stages.3" in name_lower:
                stage3_convs.append(node.name)
            elif "stages.2" in name_lower and kernel == [1, 1]:
                stage2_1x1_convs.append(node.name)
    conv_nodes_to_quantize = stage3_convs[:-1] + stage2_1x1_convs
else:
    # For RT-DETRv2 R18: keep early embedder/stage-0 stem in FP32, quantize the 13 deep backbone Convs
    backbone_convs = [
        node.name
        for node in model_onnx.graph.node
        if node.op_type == "Conv" and "/backbone/" in node.name.lower()
    ]
    conv_nodes_to_quantize = backbone_convs[-13:]

print(f"Selected {len(conv_nodes_to_quantize)} Conv nodes for INT8 quantization ({MODEL_NAME}).")

# 5. Fast MinMax Static INT8 Quantization
print("Running Safe Mixed-Precision Static INT8 Quantization...")
calibration_reader = ObjectDetectorCalibrationReader(CALIB_DIR, input_name="pixel_values")

quantize_static(
    model_input=PREPROCESSED_MODEL_PATH,
    model_output=INT8_MODEL_PATH,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QOperator,
    nodes_to_quantize=conv_nodes_to_quantize,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    extra_options={"EnableSubgraph": True},
)

# Clean up temporary preprocessed files
for tmp_path in (PREPROCESSED_MODEL_PATH, f"models/onnx_fp32/{MODEL_NAME}_prep_augmented.onnx"):
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

# 6. Report final size reduction
fp32_size = os.path.getsize(FP32_MODEL_PATH) / (1024 * 1024)
int8_size = os.path.getsize(INT8_MODEL_PATH) / (1024 * 1024)
reduction = (1 - (int8_size / fp32_size)) * 100

print("\nQuantization completed successfully!")
print(f"Saved to: {INT8_MODEL_PATH}")
print(f"FP32 Size: {fp32_size:.2f} MB")
print(f"INT8 Size: {int8_size:.2f} MB (Reduced by {reduction:.1f}%)")