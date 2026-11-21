import os
import glob
import time
import numpy as np
from PIL import Image
import onnxruntime as ort

# 1. Configuration: benchmarks all finalized models in one run
MODELS = ["rtdetrv2_r18", "dfine_small"]
CALIB_DIR = "data/calibration"
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


# 2. Preload calibration images (640x640, [0, 1] normalized RGB)
def load_images(image_folder, max_images=200):
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))[:max_images]
    print(f"Preloading {len(image_paths)} images (640x640) for benchmarking...")
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((640, 640), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensors.append(np.expand_dims(arr, axis=0))
    return tensors


# 3. Helper functions for box conversion and IoU calculation
def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def compute_iou_matrix(boxes1, boxes2):
    area1 = np.maximum(0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0, boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter_area + 1e-7
    return inter_area / union


def extract_detections(logits, pred_boxes, conf_thresh=CONF_THRESHOLD):
    probs = 1.0 / (1.0 + np.exp(-logits[0]))  # Sigmoid
    scores = np.max(probs, axis=-1)
    labels = np.argmax(probs, axis=-1)
    keep = scores >= conf_thresh
    boxes_xyxy = cxcywh_to_xyxy(pred_boxes[0][keep])
    return boxes_xyxy, scores[keep], labels[keep]


# 4. Run inference benchmark on CPU
def benchmark_session(model_path, tensors, label, conf_thresh=CONF_THRESHOLD):
    print(f"Benchmarking {label} ({model_path})...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warm-up runs
    for i in range(min(5, len(tensors))):
        session.run(None, {input_name: tensors[i]})

    latencies = []
    outputs_all = []
    for tensor in tensors:
        start = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
        outputs_all.append(extract_detections(outputs[0], outputs[1], conf_thresh=conf_thresh))

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency
    return avg_latency, fps, outputs_all


# 5. Compare FP32 vs INT8 detections
def evaluate_consistency(fp32_dets, int8_dets):
    ious = []
    matched_fp32 = 0
    total_fp32 = 0

    for (b1, s1, l1), (b2, s2, l2) in zip(fp32_dets, int8_dets):
        total_fp32 += len(b1)
        if len(b1) == 0 or len(b2) == 0:
            continue

        iou_mat = compute_iou_matrix(b1, b2)
        same_class = l1[:, None] == l2[None, :]
        valid_iou = np.where(same_class, iou_mat, 0.0)

        best_ious = np.max(valid_iou, axis=1)
        for iou_val in best_ious:
            if iou_val >= IOU_THRESHOLD:
                matched_fp32 += 1
                ious.append(iou_val)

    mean_iou = (float(np.mean(ious)) * 100.0) if ious else 0.0
    ap50_consistency = (matched_fp32 / max(1, total_fp32)) * 100.0
    est_map_drop = max(0.0, 100.0 - ap50_consistency)
    return mean_iou, ap50_consistency, est_map_drop


if __name__ == "__main__":
    int8_dir = os.environ.get("OUT_DIR", "models/onnx_int8")
    tensors = load_images(CALIB_DIR, max_images=200)

    for model_name in MODELS:
        fp32_path = f"models/onnx_fp32/{model_name}_fp32.onnx"
        int8_path = f"{int8_dir}/{model_name}_int8.onnx"

        if not (os.path.exists(fp32_path) and os.path.exists(int8_path)):
            continue

        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)

        fp32_lat, fp32_fps, fp32_dets = benchmark_session(
            fp32_path, tensors, f"FP32 {model_name}", conf_thresh=CONF_THRESHOLD
        )
        int8_lat, int8_fps, int8_dets = benchmark_session(
            int8_path, tensors, f"INT8 {model_name}", conf_thresh=CONF_THRESHOLD * 0.90
        )

        speedup = fp32_lat / int8_lat
        mean_iou, ap50_cons, map_drop = evaluate_consistency(fp32_dets, int8_dets)

        print("\n" + "=" * 68)
        print(f"        {model_name.upper()}: FP32 vs INT8 Final Benchmark")
        print("=" * 68)
        print(f"{'Metric':<30} | {'FP32':<15} | {'INT8':<15}")
        print("-" * 68)
        print(f"{'Model Size (MB)':<30} | {fp32_size:<15.2f} | {int8_size:<15.2f}")
        print(f"{'Avg Latency (ms)':<30} | {fp32_lat:<15.2f} | {int8_lat:<15.2f}")
        print(f"{'Throughput (FPS)':<30} | {fp32_fps:<15.1f} | {int8_fps:<15.1f}")
        print("-" * 68)
        print(f"Speedup:                   {speedup:.2f}x")
        print(f"Mean Box IoU Overlap:      {mean_iou:.2f}%")
        print(f"AP@50 Consistency:         {ap50_cons:.2f}%")
        print(f"Estimated mAP Drop:        {map_drop:.2f}% (Target: <= 2.00%)")
        print("=" * 68)