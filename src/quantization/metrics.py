import numpy as np


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def decode_detections(logits: np.ndarray, boxes: np.ndarray, conf_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """convert raw detector outputs into filtered xyxy boxes"""
    l0 = np.asarray(logits[0], dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-np.clip(l0, -50.0, 50.0)))
    scores = probs.max(axis=-1)
    labels = probs.argmax(axis=-1)
    keep = scores >= conf_thresh
    xyxy = cxcywh_to_xyxy(np.asarray(boxes[0], dtype=np.float32)[keep])
    return xyxy, scores[keep], labels[keep]


def detection_agreement(fp32_dets: list[tuple], int8_dets: list[tuple], iou_thresh: float) -> float:
    """measure box match ratio between fp32 and int8 outputs"""
    matched, total = 0, 0
    for (b_ref, _, l_ref), (b_hyp, _, l_hyp) in zip(fp32_dets, int8_dets):
        total += len(b_ref)
        if len(b_ref) == 0 or len(b_hyp) == 0:
            continue
        ious = iou_matrix(b_ref, b_hyp)
        used: set[int] = set()
        for i in range(len(b_ref)):
            candidates = [j for j in range(len(b_hyp)) if j not in used and l_hyp[j] == l_ref[i] and ious[i, j] >= iou_thresh]
            if candidates:
                best = max(candidates, key=lambda j: ious[i, j])
                used.add(best)
                matched += 1
    return matched / total if total else 1.0


def classification_agreement(fp32_preds: list[int], int8_preds: list[int]) -> float:
    if not fp32_preds:
        return 0.0
    hits = sum(int(a == b) for a, b in zip(fp32_preds, int8_preds))
    return hits / len(fp32_preds)
