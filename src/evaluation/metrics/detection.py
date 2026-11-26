from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from src.quantization.metrics import iou_matrix

THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _ap_for_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def _class_ap(refs: Sequence[dict], hyps: Sequence[dict], cls_id: int, iou_thr: float) -> float | None:
    gt_by_img: dict[int, np.ndarray] = {}
    npos = 0
    for img_idx, ref in enumerate(refs):
        boxes = [b for b, l in zip(ref.get("boxes", []), ref.get("labels", [])) if int(l) == cls_id]
        npos += len(boxes)
        gt_by_img[img_idx] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if npos == 0:
        return None
    preds = []
    for img_idx, hyp in enumerate(hyps):
        for box, score, label in zip(hyp.get("boxes", []), hyp.get("scores", []), hyp.get("labels", [])):
            if int(label) == cls_id:
                preds.append((float(score), img_idx, np.asarray(box, dtype=np.float32)))
    if not preds:
        return 0.0
    preds.sort(key=lambda item: item[0], reverse=True)
    used: dict[int, set[int]] = defaultdict(set)
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)
    for idx, (_, img_idx, box) in enumerate(preds):
        gt_boxes = gt_by_img[img_idx]
        if gt_boxes.shape[0] == 0:
            fp[idx] = 1.0
            continue
        ious = iou_matrix(box.reshape(1, 4), gt_boxes)[0]
        best_gt = int(np.argmax(ious))
        if ious[best_gt] >= iou_thr and best_gt not in used[img_idx]:
            tp[idx] = 1.0
            used[img_idx].add(best_gt)
        else:
            fp[idx] = 1.0
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    return _ap_for_curve(tp_cum / npos, tp_cum / np.maximum(tp_cum + fp_cum, 1e-9))


def detection_metrics(references: Sequence[dict], hypotheses: Sequence[dict]) -> dict[str, float]:
    """compute coco style map50 and map50 95"""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    classes = sorted({int(lbl) for ref in references for lbl in ref.get("labels", [])})
    if not classes:
        return {"map50": 0.0, "map50_95": 0.0}
    aps_per_thr = []
    for thr in THRESHOLDS:
        aps = [ap for c in classes if (ap := _class_ap(references, hypotheses, c, thr)) is not None]
        aps_per_thr.append(float(np.mean(aps)) if aps else 0.0)
    return {"map50": round(aps_per_thr[0], 4), "map50_95": round(float(np.mean(aps_per_thr)), 4)}
