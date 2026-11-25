from collections.abc import Sequence

from ..normalization import get_normalizer


def keyword_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """compute wake word recall and false alarm rate"""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    norm = get_normalizer()
    pos_total, pos_hit, neg_total, false_alarm = 0, 0, 0, 0
    for ref, hyp in zip(references, hypotheses):
        target, pred = norm(str(ref)), norm(str(hyp))
        if target:
            pos_total += 1
            if target == pred:
                pos_hit += 1
        else:
            neg_total += 1
            if pred:
                false_alarm += 1
    recall = pos_hit / pos_total if pos_total else 1.0
    far = false_alarm / neg_total if neg_total else 0.0
    return {"recall": round(recall, 4), "false_alarm_rate": round(far, 4)}
