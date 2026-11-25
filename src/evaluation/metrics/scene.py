from collections import Counter
from collections.abc import Sequence

from ..normalization import get_normalizer


def _token_f1(ref: str, hyp: str) -> float:
    ref_tokens, hyp_tokens = ref.split(), hyp.split()
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    overlap = sum((Counter(ref_tokens) & Counter(hyp_tokens)).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(hyp_tokens)
    rec = overlap / len(ref_tokens)
    return 2.0 * prec * rec / (prec + rec)


def scene_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """compute token f1 and exact match for scene captions"""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        return {"token_f1": 0.0, "exact_match": 0.0}
    norm = get_normalizer()
    f1_sum, exact = 0.0, 0
    for ref, hyp in zip(references, hypotheses):
        r, h = norm(str(ref)), norm(str(hyp))
        f1_sum += _token_f1(r, h)
        exact += int(r == h)
    total = len(references)
    return {"token_f1": round(f1_sum / total, 4), "exact_match": round(exact / total, 4)}
