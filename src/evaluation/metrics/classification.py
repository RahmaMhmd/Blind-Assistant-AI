from collections.abc import Sequence

ACCURACY = "accuracy"


def _is_match(reference: object, hypothesis: object) -> bool:
    ref = str(reference).strip().lower()
    hyp = str(hypothesis).strip().lower()
    if ref == hyp:
        return True
    # handle comma-separated synsets (e.g. 'tench, tinca tinca' matching 'tench')
    synonyms = [s.strip() for s in hyp.split(",") if s.strip()]
    return ref in synonyms


def classification_metrics(
    references: Sequence[object],
    hypotheses: Sequence[object],
) -> dict[str, float]:
    """Calculate accuracy between predictions and ground truth."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        return {ACCURACY: 0.0}

    matches = sum(
        1.0 for ref, hyp in zip(references, hypotheses) if _is_match(ref, hyp)
    )
    return {ACCURACY: round(matches / len(references), 4)}
