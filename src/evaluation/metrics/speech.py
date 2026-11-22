from collections.abc import Sequence
import jiwer
from ..normalization import ArabicNormalizer, get_normalizer
WER = "wer"
CER = "cer"
def speech_errors(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalizer: ArabicNormalizer | None = None,
) -> dict[str, float]:
    """Word and character error rates after the same normalization on both sides."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    normalizer = normalizer or get_normalizer()
    refs = [normalizer(text) for text in references]
    hyps = [normalizer(text) for text in hypotheses]

    for index, text in enumerate(refs):
        if not text:
            raise ValueError(f"reference {index} is empty after normalization")

    return {WER: jiwer.wer(refs, hyps), CER: jiwer.cer(refs, hyps)}
