from collections.abc import Sequence
from ..normalization import get_normalizer
from .speech import speech_errors
OCR_PROFILE = "ocr"

def ocr_errors(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    """Word and character error rates, with the text rules chosen for reading text from images."""
    return speech_errors(references, hypotheses, get_normalizer(OCR_PROFILE))
