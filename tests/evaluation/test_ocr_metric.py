import pytest

pytest.importorskip("jiwer")

from src.evaluation.metrics.ocr import ocr_errors  # noqa: E402

NO_ERRORS = 0.0
ALL_WRONG = 1.0


def test_letter_spelling_variants_are_not_errors():
    assert ocr_errors(["مدرسة"], ["مدرسه"])["cer"] == NO_ERRORS


def test_punctuation_counts_for_text_read_from_images():
    assert ocr_errors(["كلمة)"], ["كلمة"])["wer"] == ALL_WRONG
