import pytest
pytest.importorskip("jiwer")
from src.evaluation.metrics.speech import CER, WER, speech_errors  
NO_ERRORS = 0.0
ONE_WORD_OF_FOUR = 0.25
ONE_CHAR_OF_THREE = 1 / 3
def test_identical_text_has_no_errors():
    result = speech_errors(["افتح الكاميرا"], ["افتح الكاميرا"])
    assert result[WER] == NO_ERRORS and result[CER] == NO_ERRORS
def test_one_wrong_word():
    assert speech_errors(["a b c d"], ["a b c x"])[WER] == ONE_WORD_OF_FOUR
def test_one_wrong_char():
    assert speech_errors(["abc"], ["abd"])[CER] == pytest.approx(ONE_CHAR_OF_THREE)
def test_spelling_variants_are_not_errors():
    assert speech_errors(["أحمد في المدرسة"], ["احمد في المدرسه"])[WER] == NO_ERRORS
def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        speech_errors(["؟"], ["شيء"])
def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        speech_errors(["a"], ["a", "b"])
