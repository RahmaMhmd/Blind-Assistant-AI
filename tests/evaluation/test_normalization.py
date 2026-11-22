import pytest
from src.evaluation.normalization import get_normalizer
OCR_PROFILE = "ocr"
CASES = [
    ("أحمد إبراهيم آمن", "احمد ابراهيم امن"),
    ("مدرسة", "مدرسه"),
    ("على", "علي"),
    ("مسؤول", "مسءول"),
    ("مسئول", "مسءول"),
    ("مَدْرَسَـة", "مدرسه"),
    ("٣ جنيه", "3 جنيه"),
    ("مرحبا، كيف حالك؟", "مرحبا كيف حالك"),
    ("Hello   WORLD", "hello world"),
    ("", ""),
    ("3.5 جنيه", "3.5 جنيه"),
    ("٣٫٥ جنيه", "3.5 جنيه"),
    ("انتهى.", "انتهي"),
    ("نهاية.5", "نهايه 5"),
]
@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_default_profile(raw, expected):
    assert get_normalizer()(raw) == expected
def test_ocr_profile_keeps_punctuation_but_still_unifies_letters():
    assert get_normalizer(OCR_PROFILE)("مدرسة، كبيرة؟") == "مدرسه، كبيره؟"
def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        get_normalizer("not-a-profile")
