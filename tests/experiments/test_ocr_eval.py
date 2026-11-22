from experiments.ocr.cases import OcrCase
from experiments.ocr.compare import compare_ocr
from experiments.ocr.normalize import cer, normalize_text, wer


class StubReader:
    name = "stub-ocr"
    languages = frozenset({"ar", "en", "mixed"})

    def __init__(self, outputs: dict[bytes, str]):
        self._outputs = outputs

    def read_text(self, image: bytes) -> str:
        return self._outputs.get(image, "")


def test_normalize_unifies_arabic_forms_and_digits():
    raw = "إِجْمَالِي   الأَقْرَاصِ ١٥٠ جُنَيْهَة والتَّارِيخُ آمن"
    expected = "اجمالي الاقراص 150 جنيهه والتاريخ امن"

    assert normalize_text(raw) == expected


def test_cer_and_wer_are_zero_on_normalized_match():
    assert cer("صيدليةُ الشِّفَاء ١٢", "صيدليه الشفاء 12") == 0.0
    assert wer("صيدليةُ الشِّفَاء ١٢", "صيدليه الشفاء 12") == 0.0


def test_cer_and_wer_count_substitutions():
    assert wer("one two three", "one four three") == 1 / 3
    assert cer("abcd", "abxd") == 0.25


def test_compare_ocr_computes_cer_wer_and_empty_rate():
    cases = [
        OcrCase(b"1", "ar", "صيدلية الشفاء"),
        OcrCase(b"2", "en", "Pharmacy Open"),
    ]
    reader = StubReader({b"1": "صيدليه الشفاء", b"2": ""})
    stats = compare_ocr(reader, cases)

    assert stats.total == 2
    assert stats.mean_cer == 0.5
    assert stats.mean_wer == 0.5
    assert stats.empty_rate == 0.5
    assert stats.as_metrics()["stub-ocr_cer"] == 0.5
