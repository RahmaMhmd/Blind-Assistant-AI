from src.contracts import SupportedLanguage

ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
ARABIC_MAJORITY = 0.5


def _is_arabic(char: str) -> bool:
    return any(low <= ord(char) <= high for low, high in ARABIC_RANGES)


def detect_language(text: str) -> str:
    """language with the larger share of letters wins"""
    letters = [char for char in text if char.isalpha()]

    if not letters:
        return SupportedLanguage.ENGLISH.value

    share = sum(_is_arabic(char) for char in letters) / len(letters)

    if share >= ARABIC_MAJORITY:
        return SupportedLanguage.ARABIC.value

    return SupportedLanguage.ENGLISH.value
