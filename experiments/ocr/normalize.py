import re
import unicodedata

TATWEEL = "\u0640"
ALEF_VARIANTS = str.maketrans(
    {
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0671": "\u0627",
        "\u0629": "\u0647",
        "\u0649": "\u064A",
        "\u0660": "0",
        "\u0661": "1",
        "\u0662": "2",
        "\u0663": "3",
        "\u0664": "4",
        "\u0665": "5",
        "\u0666": "6",
        "\u0667": "7",
        "\u0668": "8",
        "\u0669": "9",
    }
)
SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """unify Arabic letter forms digits and spaces before scoring"""
    cleaned = "".join(
        ch
        for ch in unicodedata.normalize("NFKC", text)
        if unicodedata.category(ch) != "Mn" and ch != TATWEEL
    )
    mapped = cleaned.translate(ALEF_VARIANTS).lower()

    return SPACES.sub(" ", mapped).strip()


def edit_distance(expected: list[str], actual: list[str]) -> int:
    previous = list(range(len(actual) + 1))

    for index, left in enumerate(expected, start=1):
        current = [index]
        for pos, right in enumerate(actual, start=1):
            cost = 0 if left == right else 1
            current.append(
                min(previous[pos] + 1, current[-1] + 1, previous[pos - 1] + cost)
            )
        previous = current

    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference))
    hyp = list(normalize_text(hypothesis))

    return edit_distance(ref, hyp) / len(ref) if ref else (1.0 if hyp else 0.0)


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()

    return edit_distance(ref, hyp) / len(ref) if ref else (1.0 if hyp else 0.0)
