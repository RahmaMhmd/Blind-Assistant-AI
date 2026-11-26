from .constants import CARD_FENCE, CARD_REQUIRED_FIELDS, CARD_SEPARATOR
from .problems import Problem, ProblemCode
from .registry import DatasetSpec

DATASET_FIELD = "dataset"


def parse_card(text: str) -> dict[str, str]:
    """read the header block of a data card"""
    lines = text.splitlines()
    fields: dict[str, str] = {}

    if not lines or lines[0].strip() != CARD_FENCE:
        return fields

    for line in lines[1:]:
        if line.strip() == CARD_FENCE:
            break
        key, _, value = line.partition(CARD_SEPARATOR)
        fields[key.strip()] = value.strip()

    return fields


def card_problems(name: str, text: str | None) -> list[Problem]:
    if text is None:
        return [Problem(ProblemCode.CARD_MISSING, name)]

    fields = parse_card(text)
    missing = [f for f in CARD_REQUIRED_FIELDS if not fields.get(f)]

    if fields.get(DATASET_FIELD) not in (None, "", name):
        missing.append(DATASET_FIELD)

    return [
        Problem(ProblemCode.CARD_INCOMPLETE, f"{name}:{field}") for field in missing
    ]


def _default(spec: DatasetSpec, field: str) -> str:
    return spec.name if field == DATASET_FIELD else ""


def starter_card(spec: DatasetSpec) -> str:
    header = [CARD_FENCE]
    header += [
        f"{field}{CARD_SEPARATOR} {_default(spec, field)}".rstrip()
        for field in CARD_REQUIRED_FIELDS
    ]
    header.append(CARD_FENCE)
    labels = [f"- {column}" for column in spec.labels]

    return "\n".join([*header, "", "## Labels", *labels, "", "## Known limits", ""])
