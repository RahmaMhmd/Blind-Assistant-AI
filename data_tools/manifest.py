import csv
from pathlib import Path

from .constants import (
    BASE_COLUMNS,
    BATCH_COLUMN,
    FILE_COLUMN,
    GROUP_COLUMN,
    TEXT_ENCODING,
)
from .naming import parse_name
from .problems import Problem, ProblemCode
from .registry import DatasetSpec

Row = dict[str, str]


def columns_for(spec: DatasetSpec) -> list[str]:
    return [*BASE_COLUMNS, *spec.labels]


def read_manifest(path: Path) -> list[Row]:
    if not path.is_file():
        return []

    with path.open(newline="", encoding=TEXT_ENCODING) as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[Row], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding=TEXT_ENCODING) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def new_rows(spec: DatasetSpec, names: list[str]) -> list[Row]:
    rows = []

    for name in names:
        parts = parse_name(name)
        row = dict.fromkeys(columns_for(spec), "")
        row.update({FILE_COLUMN: name, GROUP_COLUMN: parts.group})
        row[BATCH_COLUMN] = parts.batch
        rows.append(row)

    return rows


def _label_problems(spec: DatasetSpec, row: Row) -> list[Problem]:
    problems = []

    for column, allowed in spec.labels.items():
        value = (row.get(column) or "").strip()
        subject = f"{row[FILE_COLUMN]}:{column}"

        if not value:
            problems.append(Problem(ProblemCode.MISSING_LABEL, subject))
        elif allowed is not None and value not in allowed:
            problems.append(Problem(ProblemCode.BAD_LABEL, subject))

    return problems


def manifest_problems(
    spec: DatasetSpec, rows: list[Row], names: set[str]
) -> list[Problem]:
    listed = [row[FILE_COLUMN] for row in rows]
    problems = [
        Problem(ProblemCode.NOT_IN_MANIFEST, n) for n in sorted(names - set(listed))
    ]
    problems += [
        Problem(ProblemCode.MISSING_FILE, n) for n in sorted(set(listed) - names)
    ]
    problems += [
        Problem(ProblemCode.DUPLICATE, f"manifest:{n}")
        for n in sorted({n for n in listed if listed.count(n) > 1})
    ]

    for row in rows:
        parts = parse_name(row[FILE_COLUMN])
        if parts is None or parts.group != row.get(GROUP_COLUMN):
            problems.append(Problem(ProblemCode.BAD_NAME, row[FILE_COLUMN]))
        problems += _label_problems(spec, row)

    return problems
