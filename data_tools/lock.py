import json
from pathlib import Path

from .constants import (
    FILE_COLUMN,
    GROUP_COLUMN,
    LOCK_ENTRIES_KEY,
    LOCK_JSON_INDENT,
    LOCK_PERCENT_KEY,
    LOCK_SALT_KEY,
    SPLIT_SALT,
    TEXT_ENCODING,
)
from .files import sha256_file
from .layout import Layout
from .lock_errors import BREAKING, LockConflict
from .manifest import Row
from .problems import Problem, ProblemCode
from .registry import DatasetSpec
from .split import Split, split_of


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None

    return json.loads(path.read_text(encoding=TEXT_ENCODING))


def _test_names(spec: DatasetSpec, rows: list[Row]) -> set[str]:
    return {
        row[FILE_COLUMN]
        for row in rows
        if split_of(row[GROUP_COLUMN], spec.test_percent) is Split.TEST
    }


def lock_problems(layout: Layout, spec: DatasetSpec, rows: list[Row]) -> list[Problem]:
    """check that locked test files never moved or changed"""
    wanted = _test_names(spec, rows)
    lock = _read(layout.lock(spec.name))
    entries = lock[LOCK_ENTRIES_KEY] if lock else {}
    problems = []

    if lock and (
        lock[LOCK_SALT_KEY] != SPLIT_SALT or lock[LOCK_PERCENT_KEY] != spec.test_percent
    ):
        problems.append(Problem(ProblemCode.SPLIT_RULE_CHANGED, spec.name))

    for name, digest in sorted(entries.items()):
        path = layout.dataset_dir(spec.name) / name
        if not path.is_file():
            problems.append(Problem(ProblemCode.LOCKED_MISSING, name))
        elif sha256_file(path) != digest:
            problems.append(Problem(ProblemCode.LOCKED_CHANGED, name))

    problems += [
        Problem(ProblemCode.TEST_NOT_LOCKED, n) for n in sorted(wanted - entries.keys())
    ]

    return problems


def update_lock(layout: Layout, spec: DatasetSpec, rows: list[Row]) -> int:
    """add new test files to the lock and never remove any"""
    problems = lock_problems(layout, spec, rows)
    broken = [problem for problem in problems if problem.code in BREAKING]

    if broken:
        raise LockConflict(broken)

    lock = _read(layout.lock(spec.name)) or {
        LOCK_SALT_KEY: SPLIT_SALT,
        LOCK_PERCENT_KEY: spec.test_percent,
        LOCK_ENTRIES_KEY: {},
    }
    added = [p.subject for p in problems if p.code is ProblemCode.TEST_NOT_LOCKED]

    for name in added:
        path = layout.dataset_dir(spec.name) / name
        lock[LOCK_ENTRIES_KEY][name] = sha256_file(path)

    target = layout.lock(spec.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(lock, indent=LOCK_JSON_INDENT, sort_keys=True)
    target.write_text(text + "\n", encoding=TEXT_ENCODING)

    return len(added)
