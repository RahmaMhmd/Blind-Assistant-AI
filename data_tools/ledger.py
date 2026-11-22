import json
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .constants import TEXT_ENCODING
from .problems import Problem, ProblemCode

Clock = Callable[[], datetime]


class Action(str, Enum):
    COLLECTED = "collected"
    LABELED = "labeled"
    REVIEWED = "reviewed"
    CLEANED = "cleaned"
    LOCKED = "locked"
    PUBLISHED = "published"


REQUIRED_ACTIONS = (Action.COLLECTED, Action.LABELED, Action.REVIEWED)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_entry(
    path: Path,
    *,
    who: str,
    action: Action,
    batch: str,
    count: int,
    note: str = "",
    clock: Clock = utc_now,
) -> dict:
    entry = {
        "at": clock().isoformat(timespec="seconds"),
        "who": who,
        "action": action.value,
        "batch": batch,
        "count": count,
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding=TEXT_ENCODING) as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def read_entries(path: Path) -> list[dict]:
    if not path.is_file():
        return []

    lines = path.read_text(encoding=TEXT_ENCODING).splitlines()

    return [json.loads(line) for line in lines if line.strip()]


def _people(entries: list[dict], batch: str, action: Action) -> set[str]:
    return {
        entry["who"]
        for entry in entries
        if entry["batch"] == batch and entry["action"] == action.value
    }


def ledger_problems(batches: set[str], entries: list[dict]) -> list[Problem]:
    """every batch needs a collector a labeler and another reviewer"""
    problems = []

    for batch in sorted(batches):
        for action in REQUIRED_ACTIONS:
            if not _people(entries, batch, action):
                problems.append(
                    Problem(ProblemCode.LEDGER_GAP, f"{batch}:{action.value}")
                )

        reviewers = _people(entries, batch, Action.REVIEWED)
        if reviewers and not reviewers - _people(entries, batch, Action.LABELED):
            problems.append(Problem(ProblemCode.SELF_REVIEW, batch))

    return problems
