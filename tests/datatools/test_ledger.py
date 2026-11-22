from datetime import datetime, timezone

from data_tools.ledger import Action, append_entry, ledger_problems, read_entries
from data_tools.problems import ProblemCode

MOMENT = datetime(2026, 11, 21, 8, 30, tzinfo=timezone.utc)


def _log(path, who, action, batch="b01"):
    return append_entry(path, who=who, action=action, batch=batch, count=3)


def test_ledger_round_trips_with_clock(tmp_path):
    path = tmp_path / "l.jsonl"
    append_entry(
        path,
        who="a",
        action=Action.COLLECTED,
        batch="b01",
        count=2,
        clock=lambda: MOMENT,
    )

    assert read_entries(path)[0]["at"] == "2026-11-21T08:30:00+00:00"


def test_ledger_needs_all_roles_per_batch(tmp_path):
    path = tmp_path / "l.jsonl"
    _log(path, "a", Action.COLLECTED)

    codes = [p.subject for p in ledger_problems({"b01"}, read_entries(path))]

    assert codes == ["b01:labeled", "b01:reviewed"]


def test_ledger_rejects_self_review(tmp_path):
    path = tmp_path / "l.jsonl"
    for action in (Action.COLLECTED, Action.LABELED, Action.REVIEWED):
        _log(path, "a", action)

    problems = ledger_problems({"b01"}, read_entries(path))

    assert [p.code for p in problems] == [ProblemCode.SELF_REVIEW]


def test_ledger_accepts_a_second_person_review(tmp_path):
    path = tmp_path / "l.jsonl"
    for who, action in (
        ("a", Action.COLLECTED),
        ("a", Action.LABELED),
        ("b", Action.REVIEWED),
    ):
        _log(path, who, action)

    assert ledger_problems({"b01"}, read_entries(path)) == []
