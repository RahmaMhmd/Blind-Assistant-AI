from data_tools.manifest import (
    columns_for,
    manifest_problems,
    new_rows,
    read_manifest,
    write_manifest,
)
from data_tools.problems import ProblemCode
from data_tools.registry import get_spec
from tests.datatools.support import DATASET, make_layout, name_of


def _spec(tmp_path):
    return get_spec(make_layout(tmp_path), DATASET)


def _codes(problems):
    return {problem.code for problem in problems}


def _labelled(spec, names):
    rows = new_rows(spec, names)
    for row in rows:
        row.update({"quality": "sharp", "note": "x"})

    return rows


def test_new_rows_follow_the_name(tmp_path):
    spec = _spec(tmp_path)
    row = new_rows(spec, [name_of(5)])[0]

    assert row["group"] == "g005"
    assert row["batch"] == "b01"
    assert row["quality"] == ""


def test_write_then_read_round_trips(tmp_path):
    spec = _spec(tmp_path)
    path = tmp_path / "out" / "m.csv"
    rows = _labelled(spec, [name_of(1), name_of(2)])

    write_manifest(path, rows, columns_for(spec))

    assert read_manifest(path) == rows


def test_read_missing_manifest_is_empty(tmp_path):
    assert read_manifest(tmp_path / "none.csv") == []


def test_complete_manifest_has_no_problems(tmp_path):
    spec = _spec(tmp_path)
    names = [name_of(1), name_of(2)]

    assert manifest_problems(spec, _labelled(spec, names), set(names)) == []


def test_blank_and_unknown_labels_are_reported(tmp_path):
    spec = _spec(tmp_path)
    rows = new_rows(spec, [name_of(1), name_of(2)])
    rows[1].update({"quality": "foggy", "note": "x"})
    names = {name_of(1), name_of(2)}

    codes = _codes(manifest_problems(spec, rows, names))

    assert {ProblemCode.MISSING_LABEL, ProblemCode.BAD_LABEL} <= codes


def test_file_and_row_mismatch_is_reported(tmp_path):
    spec = _spec(tmp_path)
    rows = _labelled(spec, [name_of(1)])

    codes = _codes(manifest_problems(spec, rows, {name_of(2)}))

    assert codes == {ProblemCode.NOT_IN_MANIFEST, ProblemCode.MISSING_FILE}


def test_repeated_row_is_reported(tmp_path):
    spec = _spec(tmp_path)
    rows = _labelled(spec, [name_of(1), name_of(1)])

    assert ProblemCode.DUPLICATE in _codes(manifest_problems(spec, rows, {name_of(1)}))


def test_group_must_match_file_name(tmp_path):
    spec = _spec(tmp_path)
    rows = _labelled(spec, [name_of(1)])
    rows[0]["group"] = "g999"

    assert ProblemCode.BAD_NAME in _codes(manifest_problems(spec, rows, {name_of(1)}))
