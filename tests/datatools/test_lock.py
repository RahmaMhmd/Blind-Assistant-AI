import pytest

from data_tools.lock import LockConflict, lock_problems, update_lock
from data_tools.manifest import new_rows
from data_tools.problems import ProblemCode
from data_tools.registry import get_spec
from data_tools.split import Split, split_of
from tests.datatools.support import DATASET, make_image, make_layout, name_of

GROUP_COUNT = 40


def _setup(tmp_path, count=GROUP_COUNT):
    layout = make_layout(tmp_path)
    spec = get_spec(layout, DATASET)
    names = [name_of(group) for group in range(count)]

    for index, name in enumerate(names):
        make_image(layout.dataset_dir(DATASET) / name, shade=index)

    return layout, spec, new_rows(spec, names)


def _test_names(spec, names):
    return [
        n for n in names if split_of(n.split("_")[2], spec.test_percent) is Split.TEST
    ]


def test_first_lock_adds_every_test_file(tmp_path):
    layout, spec, rows = _setup(tmp_path)
    expected = _test_names(spec, [row["file"] for row in rows])

    assert expected
    assert update_lock(layout, spec, rows) == len(expected)
    assert lock_problems(layout, spec, rows) == []


def test_second_lock_adds_nothing(tmp_path):
    layout, spec, rows = _setup(tmp_path)
    update_lock(layout, spec, rows)

    assert update_lock(layout, spec, rows) == 0


def test_new_batch_is_appended_without_touching_old_entries(tmp_path):
    layout, spec, rows = _setup(tmp_path)
    update_lock(layout, spec, rows)
    extra = [
        name_of(group, batch="b02") for group in range(GROUP_COUNT, GROUP_COUNT * 2)
    ]
    for name in extra:
        make_image(layout.dataset_dir(DATASET) / name, shade=7)

    assert update_lock(layout, spec, rows + new_rows(spec, extra)) > 0


def test_changed_locked_file_blocks_the_lock(tmp_path):
    layout, spec, rows = _setup(tmp_path)
    update_lock(layout, spec, rows)
    victim = _test_names(spec, [row["file"] for row in rows])[0]
    make_image(layout.dataset_dir(DATASET) / victim, shade=250)

    with pytest.raises(LockConflict) as conflict:
        update_lock(layout, spec, rows)

    assert conflict.value.problems[0].code is ProblemCode.LOCKED_CHANGED


def test_deleted_locked_file_is_reported(tmp_path):
    layout, spec, rows = _setup(tmp_path)
    update_lock(layout, spec, rows)
    victim = _test_names(spec, [row["file"] for row in rows])[0]
    (layout.dataset_dir(DATASET) / victim).unlink()

    codes = {p.code for p in lock_problems(layout, spec, rows)}

    assert ProblemCode.LOCKED_MISSING in codes


def test_changed_percent_is_reported(tmp_path):
    layout, spec, rows = _setup(tmp_path)
    update_lock(layout, spec, rows)
    other = spec.model_copy(update={"test_percent": spec.test_percent + 5})

    codes = {p.code for p in lock_problems(layout, other, rows)}

    assert ProblemCode.SPLIT_RULE_CHANGED in codes


def test_unlocked_test_files_are_reported(tmp_path):
    layout, spec, rows = _setup(tmp_path)

    codes = {p.code for p in lock_problems(layout, spec, rows)}

    assert codes == {ProblemCode.TEST_NOT_LOCKED}
