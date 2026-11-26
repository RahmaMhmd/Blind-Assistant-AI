from data_tools.cards import starter_card
from data_tools.checks import check_dataset
from data_tools.ingest import ingest_batch
from data_tools.ledger import Action, append_entry
from data_tools.lock import update_lock
from data_tools.manifest import columns_for, read_manifest, write_manifest
from data_tools.problems import ProblemCode
from data_tools.registry import get_spec
from tests.datatools.support import (
    DATASET,
    FakeDetector,
    make_image,
    make_layout,
    name_of,
)

GROUPS = 30


def _codes(problems):
    return [problem.code for problem in problems]


def _batch(tmp_path):
    source = tmp_path / "incoming"
    for group in range(GROUPS):
        make_image(source / name_of(group), shade=group * 3, tagged=True)

    return source


def _ready(tmp_path):
    layout = make_layout(tmp_path)
    spec = get_spec(layout, DATASET)
    ingest_batch(layout, spec, _batch(tmp_path), None)
    rows = read_manifest(layout.manifest(DATASET))
    for row in rows:
        row.update({"quality": "sharp", "note": "x"})
    write_manifest(layout.manifest(DATASET), rows, columns_for(spec))
    update_lock(layout, spec, rows)

    return layout, spec


def test_unfinished_dataset_lists_card_and_ledger_gaps(tmp_path):
    layout, spec = _ready(tmp_path)

    codes = set(_codes(check_dataset(layout, spec)))

    assert codes == {ProblemCode.CARD_MISSING, ProblemCode.LEDGER_GAP}


def test_finished_dataset_has_no_problems(tmp_path):
    layout, spec = _ready(tmp_path)
    card = starter_card(spec)
    for field in (
        "task",
        "source",
        "collected_on",
        "license",
        "consent",
        "collected_by",
    ):
        card = card.replace(f"{field}:", f"{field}: set", 1)
    layout.card(DATASET).parent.mkdir(parents=True)
    layout.card(DATASET).write_text(card)
    for who, action in (
        ("a", Action.COLLECTED),
        ("a", Action.LABELED),
        ("b", Action.REVIEWED),
    ):
        append_entry(
            layout.ledger(DATASET), who=who, action=action, batch="b01", count=GROUPS
        )

    assert check_dataset(layout, spec) == []


def test_check_reports_a_changed_locked_file(tmp_path):
    layout, spec = _ready(tmp_path)
    locked = sorted(layout.dataset_dir(DATASET).iterdir())
    for path in locked:
        make_image(path, shade=255)

    assert ProblemCode.LOCKED_CHANGED in _codes(check_dataset(layout, spec))


def test_check_reports_faces_when_a_detector_is_given(tmp_path):
    layout, spec = _ready(tmp_path)

    problems = check_dataset(layout, spec, FakeDetector({name_of(2)}))

    assert ProblemCode.FACES_FOUND in _codes(problems)
