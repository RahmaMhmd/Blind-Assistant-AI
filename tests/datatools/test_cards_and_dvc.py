from data_tools.cards import card_problems, parse_card, starter_card
from data_tools.constants import CARD_REQUIRED_FIELDS
from data_tools.dvc import publish, publish_commands
from data_tools.problems import ProblemCode
from data_tools.registry import get_spec
from tests.datatools.support import DATASET, make_layout

FILLED = "\n".join(
    [
        "---",
        *[f"{field}: value" for field in CARD_REQUIRED_FIELDS if field != "dataset"],
    ]
    + [f"dataset: {DATASET}", "---", "", "body"]
)


def _spec(tmp_path):
    return get_spec(make_layout(tmp_path), DATASET)


def test_parse_reads_header_only():
    fields = parse_card(FILLED)

    assert fields["source"] == "value"
    assert "body" not in fields


def test_parse_without_fence_is_empty():
    assert parse_card("source: x") == {}


def test_filled_card_has_no_problems():
    assert card_problems(DATASET, FILLED) == []


def test_missing_card_is_reported():
    assert card_problems(DATASET, None)[0].code is ProblemCode.CARD_MISSING


def test_starter_card_lists_every_blank_field(tmp_path):
    spec = _spec(tmp_path)
    problems = card_problems(DATASET, starter_card(spec))

    assert {p.subject.split(":")[1] for p in problems} == set(CARD_REQUIRED_FIELDS) - {
        "dataset"
    }


def test_card_for_another_dataset_is_reported():
    text = FILLED.replace(f"dataset: {DATASET}", "dataset: other")

    assert card_problems(DATASET, text)[0].subject == f"{DATASET}:dataset"


def test_publish_commands_add_then_push(tmp_path):
    layout = make_layout(tmp_path)
    target = f"data/raw/{DATASET}"

    assert publish_commands(layout, DATASET, push=False) == [["dvc", "add", target]]
    assert publish_commands(layout, DATASET, push=True)[1] == [
        "dvc",
        "push",
        f"{target}.dvc",
    ]


def test_publish_runs_each_command_in_order(tmp_path):
    layout = make_layout(tmp_path)
    seen = []

    publish(layout, DATASET, seen.append, push=True)

    assert [command[1] for command in seen] == ["add", "push"]
