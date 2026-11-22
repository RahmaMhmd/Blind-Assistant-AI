import pytest

from data_tools.constants import PERCENT_BASE
from data_tools.naming import ParsedName, build_name, parse_name
from data_tools.split import Split, bucket, split_of

VALID = "ocr-text_b01_g014_007.jpg"
INVALID = [
    "IMG_2041.jpg",
    "ocr-text_b1_g014_007.jpg",
    "ocr-text_b01_g14_007.jpg",
    "ocr-text_b01_g014_7.jpg",
    "ocr-text_b01_g014_007",
    "Ocr-text_b01_g014_007.jpg",
]
GROUPS = [f"g{number:03d}" for number in range(2000)]
PERCENT = 20
TOLERANCE = 3


def test_parse_valid_name():
    assert parse_name(VALID) == ParsedName("ocr-text", "b01", "g014", "007", ".jpg")


@pytest.mark.parametrize("name", INVALID)
def test_parse_rejects_bad_names(name):
    assert parse_name(name) is None


def test_build_name_round_trips():
    assert build_name(parse_name(VALID)) == VALID


def test_split_is_stable_per_group():
    first = [split_of(group, PERCENT) for group in GROUPS]
    second = [split_of(group, PERCENT) for group in GROUPS]

    assert first == second


def test_split_share_follows_percent():
    tests = sum(split_of(group, PERCENT) is Split.TEST for group in GROUPS)
    share = tests * PERCENT_BASE / len(GROUPS)

    assert abs(share - PERCENT) < TOLERANCE


def test_salt_changes_assignment():
    other = [bucket(group, "other") for group in GROUPS]

    assert other != [bucket(group) for group in GROUPS]


def test_growing_percent_never_moves_test_back_to_train():
    low = {g for g in GROUPS if split_of(g, 10) is Split.TEST}
    high = {g for g in GROUPS if split_of(g, 30) is Split.TEST}

    assert low <= high
