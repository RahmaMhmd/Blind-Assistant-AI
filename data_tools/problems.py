from dataclasses import dataclass
from enum import Enum


class ProblemCode(str, Enum):
    BAD_NAME = "bad_name"
    WRONG_DATASET = "wrong_dataset"
    BAD_EXTENSION = "bad_extension"
    UNREADABLE = "unreadable"
    HAS_METADATA = "has_metadata"
    FACES_FOUND = "faces_found"
    DUPLICATE = "duplicate"
    NOT_IN_MANIFEST = "not_in_manifest"
    MISSING_FILE = "missing_file"
    MISSING_LABEL = "missing_label"
    BAD_LABEL = "bad_label"
    TEST_NOT_LOCKED = "test_not_locked"
    LOCKED_CHANGED = "locked_changed"
    LOCKED_MISSING = "locked_missing"
    SPLIT_RULE_CHANGED = "split_rule_changed"
    CARD_MISSING = "card_missing"
    CARD_INCOMPLETE = "card_incomplete"
    LEDGER_GAP = "ledger_gap"
    SELF_REVIEW = "self_review"


@dataclass(frozen=True)
class Problem:
    code: ProblemCode
    subject: str

    def __str__(self) -> str:
        return f"{self.code.value}: {self.subject}"
