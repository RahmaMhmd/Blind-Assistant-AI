import hashlib
from enum import Enum

from .constants import PERCENT_BASE, SPLIT_SALT


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


def bucket(group: str, salt: str = SPLIT_SALT) -> int:
    digest = hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()

    return int(digest, 16) % PERCENT_BASE


def split_of(group: str, percent: int, salt: str = SPLIT_SALT) -> Split:
    """same group always lands in the same split"""
    return Split.TEST if bucket(group, salt) < percent else Split.TRAIN
