import re
from dataclasses import dataclass

NAME_PATTERN = re.compile(
    r"^(?P<dataset>[a-z][a-z0-9-]*)"
    r"_(?P<batch>b\d{2,})"
    r"_(?P<group>g\d{3,})"
    r"_(?P<seq>\d{3,})"
    r"(?P<ext>\.[a-z0-9]+)$"
)


@dataclass(frozen=True)
class ParsedName:
    dataset: str
    batch: str
    group: str
    seq: str
    ext: str


def parse_name(name: str) -> ParsedName | None:
    """read dataset batch and group from a file name"""
    match = NAME_PATTERN.match(name)

    return ParsedName(**match.groupdict()) if match else None


def build_name(parts: ParsedName) -> str:
    return f"{parts.dataset}_{parts.batch}_{parts.group}_{parts.seq}{parts.ext}"
