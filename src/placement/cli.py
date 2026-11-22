import sys
from pathlib import Path

from .build import (
    BUILT_FILE,
    SCHEMA_FILE,
    TABLE_FILE,
    TEXT_ENCODING,
    dump_built,
    dump_schema,
    load_sources,
)
from .render import render_table

BUILD = "build"
CHECK = "check"


def outputs(root: Path) -> dict[Path, str]:
    placement = load_sources(root)

    return {
        root / BUILT_FILE: dump_built(placement),
        root / SCHEMA_FILE: dump_schema(),
        root / TABLE_FILE: render_table(placement),
    }


def main(argv: list[str] | None = None, root: Path = Path(".")) -> int:
    command = (argv if argv is not None else sys.argv[1:] or [CHECK])[0]
    expected = outputs(root)

    if command == BUILD:
        for path, text in expected.items():
            path.write_text(text, encoding=TEXT_ENCODING)
        return 0

    stale = [
        p
        for p, t in expected.items()
        if not p.is_file() or p.read_text(encoding=TEXT_ENCODING) != t
    ]
    for path in stale:
        print(f"out of date {path}")

    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
