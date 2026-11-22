from collections import defaultdict
from pathlib import Path

from .cards import card_problems
from .constants import BATCH_COLUMN, TEXT_ENCODING
from .faces import FaceDetector
from .files import sha256_file
from .image_io import has_metadata
from .layout import Layout
from .ledger import ledger_problems, read_entries
from .lock import lock_problems
from .manifest import manifest_problems, read_manifest
from .naming import parse_name
from .problems import Problem, ProblemCode
from .registry import DatasetSpec, Kind


def _image_problems(path: Path, detector: FaceDetector | None) -> list[Problem]:
    try:
        dirty = has_metadata(path)
    except OSError:
        return [Problem(ProblemCode.UNREADABLE, path.name)]

    problems = [Problem(ProblemCode.HAS_METADATA, path.name)] if dirty else []

    if detector and detector.count_faces(path):
        problems.append(Problem(ProblemCode.FACES_FOUND, path.name))

    return problems


def _file_problems(
    spec: DatasetSpec, paths: list[Path], detector: FaceDetector | None
) -> list[Problem]:
    problems = []

    for path in paths:
        parts = parse_name(path.name)
        if parts is None:
            problems.append(Problem(ProblemCode.BAD_NAME, path.name))
        elif parts.dataset != spec.name:
            problems.append(Problem(ProblemCode.WRONG_DATASET, path.name))
        elif path.suffix.lower() not in spec.extensions:
            problems.append(Problem(ProblemCode.BAD_EXTENSION, path.name))
        elif spec.kind is Kind.IMAGE:
            problems += _image_problems(path, detector)

    return problems


def _duplicate_problems(paths: list[Path]) -> list[Problem]:
    by_hash = defaultdict(list)

    for path in paths:
        by_hash[sha256_file(path)].append(path.name)

    return [
        Problem(ProblemCode.DUPLICATE, " == ".join(names))
        for names in by_hash.values()
        if len(names) > 1
    ]


def check_dataset(
    layout: Layout, spec: DatasetSpec, detector: FaceDetector | None = None
) -> list[Problem]:
    """run every rule that must pass before a dataset is published"""
    directory = layout.dataset_dir(spec.name)
    paths = (
        sorted(p for p in directory.iterdir() if p.is_file())
        if directory.is_dir()
        else []
    )
    rows = read_manifest(layout.manifest(spec.name))
    card = layout.card(spec.name)
    card_text = card.read_text(encoding=TEXT_ENCODING) if card.is_file() else None
    batches = {row[BATCH_COLUMN] for row in rows}
    entries = read_entries(layout.ledger(spec.name))

    return [
        *_file_problems(spec, paths, detector),
        *manifest_problems(spec, rows, {p.name for p in paths}),
        *_duplicate_problems(paths),
        *lock_problems(layout, spec, rows),
        *card_problems(spec.name, card_text),
        *ledger_problems(batches, entries),
    ]
