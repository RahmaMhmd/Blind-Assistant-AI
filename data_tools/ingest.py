import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .faces import FaceDetector
from .files import list_files, sha256_file
from .image_io import UnsupportedImage, clean_image
from .layout import Layout
from .manifest import columns_for, new_rows, read_manifest, write_manifest
from .naming import parse_name
from .problems import Problem, ProblemCode
from .registry import DatasetSpec, Kind


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    rejected: list[Problem] = field(default_factory=list)


def _store(spec: DatasetSpec, source: Path, target: Path) -> None:
    if spec.kind is Kind.IMAGE:
        clean_image(source, target)
    else:
        shutil.copy2(source, target)


def _ingest_one(
    spec: DatasetSpec,
    source: Path,
    target_dir: Path,
    known: dict[str, str],
    detector: FaceDetector | None,
) -> Problem | None:
    parts = parse_name(source.name)
    target = target_dir / source.name

    if parts is None:
        return Problem(ProblemCode.BAD_NAME, source.name)
    if parts.dataset != spec.name:
        return Problem(ProblemCode.WRONG_DATASET, source.name)
    if source.suffix.lower() not in spec.extensions:
        return Problem(ProblemCode.BAD_EXTENSION, source.name)
    if target.exists():
        return Problem(ProblemCode.DUPLICATE, source.name)

    try:
        _store(spec, source, target)
    except (UnsupportedImage, OSError):
        target.unlink(missing_ok=True)
        return Problem(ProblemCode.UNREADABLE, source.name)

    if detector and detector.count_faces(target):
        target.unlink()
        return Problem(ProblemCode.FACES_FOUND, source.name)

    digest = sha256_file(target)
    if digest in known:
        target.unlink()
        return Problem(ProblemCode.DUPLICATE, f"{source.name} == {known[digest]}")

    known[digest] = source.name
    return None


def ingest_batch(
    layout: Layout,
    spec: DatasetSpec,
    source_dir: Path,
    detector: FaceDetector | None,
) -> IngestReport:
    """clean and screen one batch then list it in the manifest"""
    target_dir = layout.dataset_dir(spec.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    known = {sha256_file(p): p.name for p in list_files(target_dir, spec.extensions)}
    report = IngestReport()

    for source in sorted(p for p in source_dir.iterdir() if p.is_file()):
        problem = _ingest_one(spec, source, target_dir, known, detector)
        if problem:
            report.rejected.append(problem)
        else:
            report.added.append(source.name)

    if report.added:
        path = layout.manifest(spec.name)
        rows = read_manifest(path) + new_rows(spec, report.added)
        write_manifest(path, rows, columns_for(spec))

    return report
