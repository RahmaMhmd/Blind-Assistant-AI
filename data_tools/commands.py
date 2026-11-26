import argparse
from pathlib import Path

from .checks import check_dataset
from .command_utils import EXIT_OK, detector_from, report, setup
from .dvc import publish, subprocess_runner
from .ingest import ingest_batch
from .lock import update_lock
from .lock_errors import LockConflict
from .manifest import read_manifest


def run_ingest(args: argparse.Namespace) -> int:
    layout, spec = setup(args)
    result = ingest_batch(layout, spec, Path(args.source), detector_from(args))
    print(f"added {len(result.added)} rejected {len(result.rejected)}")

    return report(result.rejected)


def run_check(args: argparse.Namespace) -> int:
    layout, spec = setup(args)

    return report(check_dataset(layout, spec, detector_from(args)))


def run_lock(args: argparse.Namespace) -> int:
    layout, spec = setup(args)
    rows = read_manifest(layout.manifest(spec.name))

    try:
        print(f"locked {update_lock(layout, spec, rows)} new test files")
    except LockConflict as conflict:
        return report(conflict.problems)

    return EXIT_OK


def run_publish(args: argparse.Namespace) -> int:
    layout, spec = setup(args)
    problems = check_dataset(layout, spec, detector_from(args))

    if problems:
        return report(problems)

    for command in publish(layout, spec.name, subprocess_runner, args.push):
        print(" ".join(command))

    return EXIT_OK
