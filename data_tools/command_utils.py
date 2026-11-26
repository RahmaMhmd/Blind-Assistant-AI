import argparse
from pathlib import Path

from .faces import FaceDetector, YuNetDetector
from .layout import Layout
from .registry import get_spec

EXIT_OK = 0
EXIT_PROBLEMS = 1


def setup(args: argparse.Namespace):
    layout = Layout(Path(args.root))

    return layout, get_spec(layout, args.dataset)


def detector_from(args: argparse.Namespace) -> FaceDetector | None:
    model = getattr(args, "face_model", None)

    return YuNetDetector(Path(model)) if model else None


def report(problems: list) -> int:
    for problem in problems:
        print(problem)

    return EXIT_PROBLEMS if problems else EXIT_OK
