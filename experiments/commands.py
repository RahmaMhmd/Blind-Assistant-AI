import argparse

from data_tools.layout import Layout
from data_tools.registry import get_spec
from data_tools.split import Split
from src.imagequality.thresholds import save_thresholds

from .constants import PROJECT_ROOT
from .qr.cases import load_cases
from .qr.compare import compare
from .qr.decoders import default_decoders
from .quality.calibrate import calibrate
from .quality.constants import THRESHOLDS_FILE
from .quality.report import report
from .quality.samples import load_samples
from .tracking import log_run


def _publish(args: argparse.Namespace, name: str, metrics: dict[str, float]) -> None:
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")

    if not args.no_track:
        log_run(name, {"dataset": args.dataset}, metrics)


def quality_calibrate(args: argparse.Namespace) -> int:
    layout = Layout(PROJECT_ROOT)
    spec = get_spec(layout, args.dataset)
    train = load_samples(layout, spec, Split.TRAIN)
    thresholds = calibrate(train, args.dataset)
    save_thresholds(layout.root / THRESHOLDS_FILE, thresholds)
    test = load_samples(layout, spec, Split.TEST, thresholds.long_side)
    metrics = report(test, thresholds)
    metrics.update(
        min_sharpness=thresholds.min_sharpness,
        min_brightness=thresholds.min_brightness,
    )
    _publish(args, "quality-calibrate", metrics)

    return 0


def qr_compare(args: argparse.Namespace) -> int:
    layout = Layout(PROJECT_ROOT)
    spec = get_spec(layout, args.dataset)
    cases = load_cases(layout, spec, Split(args.split))
    metrics = {}

    for decoder in default_decoders():
        metrics.update(compare(decoder, cases).as_metrics())

    _publish(args, "qr-compare", metrics)

    return 0
