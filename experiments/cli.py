import argparse

from data_tools.split import Split

from .commands import qr_compare, quality_calibrate

QUALITY_DATASET = "image-quality"
QR_DATASET = "qr-barcode"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    quality = commands.add_parser("quality-calibrate")
    quality.add_argument("--dataset", default=QUALITY_DATASET)
    quality.set_defaults(run=quality_calibrate)

    qr = commands.add_parser("qr-compare")
    qr.add_argument("--dataset", default=QR_DATASET)
    splits = [split.value for split in Split]
    qr.add_argument("--split", default=Split.TRAIN.value, choices=splits)
    qr.set_defaults(run=qr_compare)

    for sub in (quality, qr):
        sub.add_argument("--no-track", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    return args.run(args)
