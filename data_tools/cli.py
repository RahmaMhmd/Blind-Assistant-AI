import argparse
import sys

from . import commands, commands_meta
from .ledger import Action


def _dataset_parser(subparsers, name: str, handler) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("dataset")
    parser.set_defaults(handler=handler)

    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data_tools")
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = _dataset_parser(subparsers, "ingest", commands.run_ingest)
    ingest.add_argument("source")
    faces = ingest.add_mutually_exclusive_group(required=True)
    faces.add_argument("--face-model")
    faces.add_argument("--skip-faces", action="store_true")

    for name, handler in (
        ("check", commands.run_check),
        ("publish", commands.run_publish),
    ):
        _dataset_parser(subparsers, name, handler).add_argument("--face-model")
    subparsers.choices["publish"].add_argument("--push", action="store_true")

    _dataset_parser(subparsers, "lock", commands.run_lock)
    _dataset_parser(subparsers, "card", commands_meta.run_card)

    log = _dataset_parser(subparsers, "log", commands_meta.run_log)
    log.add_argument("--who", required=True)
    log.add_argument("--action", required=True, choices=[a.value for a in Action])
    log.add_argument("--batch", required=True)
    log.add_argument("--count", type=int, required=True)
    log.add_argument("--note", default="")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
