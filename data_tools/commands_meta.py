import argparse

from .cards import starter_card
from .command_utils import EXIT_OK, EXIT_PROBLEMS, setup
from .constants import TEXT_ENCODING
from .ledger import Action, append_entry


def run_log(args: argparse.Namespace) -> int:
    layout, spec = setup(args)
    entry = append_entry(
        layout.ledger(spec.name),
        who=args.who,
        action=Action(args.action),
        batch=args.batch,
        count=args.count,
        note=args.note,
    )
    print(entry)

    return EXIT_OK


def run_card(args: argparse.Namespace) -> int:
    layout, spec = setup(args)
    path = layout.card(spec.name)

    if path.exists():
        print(f"card exists {path}")
        return EXIT_PROBLEMS

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(starter_card(spec), encoding=TEXT_ENCODING)

    return EXIT_OK
