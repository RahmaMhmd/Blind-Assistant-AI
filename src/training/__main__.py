import argparse
from pathlib import Path
import sys

from .currency_calib import prepare_currency_calib
from .currency_train import train_currency
from .intent_data import prepare_intent_dataset
from .intent_train import train_intent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.training")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cur = sub.add_parser("currency")
    cur.add_argument("--epochs", type=int, default=10)
    cur.add_argument("--batch-size", type=int, default=32)
    cur.add_argument("--lr", type=float, default=3e-4)
    cur.add_argument("--no-track", action="store_true")

    sub.add_parser("currency-calib")

    idata = sub.add_parser("intent-data")
    idata.add_argument("--csv", required=True)
    idata.add_argument("--min-per-intent", type=int, default=20)

    itrain = sub.add_parser("intent")
    itrain.add_argument("--train", default="data/intent/train.jsonl")
    itrain.add_argument("--epochs", type=int, default=5)
    itrain.add_argument("--batch-size", type=int, default=16)
    itrain.add_argument("--lr", type=float, default=3e-5)
    itrain.add_argument("--no-track", action="store_true")
    return parser


def main(argv: list[str] | None = None, root: Path = Path(".")) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "currency":
        out = train_currency(root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, track=not args.no_track)
        for k, v in out["metrics"].items():
            print(f"{k}: {v}")
    elif args.cmd == "currency-calib":
        calib, bench = prepare_currency_calib(root)
        print(f"prepared {calib} and {bench}")
    elif args.cmd == "intent-data":
        ev, tr = prepare_intent_dataset(Path(args.csv), root, min_per_intent=args.min_per_intent)
        print(f"wrote {tr} and {ev}")
    elif args.cmd == "intent":
        out = train_intent(Path(args.train), root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, track=not args.no_track)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
