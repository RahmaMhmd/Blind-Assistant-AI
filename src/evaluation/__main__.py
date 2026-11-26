import argparse
import sys

from .registry import build_adapter
from .report import format_rows, format_summary, row_records, summarize
from .runner import run
from .testset import load_evalset
from .tracking import log_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation",
        description="Measure one model on one test set.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--details", action="store_true", help="also print each sample")
    parser.add_argument("--track", action="store_true", help="save the run in MLflow")
    args = parser.parse_args(argv)

    try:
        adapter = build_adapter(args.model)
        result = run(adapter, load_evalset(args.testset))
        summary = summarize(result)
        print(format_summary(summary))

        if args.details:
            print("\n" + format_rows(result))

        if args.track:
            run_id = log_evaluation(summary, row_records(result))
            print(f"\nsaved in MLflow, run id: {run_id}")
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
