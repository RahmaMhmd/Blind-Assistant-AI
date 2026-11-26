import argparse
import sys

from .benchmark import benchmark_model
from .calibration import prepare_calibration
from .export import export_fp32
from .quantize import quantize_int8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.quantization")
    parser.add_argument("action", choices=["calibrate", "export", "quantize", "benchmark"])
    parser.add_argument("--model", default="rtdetrv2_r18")
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.action == "calibrate":
            saved = prepare_calibration()
            print(f"saved {len(saved)} calibration images")
        elif args.action == "export":
            out = export_fp32(args.model)
            print(f"exported {out}")
        elif args.action == "quantize":
            out = quantize_int8(args.model)
            print(f"quantized {out}")
        elif args.action == "benchmark":
            summary = benchmark_model(args.model, track=args.track)
            for k, v in summary.items():
                print(f"{k}: {v}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
