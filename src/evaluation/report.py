from statistics import mean

from .metrics import METRICS
from .runner import RunResult

FLOAT_DIGITS = 4
SCORE_DIGITS = 2
DISPLAY_LIMIT = 300  # long texts are cut in the printout only, never in the measurement


def _metric_for(result: RunResult):
    if result.task not in METRICS:
        raise ValueError(f"no metric for task '{result.task}'")

    return METRICS[result.task]


def summarize(result: RunResult) -> dict:
    scores = _metric_for(result)(
        [row.reference for row in result.rows],
        [row.output for row in result.rows],
    )
    seconds = [row.seconds for row in result.rows]

    return {
        "model": result.model,
        "testset": result.testset,
        "task": result.task,
        "samples": len(result.rows),
        **scores,
        "mean_seconds": mean(seconds),
        "max_seconds": max(seconds),
    }


def row_records(result: RunResult) -> list[dict]:
    """One record per sample: answer, reference, time and the sample's own scores."""
    metric = _metric_for(result)

    return [
        {
            "id": row.id,
            "seconds": row.seconds,
            "reference": row.reference,
            "output": row.output,
            **metric([row.reference], [row.output]),
        }
        for row in result.rows
    ]


def format_summary(summary: dict) -> str:
    return "\n".join(
        f"{key}: {round(value, FLOAT_DIGITS) if isinstance(value, float) else value}"
        for key, value in summary.items()
    )


def _shorten(value) -> str:
    text = str(value)
    hidden = len(text) - DISPLAY_LIMIT

    return text if hidden <= 0 else f"{text[:DISPLAY_LIMIT]}... (+{hidden} more characters)"


def _scores_text(record: dict, score_names: list[str]) -> str:
    return " ".join(f"{name}={record[name]:.{SCORE_DIGITS}f}" for name in score_names)


def format_rows(result: RunResult) -> str:
    records = row_records(result)
    fixed = {"id", "seconds", "reference", "output"}
    score_names = [name for name in records[0] if name not in fixed] if records else []

    return "\n".join(
        f"[{record['id']}] {record['seconds']:.2f}s {_scores_text(record, score_names)}\n"
        f"  reference: {_shorten(record['reference'])}\n"
        f"  output:    {_shorten(record['output'])}"
        for record in records
    )
