import time
from dataclasses import dataclass
from typing import Any

from .adapters.base import Adapter
from .testset import EvalSet


@dataclass(frozen=True)
class Row:
    id: str
    output: Any
    reference: Any
    seconds: float


@dataclass(frozen=True)
class RunResult:
    model: str
    testset: str
    task: str
    rows: list[Row]


def run(adapter: Adapter, evalset: EvalSet) -> RunResult:
    if adapter.task != evalset.task:
        raise ValueError(
            f"model '{adapter.name}' does task '{adapter.task}' "
            f"but test set '{evalset.name}' is for '{evalset.task}'"
        )
    rows = []
    for sample in evalset.samples:
        start = time.perf_counter()
        output = adapter.predict(sample.path)
        rows.append(Row(sample.id, output, sample.reference, time.perf_counter() - start))
    return RunResult(adapter.name, evalset.name, evalset.task, rows)
