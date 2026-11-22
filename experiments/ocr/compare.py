import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cases import OcrCase
from .normalize import cer, wer

MS_PER_SECOND = 1000
P95 = 95


class OcrReader(Protocol):
    name: str
    languages: frozenset[str]

    def read_text(self, image: bytes) -> str: ...


@dataclass(frozen=True)
class OcrStats:
    name: str
    total: int
    mean_cer: float
    mean_wer: float
    empty_rate: float
    median_ms: float
    p95_ms: float

    def as_metrics(self) -> dict[str, float]:
        return {
            f"{self.name}_cer": self.mean_cer,
            f"{self.name}_wer": self.mean_wer,
            f"{self.name}_empty_rate": self.empty_rate,
            f"{self.name}_median_ms": self.median_ms,
            f"{self.name}_p95_ms": self.p95_ms,
        }


def _timed(reader: OcrReader, case: OcrCase) -> tuple[str, float]:
    started = time.perf_counter()
    text = reader.read_text(case.image)

    return text, (time.perf_counter() - started) * MS_PER_SECOND


def compare_ocr(reader: OcrReader, cases: list[OcrCase]) -> OcrStats:
    """measure character error word error and latency on matching language cases"""
    own = [case for case in cases if case.language in reader.languages]
    runs = [(case, *_timed(reader, case)) for case in own]
    cers = [cer(case.text, pred) for case, pred, _ in runs]
    wers = [wer(case.text, pred) for case, pred, _ in runs]
    empties = sum(1 for _, pred, _ in runs if not pred.strip())
    times = np.array([elapsed for _, _, elapsed in runs] or [0.0])
    count = len(own)

    return OcrStats(
        reader.name,
        count,
        float(np.mean(cers)) if count else 0.0,
        float(np.mean(wers)) if count else 0.0,
        empties / count if count else 0.0,
        float(np.median(times)),
        float(np.percentile(times, P95)),
    )
