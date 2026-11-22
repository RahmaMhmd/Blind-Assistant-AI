import time
from dataclasses import dataclass

import numpy as np

from .cases import Case
from .decoders import Decoder

MS_PER_SECOND = 1000
P95 = 95


@dataclass(frozen=True)
class DecoderStats:
    name: str
    total: int
    correct: int
    wrong: int
    median_ms: float
    p95_ms: float

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.total if self.total else 0.0

    def as_metrics(self) -> dict[str, float]:
        prefix = self.name
        return {
            f"{prefix}_correct_rate": self.correct_rate,
            f"{prefix}_wrong_rate": self.wrong_rate,
            f"{prefix}_median_ms": self.median_ms,
            f"{prefix}_p95_ms": self.p95_ms,
        }


def _timed(decoder: Decoder, case: Case) -> tuple[list[str], float]:
    started = time.perf_counter()
    decoded = decoder.decode(case.gray)

    return decoded, (time.perf_counter() - started) * MS_PER_SECOND


def compare(decoder: Decoder, cases: list[Case]) -> DecoderStats:
    """a read is wrong when something was decoded but not the expected value"""
    own = [case for case in cases if case.kind in decoder.kinds]
    runs = [(case, *_timed(decoder, case)) for case in own]
    correct = sum(1 for case, decoded, _ in runs if case.value in decoded)
    wrong = sum(1 for case, decoded, _ in runs if decoded and case.value not in decoded)
    times = np.array([elapsed for _, _, elapsed in runs] or [0.0])

    return DecoderStats(
        decoder.name,
        len(own),
        correct,
        wrong,
        float(np.median(times)),
        float(np.percentile(times, P95)),
    )
