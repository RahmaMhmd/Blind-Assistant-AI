from pathlib import Path
from typing import Any


class FakeAdapter:
    """Return one fixed answer for every sample, to test the tool itself."""

    def __init__(self, name: str, task: str, output: Any):
        self.name = name
        self.task = task
        self._output = output

    def predict(self, sample_path: Path) -> Any:
        return self._output
