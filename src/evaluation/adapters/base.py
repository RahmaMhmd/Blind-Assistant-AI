from pathlib import Path
from typing import Any, Protocol


class Adapter(Protocol):
    """Every model is wrapped in this shape: a sample file in, the raw answer out."""

    name: str
    task: str

    def predict(self, sample_path: Path) -> Any: ...
