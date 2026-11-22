import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .config import REPO_ROOT, load_config
MANIFEST_NAME = "manifest.json"
@dataclass(frozen=True)
class Sample:
    id: str
    path: Path
    reference: Any
@dataclass(frozen=True)
class EvalSet:
    name: str
    task: str
    samples: list[Sample]

def _field(item: dict, key: str, where: str):
    if key not in item:
        raise ValueError(f"{where}: missing '{key}'")
    return item[key]
def _validate(name: str, samples: list[Sample]) -> None:
    if not samples:
        raise ValueError(f"test set '{name}' has no samples")
    ids = [sample.id for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"test set '{name}' has duplicate sample ids")
    for sample in samples:
        if not sample.path.is_file():
            raise FileNotFoundError(f"sample '{sample.id}' file not found: {sample.path}")
def load_evalset(name: str, root: Path | None = None) -> EvalSet:
    root = Path(root) if root else REPO_ROOT / load_config()["paths"]["testsets_dir"]
    folder = root / name
    manifest = folder / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"no manifest for test set '{name}': {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    task = _field(data, "task", name)
    samples = [
        Sample(
            id=_field(item, "id", name),
            path=folder / _field(item, "file", name),
            reference=_field(item, "reference", name),
        )
        for item in _field(data, "samples", name)
    ]
    _validate(name, samples)

    return EvalSet(name, task, samples)
