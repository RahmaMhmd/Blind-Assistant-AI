from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "quantization.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_spec(model_name: str, cfg: dict | None = None) -> dict:
    models = (cfg or load_config())["models"]
    if model_name not in models:
        raise ValueError(f"unknown model '{model_name}', available: {sorted(models)}")
    return models[model_name]


def resolve_path(rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else REPO_ROOT / path
