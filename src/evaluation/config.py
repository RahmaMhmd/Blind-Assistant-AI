from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "evaluation.yaml"
MODELS_DIR = REPO_ROOT / "configs" / "models"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """load base config and extra model files"""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    models = dict(cfg.get("models", {}))
    if MODELS_DIR.is_dir():
        for extra_path in sorted(MODELS_DIR.glob("*.yaml")):
            with extra_path.open(encoding="utf-8") as extra_file:
                extra = yaml.safe_load(extra_file) or {}
            models.update(extra.get("models", {}))
    cfg["models"] = models
    return cfg
