from pathlib import Path

DATA_DIR = "data"
RAW_DIR = "raw"
MANIFEST_DIR = "manifests"
LOCK_DIR = "locks"
CARD_DIR = "cards"
LEDGER_DIR = "ledger"
SPEC_NAME = "datasets.json"
MANIFEST_SUFFIX = ".csv"
LOCK_SUFFIX = ".json"
CARD_SUFFIX = ".md"
LEDGER_SUFFIX = ".jsonl"
DVC_SUFFIX = ".dvc"


class Layout:
    """one place for every dataset path"""

    def __init__(self, root: Path = Path(".")):
        self.root = root
        self.data = root / DATA_DIR

    @property
    def spec_file(self) -> Path:
        return self.data / SPEC_NAME

    def dataset_dir(self, name: str) -> Path:
        return self.data / RAW_DIR / name

    def manifest(self, name: str) -> Path:
        return self.data / MANIFEST_DIR / f"{name}{MANIFEST_SUFFIX}"

    def lock(self, name: str) -> Path:
        return self.data / LOCK_DIR / f"{name}{LOCK_SUFFIX}"

    def card(self, name: str) -> Path:
        return self.data / CARD_DIR / f"{name}{CARD_SUFFIX}"

    def ledger(self, name: str) -> Path:
        return self.data / LEDGER_DIR / f"{name}{LEDGER_SUFFIX}"

    def dvc_target(self, name: str) -> str:
        return (self.data / RAW_DIR / name).relative_to(self.root).as_posix()
