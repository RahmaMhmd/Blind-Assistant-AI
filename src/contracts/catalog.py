import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

CONTRACTS_DIR_ENV = "AI_CONTRACTS_DIR"
DEFAULT_CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"
CONTRACTS_DIR = Path(os.environ.get(CONTRACTS_DIR_ENV) or DEFAULT_CONTRACTS_DIR)
TEXT_ENCODING = "utf-8"
VERSION_FILE = "contract-version.txt"
ERROR_CODES_FILE = "error-codes.json"
ENUMS_FILE = "enums.json"
SUCCESS_FIXTURES_DIR = "fixtures/success"
ERROR_FIXTURES_DIR = "fixtures/error"


def _read_text(relative_path: str) -> str:
    return (CONTRACTS_DIR / relative_path).read_text(encoding=TEXT_ENCODING)


@lru_cache
def read_json(relative_path: str) -> Any:
    """load and cache one json contract file"""
    return json.loads(_read_text(relative_path))


def contract_version() -> str:
    return _read_text(VERSION_FILE).strip()


def success_fixture(feature: str) -> dict[str, Any]:
    return read_json(f"{SUCCESS_FIXTURES_DIR}/{feature}.json")
