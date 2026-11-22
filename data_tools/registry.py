import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    AUDIO_EXTENSIONS,
    DEFAULT_TEST_PERCENT,
    IMAGE_EXTENSIONS,
    PERCENT_BASE,
    TEXT_ENCODING,
)
from .layout import Layout

SPEC_ROOT_KEY = "datasets"


class Kind(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Kind
    labels: dict[str, list[str] | None] = Field(default_factory=dict)
    test_percent: int = Field(default=DEFAULT_TEST_PERCENT, ge=1, lt=PERCENT_BASE)

    @property
    def extensions(self) -> frozenset[str]:
        return IMAGE_EXTENSIONS if self.kind is Kind.IMAGE else AUDIO_EXTENSIONS


class UnknownDataset(KeyError):
    pass


def load_specs(layout: Layout) -> dict[str, DatasetSpec]:
    raw = json.loads(layout.spec_file.read_text(encoding=TEXT_ENCODING))

    return {
        name: DatasetSpec(name=name, **body)
        for name, body in raw[SPEC_ROOT_KEY].items()
    }


def get_spec(layout: Layout, name: str) -> DatasetSpec:
    specs = load_specs(layout)

    if name not in specs:
        raise UnknownDataset(name)

    return specs[name]
