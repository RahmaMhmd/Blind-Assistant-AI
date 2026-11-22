from datetime import date

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from src.contracts.models import StrictModel
from src.contracts.patterns import MODEL_NAME_PATTERN

from .enums import Commercial, Kind, Outcome, Placement, Runtime, Status, Usage

SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ROLLBACK_PATTERN = r"^[a-z0-9][a-z0-9_.-]*@[A-Za-z0-9_.-]+$"


class PlacementModel(StrictModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class License(PlacementModel):
    name: str
    commercial_use: Commercial
    source_url: str
    verified_on: date | None = None


class ModelRef(PlacementModel):
    name: str = Field(pattern=MODEL_NAME_PATTERN.pattern)
    kind: Kind
    version: str | None = None
    runtime: Runtime
    placement: Placement
    languages: list[str] = Field(default_factory=list)
    usage: Usage
    license: License
    size_mb: float | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rollback_target: str | None = Field(default=None, pattern=ROLLBACK_PATTERN)
    notes: str = ""


class Considered(PlacementModel):
    name: str
    outcome: Outcome
    reason: str


class FeatureEntry(PlacementModel):
    status: Status
    confirm_in: str | None = None
    verified_on: date
    primary: ModelRef
    fallback: ModelRef | None = None
    considered: list[Considered] = Field(default_factory=list)


class PlacementFile(PlacementModel):
    schema_version: str = SCHEMA_VERSION
    features: dict[str, FeatureEntry]
