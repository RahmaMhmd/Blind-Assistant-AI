from typing import Literal

from pydantic import BaseModel

from src.contracts.models import StrictModel


class InferenceRequest(StrictModel):
    imageBase64: str | None = None
    userSpeech: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    contractVersion: str
    features: list[str]
