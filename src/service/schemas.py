from typing import Literal

from pydantic import BaseModel, ConfigDict


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    imageBase64: str | None = None
    userSpeech: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    features: list[str]
