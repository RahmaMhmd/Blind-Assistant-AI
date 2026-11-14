from pydantic import Field, field_validator

from .models import StrictModel
from .validation_messages import Problem


def _require_text(value: str, problem: str) -> str:
    if not value.strip():
        raise ValueError(problem)

    return value


class ImageInput(StrictModel):
    imageBase64: str = Field(min_length=1)

    @field_validator("imageBase64")
    @classmethod
    def image_not_blank(cls, value: str) -> str:
        return _require_text(value, Problem.IMAGE_BLANK)


class VoiceCommandInput(StrictModel):
    userSpeech: str = Field(min_length=1)

    @field_validator("userSpeech")
    @classmethod
    def speech_not_blank(cls, value: str) -> str:
        return _require_text(value, Problem.SPEECH_BLANK)
