from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.contracts.image_rules import DEFAULT_MAX_IMAGE_BYTES
from src.contracts.models import MAX_CONFIDENCE, MIN_CONFIDENCE

MIN_PORT = 1
MAX_PORT = 65535


class ServiceSettings(BaseSettings):
    """read service settings from env"""

    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        extra="ignore",
    )

    service_name: str = "ai-inference"
    service_version: str = "0.1.0"
    host: str = "0.0.0.0"
    healthcheck_host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=MIN_PORT, le=MAX_PORT)
    log_level: str = "info"
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    min_confidence: float = Field(
        default=0.6, ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE
    )
    max_image_bytes: int = Field(default=DEFAULT_MAX_IMAGE_BYTES, gt=0)
    fake_model_suffix: str = "-fake"
    fake_model_version: str = "0.0.0-fake"
