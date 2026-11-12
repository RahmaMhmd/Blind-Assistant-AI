from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        extra="ignore",
    )

    service_name: str = "ai-inference"
    service_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
