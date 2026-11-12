import uvicorn

from .settings import ServiceSettings

settings = ServiceSettings()

uvicorn.run(
    "src.service.main:app",
    host=settings.host,
    port=settings.port,
    log_level=settings.log_level,
)
