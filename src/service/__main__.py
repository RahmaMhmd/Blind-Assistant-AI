import uvicorn

from .main import create_app
from .settings import ServiceSettings


def main() -> None:
    settings = ServiceSettings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
