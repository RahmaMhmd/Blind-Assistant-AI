import sys
import urllib.request
from http import HTTPStatus

from .routes.paths import HEALTH_PATH
from .settings import ServiceSettings


def main() -> None:
    """check local health endpoint"""
    settings = ServiceSettings()
    url = f"http://{settings.healthcheck_host}:{settings.port}{HEALTH_PATH}"

    with urllib.request.urlopen(
        url, timeout=settings.request_timeout_seconds
    ) as response:
        if response.status != HTTPStatus.OK:
            sys.exit(1)


if __name__ == "__main__":
    main()
