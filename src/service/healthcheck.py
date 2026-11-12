import urllib.request

from .settings import ServiceSettings

settings = ServiceSettings()

urllib.request.urlopen(f"http://127.0.0.1:{settings.port}/health")
