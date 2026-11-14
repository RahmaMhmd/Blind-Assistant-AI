from fastapi import Request

from .registry import HandlerRegistry
from .settings import ServiceSettings


def get_settings(request: Request) -> ServiceSettings:
    return request.app.state.settings


def get_registry(request: Request) -> HandlerRegistry:
    return request.app.state.registry
