from fastapi import FastAPI

from .errors.handlers import register_error_handlers
from .handlers.defaults import default_handlers
from .registry import HandlerRegistry
from .routes import health, infer
from .runtime.context import RequestContextMiddleware
from .settings import ServiceSettings


def create_app(
    settings: ServiceSettings | None = None,
    registry: HandlerRegistry | None = None,
) -> FastAPI:
    """build fastapi app and error handlers"""
    settings = settings or ServiceSettings()

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
    )

    app.state.settings = settings
    app.state.registry = registry or HandlerRegistry(default_handlers(settings))

    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(infer.router)

    return app


app = create_app()
