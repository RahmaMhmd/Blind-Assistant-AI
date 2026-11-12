from fastapi import APIRouter, Depends

from ..dependencies import get_registry, get_settings
from ..registry import HandlerRegistry
from ..schemas import HealthResponse
from ..settings import ServiceSettings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    registry: HandlerRegistry = Depends(get_registry),
    settings: ServiceSettings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        service=settings.service_name,
        version=settings.service_version,
        features=registry.features,
    )
