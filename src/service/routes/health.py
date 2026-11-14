from fastapi import APIRouter, Depends

from src.contracts import contract_version

from ..dependencies import get_registry, get_settings
from ..registry import HandlerRegistry
from ..schemas import HealthResponse
from ..settings import ServiceSettings
from .paths import HEALTH_PATH

router = APIRouter()


@router.get(HEALTH_PATH, response_model=HealthResponse)
async def health(
    registry: HandlerRegistry = Depends(get_registry),
    settings: ServiceSettings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        service=settings.service_name,
        version=settings.service_version,
        contractVersion=contract_version(),
        features=registry.features,
    )
