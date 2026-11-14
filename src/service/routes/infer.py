from fastapi import APIRouter, Depends, Request

from src.contracts import InferenceFeature, SuccessResponse

from ..dependencies import get_registry, get_settings
from ..errors.docs import ERROR_RESPONSES
from ..pipeline import run_inference
from ..registry import HandlerRegistry
from ..schemas import InferenceRequest
from ..settings import ServiceSettings
from .paths import API_V1_PREFIX, INFER_ROUTE

router = APIRouter(prefix=API_V1_PREFIX)


@router.post(
    INFER_ROUTE,
    response_model=SuccessResponse,
    responses=ERROR_RESPONSES,
)
async def infer(
    feature: InferenceFeature,
    payload: InferenceRequest,
    request: Request,
    registry: HandlerRegistry = Depends(get_registry),
    settings: ServiceSettings = Depends(get_settings),
) -> SuccessResponse:
    return await run_inference(request, feature, payload, registry, settings)
