from fastapi import Request

from src.contracts import (
    InferenceFeature,
    SuccessResponse,
    validate_feature_payload,
)

from .registry import HandlerRegistry
from .runtime.context import elapsed_ms, ensure_context
from .runtime.guards import ensure_confident, with_deadline
from .schemas import InferenceRequest
from .settings import ServiceSettings


async def run_inference(
    request: Request,
    feature: InferenceFeature,
    payload: InferenceRequest,
    registry: HandlerRegistry,
    settings: ServiceSettings,
) -> SuccessResponse:
    """check input and run model handler"""
    validate_feature_payload(
        feature,
        image_base64=payload.imageBase64,
        user_speech=payload.userSpeech,
        max_image_bytes=settings.max_image_bytes,
    )

    handler = registry.resolve(feature)

    result = await with_deadline(
        handler.predict(payload),
        settings.request_timeout_seconds,
        handler.model_name,
    )

    ensure_confident(result, settings.min_confidence, handler.model_name)

    return SuccessResponse(
        requestId=ensure_context(request),
        modelName=handler.model_name,
        modelVersion=handler.model_version,
        processingTimeMs=elapsed_ms(request),
        data=result.model_dump(mode="json"),
    )
