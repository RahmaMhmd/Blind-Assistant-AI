from src.contracts import InferenceFeature

HEALTH_PATH = "/health"
API_V1_PREFIX = "/v1"
INFER_SEGMENT = "/infer"
INFER_ROUTE = f"{INFER_SEGMENT}/{{feature}}"


def infer_path(feature: InferenceFeature) -> str:
    return f"{API_V1_PREFIX}{INFER_SEGMENT}/{feature.value}"
