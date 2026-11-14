from src.contracts import ErrorResponse, ai_emittable_codes, error_http_status

ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in {error_http_status(code) for code in ai_emittable_codes()}
}
