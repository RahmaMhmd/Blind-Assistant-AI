from src.contracts import ErrorCode, ErrorResponse, error_http_status

ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in {error_http_status(code) for code in ErrorCode}
}
