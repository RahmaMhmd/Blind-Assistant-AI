from src.contracts import ErrorCode


class ModelServiceError(Exception):
    def __init__(
        self,
        error_code: ErrorCode,
        details: list[str] | None = None,
        model_name: str | None = None,
    ):
        self.error_code = error_code
        self.details = details or []
        self.model_name = model_name

        super().__init__(error_code.value)
