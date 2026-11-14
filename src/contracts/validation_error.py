from .error_catalog import get_error_message
from .errors import ErrorCode


class ContractValidationError(ValueError):
    def __init__(self, error_code: ErrorCode, message: str | None = None):
        self.error_code = error_code
        self.message = message or get_error_message(error_code)

        super().__init__(self.message)
