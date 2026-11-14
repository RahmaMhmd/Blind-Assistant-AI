from pydantic import BaseModel, ConfigDict

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_DISTANCE = 0.0
SUCCESS_STATUS_MIN = 200
SUCCESS_STATUS_MAX = 299
ERROR_STATUS_MIN = 400
ERROR_STATUS_MAX = 599


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContractResponseModel(StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
