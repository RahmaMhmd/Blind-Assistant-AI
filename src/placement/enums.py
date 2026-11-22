from enum import Enum


class Placement(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class Status(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


class Usage(str, Enum):
    PRODUCTION = "production"
    RESEARCH_ONLY = "research_only"
    BASELINE_ONLY = "baseline_only"


class Commercial(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    UNVERIFIED = "unverified"


class Kind(str, Enum):
    MODEL = "model"
    ALGORITHM = "algorithm"


class Runtime(str, Enum):
    MLKIT = "mlkit"
    ONNXRUNTIME = "onnxruntime"
    TFLITE = "tflite"
    OPENCV = "opencv"
    ZXING = "zxing"
    PADDLE = "paddle"
    TRANSFORMERS = "transformers"


class Outcome(str, Enum):
    REJECTED = "rejected"
    RESEARCH_ONLY = "research_only"
    BASELINE = "baseline"
    DEFERRED = "deferred"
