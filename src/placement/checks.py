from src.contracts import InferenceFeature

from .enums import Commercial, Kind, Status, Usage
from .models import FeatureEntry, ModelRef

EXTRA_FEATURES = frozenset({
    "image_quality",
    "qr_barcode",
    "object_detection",
    "depth_estimation",
    "object_tracking",
    "speech_recognition",
    "keyword_spotter",
})
KNOWN_FEATURES = frozenset(f.value for f in InferenceFeature) | EXTRA_FEATURES


def _ref_problems(where: str, ref: ModelRef, confirmed: bool) -> list[str]:
    problems = []
    blocked = ref.license.commercial_use is Commercial.BLOCKED

    if ref.usage is Usage.PRODUCTION and blocked:
        problems.append(f"{where} production use with blocked license")

    if confirmed:
        needed = {
            "version": ref.version,
            "license verified date": ref.license.verified_on,
        }
        if ref.kind is Kind.MODEL:
            needed.update({"sha256": ref.sha256, "size": ref.size_mb})
        problems += [
            f"{where} confirmed without {key}" for key, v in needed.items() if v is None
        ]
        if ref.usage is not Usage.PRODUCTION:
            problems.append(f"{where} confirmed but not production")
        if ref.license.commercial_use is not Commercial.ALLOWED:
            problems.append(f"{where} confirmed without allowed license")

    return problems


def entry_problems(feature: str, entry: FeatureEntry) -> list[str]:
    """rules that json schema cannot express"""
    confirmed = entry.status is Status.CONFIRMED
    refs = {"primary": entry.primary, "fallback": entry.fallback}
    problems = [] if feature in KNOWN_FEATURES else [f"{feature} unknown feature"]

    for role, ref in refs.items():
        if ref:
            problems += _ref_problems(f"{feature}.{role}", ref, confirmed)

    if entry.fallback and entry.fallback.name == entry.primary.name:
        problems.append(f"{feature} primary and fallback are the same")

    used = {ref.name for ref in refs.values() if ref}
    problems += [
        f"{feature} lists {c.name} twice" for c in entry.considered if c.name in used
    ]

    return problems
