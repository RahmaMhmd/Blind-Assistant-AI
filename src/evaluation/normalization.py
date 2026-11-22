import re
from functools import lru_cache
from .config import load_config
CONFIG_SECTION = "normalization"
DEFAULT_PROFILE = "default"
PROFILES_KEY = "profiles"
class ArabicNormalizer:
    """Make Arabic and English text comparable before measuring errors."""
    def __init__(self, rules: dict):
        self._char_map = str.maketrans(rules["char_map"])
        self._digit_map = str.maketrans(rules["digits_from"], rules["digits_to"])
        self._remove = re.compile(rules["remove_pattern"])
        self._lowercase = rules["lowercase"]
        pattern = rules["punctuation_pattern"]
        self._punctuation = re.compile(pattern) if pattern else None

    def __call__(self, text: str) -> str:
        text = self._remove.sub("", text)
        text = text.translate(self._char_map).translate(self._digit_map)

        if self._punctuation:
            text = self._punctuation.sub(" ", text)

        if self._lowercase:
            text = text.lower()
        return " ".join(text.split())

def _rules_for(config: dict, profile: str) -> dict:
    overrides = config.get(PROFILES_KEY, {})

    if profile != DEFAULT_PROFILE and profile not in overrides:
        raise ValueError(f"unknown normalization profile: {profile}")

    return {**config[DEFAULT_PROFILE], **overrides.get(profile, {})}
@lru_cache(maxsize=None)
def get_normalizer(profile: str = DEFAULT_PROFILE) -> ArabicNormalizer:
    return ArabicNormalizer(_rules_for(load_config()[CONFIG_SECTION], profile))
