import re

DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024

IMAGE_SIGNATURES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
}

SUPPORTED_IMAGE_MIME_TYPES = frozenset(IMAGE_SIGNATURES)

DATA_URI_PATTERN = re.compile(
    rf"^data:({'|'.join(map(re.escape, IMAGE_SIGNATURES))});base64,(.+)$",
    re.DOTALL,
)
