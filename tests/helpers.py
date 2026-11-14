import base64

from src.contracts.image_rules import IMAGE_SIGNATURES

JPEG_MIME = "image/jpeg"
PNG_MIME = "image/png"
JPEG_BYTES = IMAGE_SIGNATURES[JPEG_MIME] + b"sample"
PNG_BYTES = IMAGE_SIGNATURES[PNG_MIME] + b"sample"


def data_uri(mime: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")

    return f"data:{mime};base64,{encoded}"


def image_body(
    mime: str = JPEG_MIME,
    content: bytes = JPEG_BYTES,
) -> dict[str, str]:
    return {"imageBase64": data_uri(mime, content)}


def sized_jpeg_body(size: int) -> dict[str, str]:
    head = IMAGE_SIGNATURES[JPEG_MIME]
    padding = b"0" * max(0, size - len(head))

    return image_body(JPEG_MIME, head + padding)
