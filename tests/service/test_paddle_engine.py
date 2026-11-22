import os
from pathlib import Path

import pytest

SMOKE_IMAGE_ENV = "AI_PADDLE_SMOKE_IMAGE"


@pytest.mark.skipif(
    SMOKE_IMAGE_ENV not in os.environ,
    reason="set AI_PADDLE_SMOKE_IMAGE to an image with text",
)
def test_paddle_engine_reads_text_from_real_weights():
    from src.service.handlers.paddle_engine import PaddleOcrEngine

    lines = PaddleOcrEngine().read(Path(os.environ[SMOKE_IMAGE_ENV]).read_bytes())

    assert lines
    assert all(line.text and 0 <= line.confidence <= 1 for line in lines)
