import numpy as np
import pytest
soundfile = pytest.importorskip("soundfile")
from src.evaluation.adapters.audio import load_mono  # noqa: E402
SOURCE_RATE = 8000
TARGET_RATE = 16000
SECONDS = 1
TONE_HZ = 440
def _write_stereo_wav(path):
    time = np.arange(SOURCE_RATE * SECONDS) / SOURCE_RATE
    tone = np.sin(2 * np.pi * TONE_HZ * time).astype("float32")
    soundfile.write(str(path), np.stack([tone, tone], axis=1), SOURCE_RATE)
def test_stereo_is_mixed_to_one_channel_and_resampled(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_stereo_wav(wav)
    samples = load_mono(wav, TARGET_RATE)

    assert samples.ndim == 1
    assert len(samples) == TARGET_RATE * SECONDS
