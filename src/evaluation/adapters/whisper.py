from pathlib import Path

from .audio import load_mono

TRANSCRIBE = "transcribe"


class WhisperAdapter:
    """Run a Whisper model from Hugging Face on one audio file, return the text."""

    def __init__(
        self,
        name: str,
        task: str,
        model_id: str,
        language: str,
        sampling_rate: int,
        chunk_length_s: int | None = None,
        generate_kwargs: dict | None = None,
    ):
        from transformers import pipeline  # heavy, loaded only for this model

        self.name = name
        self.task = task
        self._rate = sampling_rate
        self._generate_kwargs = {
            "language": language,
            "task": TRANSCRIBE,
            **(generate_kwargs or {}),
        }
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=chunk_length_s,
        )

    def predict(self, sample_path: Path) -> str:
        audio = load_mono(sample_path, self._rate)
        result = self._pipe(
            {"raw": audio, "sampling_rate": self._rate},
            generate_kwargs=self._generate_kwargs,
        )
        return result["text"].strip()
