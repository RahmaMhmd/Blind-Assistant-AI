from pathlib import Path


class MobileNetAdapter:
    """Classify an image using a MobileNetV3 model from Hugging Face."""

    def __init__(
        self,
        name: str,
        task: str,
        model_id: str = "timm/mobilenetv3_large_100.ra_in1k",
    ):
        from transformers import pipeline

        self.name = name
        self.task = task
        self._pipe = pipeline("image-classification", model=model_id)

    def predict(self, sample_path: Path) -> str:
        results = self._pipe(str(sample_path))
        return str(results[0]["label"]).strip()
