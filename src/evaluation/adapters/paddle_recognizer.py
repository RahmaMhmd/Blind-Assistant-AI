from pathlib import Path
class PaddleRecognizerAdapter:
    """Read the text in one cropped word image with a PaddleOCR recognition model."""

    def __init__(self, name: str, task: str, model_name: str):
        from paddleocr import TextRecognition 
        self.name = name
        self.task = task
        self._model = TextRecognition(model_name=model_name)
    def predict(self, sample_path: Path) -> str:
        result = next(iter(self._model.predict(input=str(sample_path), batch_size=1)))
        return str(result["rec_text"]).strip()
