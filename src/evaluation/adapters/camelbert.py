from pathlib import Path


class CamelBertAdapter:
    """Classify Arabic voice command intent with CAMeLBERT."""

    def __init__(
        self,
        name: str,
        task: str,
        model_id: str = "CAMeL-Lab/bert-base-arabic-camelbert-da",
        intents: list[str] | None = None,
    ):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        self.name = name
        self.task = task
        self._intents = intents or [
            "READ_DOCUMENT",
            "RECOGNIZE_CURRENCY",
            "INDOOR_NAVIGATION",
            "DESCRIBE_SCENE",
            "DETECT_COLOR",
            "READ_TRAFFIC_SIGN",
            "DETECT_OBSTACLE",
            "DETECT_EXPRESSION",
        ]
        id2label = {i: label for i, label in enumerate(self._intents)}
        label2id = {label: i for i, label in enumerate(self._intents)}

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=len(self._intents),
            id2label=id2label,
            label2id=label2id,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(self, sample_path: Path) -> str:
        text = sample_path.read_text(encoding="utf-8").strip()
        result = self._pipe(text)
        return str(result[0]["label"]).strip()
