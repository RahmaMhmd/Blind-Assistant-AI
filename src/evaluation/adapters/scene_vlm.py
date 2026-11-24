from pathlib import Path

from PIL import Image


class SceneVlmAdapter:
    """run vision language model to describe one scene image"""

    def __init__(
        self,
        name: str,
        task: str,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 80,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = name
        self.task = task
        self._prompt = prompt
        self._max_tokens = int(max_new_tokens)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(model_id)
        self._model.eval()

    def predict(self, sample_path: Path) -> str:
        with Image.open(sample_path) as img:
            rgb = img.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self._prompt}],
            }
        ]
        text_in = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[text_in], images=[rgb], return_tensors="pt")
        out_ids = self._model.generate(**inputs, max_new_tokens=self._max_tokens)
        gen = out_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen, skip_special_tokens=True).strip()
