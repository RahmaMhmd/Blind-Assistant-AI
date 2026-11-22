import cv2
import numpy as np

from .ocr import OcrLine

DEFAULT_REC_MODEL = "arabic_PP-OCRv5_mobile_rec"
TEXTS_KEY = "rec_texts"
SCORES_KEY = "rec_scores"


class PaddleOcrEngine:
    """adapter for paddleocr 3 and it has not run on real weights yet"""

    def __init__(self, rec_model: str = DEFAULT_REC_MODEL):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def read(self, image: bytes) -> list[OcrLine]:
        frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)

        return [
            OcrLine(text, float(score))
            for page in self._ocr.predict(frame)
            for text, score in zip(page[TEXTS_KEY], page[SCORES_KEY])
        ]
