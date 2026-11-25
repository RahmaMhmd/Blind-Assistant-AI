from pathlib import Path

from src.quantization.config import resolve_path

from .audio import load_mono


class KeywordSpotterAdapter:
    """run sherpa onnx wake word spotter on wav audio"""

    def __init__(
        self,
        name: str,
        task: str,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        keywords_file: str,
        sampling_rate: int = 16000,
    ):
        import sherpa_onnx

        self.name = name
        self.task = task
        self._rate = int(sampling_rate)
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(resolve_path(tokens)),
            encoder=str(resolve_path(encoder)),
            decoder=str(resolve_path(decoder)),
            joiner=str(resolve_path(joiner)),
            keywords_file=str(resolve_path(keywords_file)),
            num_threads=2,
            sample_rate=self._rate,
        )

    def predict(self, sample_path: Path) -> str:
        audio = load_mono(sample_path, self._rate)
        stream = self._spotter.create_stream()
        stream.accept_waveform(self._rate, audio)
        stream.input_finished()
        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
        return self._spotter.get_result(stream).strip()
