from contextlib import contextmanager
from types import SimpleNamespace

from src.evaluation.config import load_config
from src.evaluation.tracking import SAMPLES_FILE, log_evaluation

RUN_ID = "run-1"
SUMMARY = {
    "model": "whisper-small-eg",
    "testset": "audio_test",
    "task": "speech_recognition",
    "samples": 2,
    "wer": 0.5,
    "mean_seconds": 1.5,
}
RECORDS = [{"id": "a", "wer": 0.0}, {"id": "b", "wer": 1.0}]


class RecordingMlflow:
    """Stands in for MLflow and remembers what it was asked to save."""

    def __init__(self):
        self.params = self.metrics = self.saved = self.run_name = self.tag = None

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self, run_name):
        self.run_name = run_name
        yield SimpleNamespace(info=SimpleNamespace(run_id=RUN_ID))

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics):
        self.metrics = metrics

    def set_tag(self, key, value):
        self.tag = (key, value)

    def log_dict(self, data, file_name):
        self.saved = (data, file_name)


def test_run_is_saved_with_settings_numbers_tag_and_samples():
    fake = RecordingMlflow()
    settings = load_config()["tracking"]
    run_id = log_evaluation(SUMMARY, RECORDS, mlflow=fake)

    assert run_id == RUN_ID
    assert fake.experiment == settings["experiment"]
    assert fake.uri.startswith("sqlite:///") and fake.uri.endswith(settings["database"])
    assert fake.run_name == "whisper-small-eg - audio_test"
    assert fake.tag == ("run_type", settings["run_type"])
    assert fake.metrics == {"wer": 0.5, "mean_seconds": 1.5}
    assert fake.params["samples"] == 2 and fake.params["adapter"] == "whisper"
    assert fake.params["param.model_id"].endswith("whisper-small-eg")
    assert fake.saved == (RECORDS, SAMPLES_FILE)


def test_samples_file_is_optional():
    fake = RecordingMlflow()
    log_evaluation(SUMMARY, mlflow=fake)

    assert fake.saved is None
