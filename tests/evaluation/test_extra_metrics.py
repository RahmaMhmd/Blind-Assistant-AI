import pytest

from src.evaluation.config import load_config
from src.evaluation.metrics.keyword import keyword_metrics
from src.evaluation.metrics.scene import scene_metrics


def test_keyword_metrics_recall_and_false_alarm():
    refs = ["افتح الكاميرا", "أفتح الكاميرا", "", ""]
    hyps = ["افتح الكاميرا", "افتح الكاميره", "", "افتح الكاميرا"]
    scores = keyword_metrics(refs, hyps)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["false_alarm_rate"] == pytest.approx(0.5)


def test_scene_metrics_arabic_token_f1_and_exact():
    refs = ["يوجد كرسي أبيض أمامك", "لا يوجد عائق"]
    hyps = ["يوجد كرسي ابيض", "لا يوجد عائق"]
    scores = scene_metrics(refs, hyps)
    assert scores["exact_match"] == pytest.approx(0.5)
    assert scores["token_f1"] > 0.8


def test_modular_configs_loaded():
    models = load_config()["models"]
    assert {"rtdetrv2-r18-int8", "dfine-small-int8", "mobilenetv3-large-int8"} <= set(models)
    assert {"whisper-large-v3-turbo", "sherpa-kws-default", "qwen25-vl-scene"} <= set(models)
