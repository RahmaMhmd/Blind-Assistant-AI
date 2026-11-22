from unittest.mock import MagicMock

from experiments.constants import EXPERIMENT_NAME, TRACKING_URI
from experiments.tracking import log_run


def test_logs_params_and_metrics_inside_one_run():
    mlflow = MagicMock()

    log_run("run-1", {"dataset": "demo"}, {"recall": 0.9}, mlflow_module=mlflow)

    mlflow.set_tracking_uri.assert_called_once_with(TRACKING_URI)
    mlflow.set_experiment.assert_called_once_with(EXPERIMENT_NAME)
    mlflow.start_run.assert_called_once_with(run_name="run-1")
    mlflow.log_params.assert_called_once_with({"dataset": "demo"})
    mlflow.log_metrics.assert_called_once_with({"recall": 0.9})
