from types import ModuleType

from .constants import EXPERIMENT_NAME, TRACKING_URI


def _load_mlflow() -> ModuleType:
    import mlflow

    return mlflow


def log_run(
    run_name: str,
    params: dict[str, str | int | float],
    metrics: dict[str, float],
    mlflow_module: ModuleType | None = None,
) -> None:
    mlflow = mlflow_module or _load_mlflow()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
