from .config import REPO_ROOT, load_config

TRACKING_SECTION = "tracking"
SAMPLES_FILE = "samples.json"
PARAM_PREFIX = "param."
RUN_TYPE_TAG = "run_type"


def _tracking_uri(settings: dict) -> str:
    return f"sqlite:///{(REPO_ROOT / settings['database']).as_posix()}"


def _params(summary: dict) -> dict:
    """What was run: model, test set, task, and the model's own settings from the config."""
    spec = load_config()["models"][summary["model"]]
    params = {
        "model": summary["model"],
        "testset": summary["testset"],
        "task": summary["task"],
        "samples": summary["samples"],
        "adapter": spec["adapter"],
    }
    params.update({f"{PARAM_PREFIX}{key}": str(value) for key, value in spec.get("params", {}).items()})

    return params


def _metrics(summary: dict) -> dict:
    return {
        key: float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "samples"
    }


def log_evaluation(summary: dict, records: list[dict] | None = None, mlflow=None) -> str:
    """Save one evaluation in MLflow and return the run id."""
    if mlflow is None:
        import mlflow  # imported here so the tool still works without MLflow when not tracking

    settings = load_config()[TRACKING_SECTION]
    mlflow.set_tracking_uri(_tracking_uri(settings))
    mlflow.set_experiment(settings["experiment"])

    with mlflow.start_run(run_name=f"{summary['model']} - {summary['testset']}") as run:
        mlflow.log_params(_params(summary))
        mlflow.log_metrics(_metrics(summary))
        mlflow.set_tag(RUN_TYPE_TAG, settings["run_type"])

        if records is not None:
            mlflow.log_dict(records, SAMPLES_FILE)

        return run.info.run_id
