import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Blind Assistant Experiments")

with mlflow.start_run():

    mlflow.log_param("model_name", "Test Model")
    mlflow.log_param("language", "Egyptian Arabic")

    accuracy = 0.85
    processing_time = 1.5

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("processing_time", processing_time)

    print("Experiment logged successfully!")