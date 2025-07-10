import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.register_model(
    model_uri="runs:/<your-run-id>/model",
    name="SalesClassifier"
)
