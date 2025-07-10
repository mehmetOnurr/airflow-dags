import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import joblib

data = pd.DataFrame({
    "feature": [1, 2, 3, 4],
    "label": [0, 0, 1, 1]
})

X = data[["feature"]]
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("SalesPrediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, "/opt/airflow/dags/model.pkl")
