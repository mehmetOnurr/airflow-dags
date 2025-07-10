import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow

model = joblib.load("/opt/airflow/dags/model.pkl")
X_test = np.array([[1], [2], [3], [4]])
y_test = [0, 0, 1, 1]

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

mlflow.set_tracking_uri("http://mlflow:5000")
with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    if acc >= 0.9:
        with open("/opt/airflow/dags/data/accuracy_ok.txt", "w") as f:
            f.write("OK")
