import os, mlflow

MLFLOW_TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
THRESHOLD_RMSE  = float(os.getenv("RMSE_THRESHOLD", "0.30"))

def _latest_run(metric="rmse"):
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING)
    exp = client.get_experiment_by_name("charges_gbt_experiment")
    runs = client.search_runs(experiment_ids=[exp.experiment_id],
                              order_by=[f"metrics.{metric} ASC"],
                              max_results=1)
    return runs[0]

def should_register(**context):
    run = _latest_run()
    rmse = run.data.metrics["rmse"]
    if rmse <= THRESHOLD_RMSE:
        context["ti"].xcom_push(key="run_id", value=run.info.run_id)
        return "register_model"
    return "notify_team"
