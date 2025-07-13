from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime
import os, textwrap

default_args = {"start_date": datetime(2025, 1, 1)}

with DAG(
    dag_id="kafka_spark_mlflow_pipeline",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:

    # 0) (Opsiyonel) Kafka topic’i oluştur
    create_topic = BashOperator(
        task_id="create_kafka_topic",
        bash_command=textwrap.dedent("""
          set -e
          kubectl exec -it $(kubectl get pods -l app=kafka -o jsonpath='{.items[0].metadata.name}') -- \
          /opt/kafka/bin/kafka-topics.sh --create --if-not-exists \
            --topic realtime-sales --bootstrap-server localhost:9092 \
            --replication-factor 1 --partitions 1
        """)
    )

    preprocess = SparkSubmitOperator(
        task_id="spark_preprocess",
        application="/opt/spark_jobs/preprocessing.py",
        name="kafka_preprocess",
        conn_id="spark_k8s",          # Airflow Connection tanımı
        conf={
            "spark.kubernetes.container.image": "my-spark:latest",
            "spark.executor.instances": "2",
        },
        env_vars={
            "KAFKA_BOOTSTRAP": os.getenv("KAFKA_BOOTSTRAP"),
            "PG_JDBC_URL": os.getenv("PG_JDBC_URL"),
            "PG_USER": os.getenv("PG_USER"),
            "PG_PW": os.getenv("PG_PW"),
        },
        verbose=True,
    )

    train = SparkSubmitOperator(
        task_id="spark_train",
        application="/opt/spark_jobs/training.py",
        name="train_gbt",
        conn_id="spark_k8s",
        conf={"spark.kubernetes.container.image": "my-spark:latest"},
        env_vars={
            "PG_JDBC_URL": os.getenv("PG_JDBC_URL"),
            "PG_USER": os.getenv("PG_USER"),
            "PG_PW": os.getenv("PG_PW"),
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        },
        verbose=True,
    )

    check_eval = BranchPythonOperator(
        task_id="check_evaluation",
        python_callable=__import__("scripts.check_evaluation").check_evaluation.should_register,
        provide_context=True,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="echo '✔️  Model approved & registered!'"
    )

    notify_team = EmailOperator(
        task_id="notify_team",
        to="ml-team@example.com",
        subject="Model retraining failed threshold",
        html_content="Yeni run istenen RMSE eşiğini geçemedi.",
    )

    create_topic >> preprocess >> train >> check_eval
    check_eval >> [register_model, notify_team]
