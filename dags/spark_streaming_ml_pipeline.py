from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import random

default_args = {"start_date": datetime(2024, 1, 1)}

def should_register():
    # Gerçek durumda burada accuracy karşılaştırması olur
    return "register_model" if random.random() > 0.5 else "notify_team"

with DAG(
    dag_id="spark_streaming_ml_pipeline",
    schedule_interval="@once",
    default_args=default_args,
    catchup=False
) as dag:

    create_kafka_topic = BashOperator(
        task_id="create_kafka_topic",
        bash_command="kubectl exec -it $(kubectl get pods -l app=kafka -o jsonpath='{.items[0].metadata.name}') -- bash -c \"/opt/kafka/bin/kafka-topics.sh --create --topic realtime-sales --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1\""
    )

    wait_for_trigger = FileSensor(
        task_id="wait_for_trigger",
        filepath="/opt/airflow/dags/data/trigger.txt",  # örnek tetik dosyası
        poke_interval=10,
        timeout=600
    )

    run_spark_job = BashOperator(
        task_id="run_spark_job",
        bash_command="spark-submit --master spark://spark-master:7077 /opt/airflow/dags/streaming.py"
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python3 /opt/airflow/dags/train_model.py"
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="python3 /opt/airflow/dags/evaluate_model.py"
    )

    check_accuracy = BranchPythonOperator(
        task_id="check_accuracy",
        python_callable=should_register
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python3 /opt/airflow/dags/register_model.py"
    )

    notify_team = EmailOperator(
        task_id="notify_team",
        to="data-team@example.com",
        subject="Model Eğitimi Tamamlandı",
        html_content="Model başarıyla eğitildi ya da kayıt edilmedi. Kontrol edin."
    )

    create_kafka_topic >> wait_for_trigger >> run_spark_job >> train_model >> evaluate_model >> check_accuracy
    check_accuracy >> [register_model, notify_team]
