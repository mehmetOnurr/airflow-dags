import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, VectorAssembler,
                                StandardScaler, Interaction, PowerTransformer)
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark

PG_URL = os.getenv("PG_JDBC_URL")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PW   = os.getenv("PG_PW",   "postgres")
MLFLOW_TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "charges_gbt_experiment"

spark = SparkSession.builder.appName("TrainGBT").getOrCreate()
mlflow.set_tracking_uri(MLFLOW_TRACKING)
mlflow.set_experiment(EXPERIMENT_NAME)

df = (
    spark.read.format("jdbc")
         .option("url", PG_URL)
         .option("dbtable", "sales_clean")
         .option("user", PG_USER)
         .option("password", PG_PW)
         .load()
)

# ---------------- Feature Pipeline (özet) ----------------
categorical = ["sex", "smoker", "region", "income_group"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                          handleInvalid="keep") for c in categorical]
encoders = [OneHotEncoder(inputCol=f"{c}_idx",
                          outputCol=f"{c}_ohe") for c in categorical]
interaction = Interaction(inputCols=["bmi", "smoker_ohe"],
                          outputCol="bmi_smoker_inter")
power = PowerTransformer(inputCols=["age","bmi","children"],
                         outputCols=["age_pow","bmi_pow","children_pow"],
                         power=0.4)
assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical] +
              ["age_pow","bmi_pow","children_pow","bmi_smoker_inter"],
    outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw",outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="charges_log",
                   maxIter=200, maxDepth=5, stepSize=0.05)

pipeline = Pipeline(stages=indexers+encoders+[interaction,power,assembler,scaler,gbt])

train, test = df.randomSplit([0.8,0.2], seed=42)
with mlflow.start_run() as run:
    model = pipeline.fit(train)
    preds = model.transform(test)

    evaluator = RegressionEvaluator(labelCol="charges_log",
                                    predictionCol="prediction",
                                    metricName="rmse")
    rmse = evaluator.evaluate(preds)
    r2   = evaluator.setMetricName("r2").evaluate(preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2",   r2)
    mlflow.spark.log_model(model, "model",
                           registered_model_name="ChargesGBT")

    print(f"Logged run_id={run.info.run_id}  RMSE={rmse:.3f}  R2={r2:.3f}")
