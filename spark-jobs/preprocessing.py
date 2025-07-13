import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

POSTGRES_URL = os.getenv("PG_JDBC_URL")
POSTGRES_USER = os.getenv("PG_USER", "postgres")
POSTGRES_PW   = os.getenv("PG_PW",   "postgres")
TARGET_TABLE  = "sales_clean"

spark = (
    SparkSession.builder.appName("KafkaPreprocess")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

###############################################################################
# 1) Kafka kaynağı
###############################################################################
schema = (
    StructType()
      .add("age", IntegerType())
      .add("sex", StringType())
      .add("bmi", DoubleType())
      .add("children", IntegerType())
      .add("smoker", StringType())
      .add("region", StringType())
      .add("charges", DoubleType())
)

raw = (
    spark.readStream.format("kafka")
         .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP"))
         .option("subscribe", "realtime-sales")
         .option("startingOffsets", "latest")
         .load()
)

json_df = raw.selectExpr("CAST(value AS STRING) as json") \
             .select(F.from_json("json", schema).alias("data")) \
             .select("data.*")

###############################################################################
# 2) Temizlik + Basit Feature Engineering
###############################################################################
clean = (
    json_df.dropDuplicates()
           .na.drop()
           .withColumn("charges_log", F.log1p("charges"))
           .withColumn("income_group",
                       F.when(F.col("charges") < 1e4, "low")
                        .when(F.col("charges") < 2e4, "mid")
                        .otherwise("high"))
)

###############################################################################
# 3) Postgres’e upsert (micro-batch)
###############################################################################
def foreach_batch(df, epoch_id):
    (df.write.format("jdbc")
        .option("url", POSTGRES_URL)
        .option("dbtable", TARGET_TABLE)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PW)
        .option("driver", "org.postgresql.Driver")
        .mode("append")          # idempotentlik için PK/ON CONFLICT kur
        .save())

(clean.writeStream
      .foreachBatch(foreach_batch)
      .outputMode("update")
      .option("checkpointLocation", "/chk/kafka_preprocess")
      .start()
      .awaitTermination())
