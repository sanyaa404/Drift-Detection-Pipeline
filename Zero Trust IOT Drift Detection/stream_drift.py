from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

schema = StructType([
    StructField("mean_intensity", DoubleType(), True),
    StructField("contrast", DoubleType(), True),
    StructField("sharpness", DoubleType(), True),
    StructField("noise_level", DoubleType(), True),
    StructField("label", StringType(), True),
])

FEATURES = ["mean_intensity", "contrast", "sharpness", "noise_level"]

print("🚀 Starting Spark Streaming...")

spark = SparkSession.builder \
    .appName("IoT Streaming Drift Detection") \
    .getOrCreate()

# Read streaming data
stream_df = spark.readStream \
    .option("header", True) \
    .schema(schema) \
    .csv("data/stream")

def process_batch(batch_df, batch_id):
    print(f"\n📡 Batch {batch_id}")

    if batch_df.count() == 0:
        return

    pdf = batch_df.toPandas()

    print(f"Rows: {len(pdf)}")

    # Simple drift check
    for col in FEATURES:
        print(f"{col}: {pdf[col].mean():.2f}")

    # 🚨 Alert condition
    if pdf["noise_level"].mean() > 4000:
        print("🚨 ALERT: Drift detected (high noise)")

query = stream_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .start()

query.awaitTermination()