import argparse
import time
import os
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
from pyspark.sql.window import Window

def main(args):
    spark = (
        SparkSession.builder
        .appName("NYC Taxi Benchmark")
        .config("spark.executor.instances", str(args.executor_instances))
        .config("spark.executor.cores", str(args.executor_cores))
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", str(args.aqe).lower())
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

    start_time = time.time()

    # Full list of all expected columns including any extras
    all_columns = [
        "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
        "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID",
        "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
        "improvement_surcharge", "total_amount", "congestion_surcharge", "Airport_fee"
    ]

    from google.cloud import storage
    from pyspark.sql.functions import lit
    from pyspark.sql.types import LongType, IntegerType, DoubleType, StringType, TimestampType

    client = storage.Client()
    bucket = client.bucket("egd-europewest-1")
    blobs = list(bucket.list_blobs(prefix="yellow_tripdata/", delimiter=None))
    files = [f"gs://egd-europewest-1/{blob.name}" for blob in blobs if blob.name.endswith(".parquet")]

    def unify_columns(df, all_cols):
        missing_cols = set(all_cols) - set(df.columns)
        for c in missing_cols:
            df = df.withColumn(c, lit(None))
        return df.select(all_cols)

    dfs = []
    for file_path in files:
        df_tmp = spark.read.parquet(file_path)

        # Cast columns to expected types to avoid conflicts
        df_tmp = df_tmp.withColumn("VendorID", col("VendorID").cast(LongType()))
        df_tmp = df_tmp.withColumn("passenger_count", col("passenger_count").cast(IntegerType()))
        df_tmp = df_tmp.withColumn("trip_distance", col("trip_distance").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("RatecodeID", col("RatecodeID").cast(IntegerType()))
        df_tmp = df_tmp.withColumn("store_and_fwd_flag", col("store_and_fwd_flag").cast(StringType()))
        df_tmp = df_tmp.withColumn("PULocationID", col("PULocationID").cast(IntegerType()))
        df_tmp = df_tmp.withColumn("DOLocationID", col("DOLocationID").cast(IntegerType()))
        df_tmp = df_tmp.withColumn("payment_type", col("payment_type").cast(IntegerType()))
        df_tmp = df_tmp.withColumn("fare_amount", col("fare_amount").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("extra", col("extra").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("mta_tax", col("mta_tax").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("tip_amount", col("tip_amount").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("tolls_amount", col("tolls_amount").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("improvement_surcharge", col("improvement_surcharge").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("total_amount", col("total_amount").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("congestion_surcharge", col("congestion_surcharge").cast(DoubleType()))
        df_tmp = df_tmp.withColumn("Airport_fee", col("Airport_fee").cast(DoubleType()))

        # Add missing columns as nulls and reorder columns
        df_tmp = unify_columns(df_tmp, all_columns)

        dfs.append(df_tmp)

    # Union all DataFrames
    df = dfs[0]
    for df_part in dfs[1:]:
        df = df.unionByName(df_part)

    # Continue with your preprocessing and pipeline as before
    df = (
        df
        .withColumn("tpep_pickup_datetime", F.to_timestamp("tpep_pickup_datetime"))
        .withColumn("tpep_dropoff_datetime", F.to_timestamp("tpep_dropoff_datetime"))
        .filter(F.col("tpep_pickup_datetime").between("2023-02-01", "2025-02-01"))
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("total_amount") > 0)
        .withColumn('pickup_hour', F.hour('tpep_pickup_datetime'))
        .withColumn('pickup_month', F.month('tpep_pickup_datetime'))
        .withColumn('trip_duration', F.unix_timestamp('tpep_dropoff_datetime') - F.unix_timestamp('tpep_pickup_datetime'))
        .withColumn('total_amount_per_minute', F.col('total_amount') / (F.col('trip_duration') / 60))
        .withColumn('pickup_day_of_week', F.date_format('tpep_pickup_datetime', 'u'))
    )

    # from pyspark.sql.functions import expr

    # # Simulate varied weekdays from 1 to 7
    # df = df.withColumn("pickup_day_of_week", (F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) % 7 + 1).cast("string"))


    sample_df = df.limit(10000000)
    train_df, test_df = sample_df.randomSplit([0.8, 0.2], seed=42)
    #train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import RegressionEvaluator

    # Define indexers, encoders, and assembler like before
    indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in ['PULocationID', 'DOLocationID', 'pickup_day_of_week']
    ]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in ['PULocationID', 'DOLocationID', 'pickup_day_of_week']]
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in ['PULocationID', 'DOLocationID', 'pickup_day_of_week']] + ['pickup_hour', 'pickup_month'],
        outputCol="features"
    )

    # Define Random Forest with fixed hyperparameters (comparable setup)
    rf = RandomForestRegressor(
        labelCol='total_amount',
        featuresCol='features',
        numTrees=20,     # fixed to reduce variability
        maxDepth=10      # fixed to match LinearReg simplicity
    )

    # Build pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    # Fit and predict
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    # Evaluate
    evaluator = RegressionEvaluator(labelCol='total_amount', predictionCol='prediction')
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: 'rmse'})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})

    duration = round(time.time() - start_time, 2)
    print(f"Execution Time: {duration:.2f}s | RMSE: {rmse:.4f} | R²: {r2:.4f}")





    # Save log
    os.makedirs("logs", exist_ok=True)
    import csv
    import tempfile
    from google.cloud import storage

    # Collect result values
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, args.executor_instances, args.executor_cores, args.executor_memory,
        args.shuffle_partitions, args.aqe, duration, round(rmse, 4), round(r2, 4)]

    # Step 1: Write row to local temp file
    temp_path = "/tmp/results.csv"
    with open(temp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "executor_instances", "executor_cores", "executor_memory",
                        "shuffle_partitions", "aqe", "duration", "rmse", "r2"])  # Header
        writer.writerow(row)

    # Step 2: Upload to GCS
    bucket_name = "egd-europewest-1"
    gcs_file_path = "logs/results.csv"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)

    # Download existing data if present
    if blob.exists():
        print("Appending to existing GCS CSV...")
        existing = blob.download_as_text()
        new = existing.strip() + "\n" + ",".join(map(str, row))
        blob.upload_from_string(new)
    else:
        print("Creating new GCS CSV...")
        blob.upload_from_filename(temp_path)

    print(f"✅ Log written to gs://{bucket_name}/{gcs_file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor_instances", type=int, default=2)
    parser.add_argument("--executor_cores", type=int, default=1)
    parser.add_argument("--executor_memory", type=str, default="3g")
    parser.add_argument("--shuffle_partitions", type=int, default=50)
    parser.add_argument("--aqe", type=bool, default=False)
    args = parser.parse_args()
    main(args)