"""
Step 1 - Data Ingestion & Cleaning (PySpark)
============================================

Run:  python spark_pipeline/s01_ingest_clean.py

For BOTH datasets:
  1. Ingest raw CSV files with an explicit schema (faster + safer than guessing types)
  2. Data-quality checks: row counts, missing values, duplicates, invalid values
  3. Cleaning: drop missing/duplicate/invalid rows, fix types, standardise the label
  4. Save as Parquet (compressed, columnar format used in Big Data systems)
"""

import json
import time

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

import config
from spark_session import get_spark


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def missing_value_report(df: DataFrame) -> dict:
    """Count NULL values in every column in ONE distributed pass."""
    counts = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first().asDict()
    return {column: int(count or 0) for column, count in counts.items() if count}


# --------------------------------------------------------------------------
# Dataset 1 - ULB European cardholders (Time, V1..V28, Amount, Class)
# --------------------------------------------------------------------------
ULB_SCHEMA = StructType(
    [StructField("Time", DoubleType())]
    + [StructField(f"V{i}", DoubleType()) for i in range(1, 29)]
    + [StructField("Amount", DoubleType()), StructField("Class", IntegerType())]
)


def ingest_ulb(spark) -> dict:
    section("DATASET 1: ULB European Credit Card Transactions")
    raw = spark.read.csv(config.spark_path(config.ULB_RAW), header=True, schema=ULB_SCHEMA)

    rows_raw = raw.count()
    missing = missing_value_report(raw)
    print(f"  Raw rows: {rows_raw:,}   columns: {len(raw.columns)}")
    print(f"  Missing values: {missing or 'none'}")

    clean = (
        raw.dropna()
        .filter(F.col("Amount") >= 0)
        .dropDuplicates()
        .withColumnRenamed("Class", config.LABEL)
    )
    rows_clean = clean.count()
    fraud = clean.filter(F.col(config.LABEL) == 1).count()
    print(f"  Removed {rows_raw - rows_clean:,} duplicate/invalid rows -> {rows_clean:,} rows")
    print(f"  Fraud: {fraud:,} ({fraud / rows_clean:.4%})")

    clean.write.mode("overwrite").parquet(config.spark_path(config.ULB_CLEAN))
    print(f"  Saved -> {config.ULB_CLEAN.relative_to(config.PROJECT_ROOT)}")
    return {"rows_raw": rows_raw, "rows_clean": rows_clean, "fraud": fraud, "missing": missing}


# --------------------------------------------------------------------------
# Dataset 2 - Sparkov simulated card transactions (readable columns)
# --------------------------------------------------------------------------
SPARKOV_SCHEMA = StructType(
    [
        StructField("row_index", LongType()),
        StructField("trans_date_trans_time", StringType()),
        StructField("cc_num", LongType()),
        StructField("merchant", StringType()),
        StructField("category", StringType()),
        StructField("amt", DoubleType()),
        StructField("first", StringType()),
        StructField("last", StringType()),
        StructField("gender", StringType()),
        StructField("street", StringType()),
        StructField("city", StringType()),
        StructField("state", StringType()),
        StructField("zip", IntegerType()),
        StructField("lat", DoubleType()),
        StructField("long", DoubleType()),
        StructField("city_pop", LongType()),
        StructField("job", StringType()),
        StructField("dob", StringType()),
        StructField("trans_num", StringType()),
        StructField("unix_time", LongType()),
        StructField("merch_lat", DoubleType()),
        StructField("merch_long", DoubleType()),
        StructField("is_fraud", IntegerType()),
    ]
)

# Personal details we do NOT need for modelling -> removed for privacy (data minimisation).
# (cc_num is kept only as an anonymous card key for per-card features in step 3.)
SPARKOV_PII_COLUMNS = ["row_index", "first", "last", "street", "zip", "trans_num"]


def ingest_sparkov(spark) -> dict:
    section("DATASET 2: Sparkov Simulated Card Transactions")

    # The authors ship a time-based split: train = Jan 2019-Jun 2020, test = Jun-Dec 2020.
    train = spark.read.csv(config.spark_path(config.SPARKOV_TRAIN_RAW), header=True, schema=SPARKOV_SCHEMA)
    test = spark.read.csv(config.spark_path(config.SPARKOV_TEST_RAW), header=True, schema=SPARKOV_SCHEMA)
    raw = train.withColumn("split", F.lit("train")).unionByName(test.withColumn("split", F.lit("test")))

    rows_raw = raw.count()
    missing = missing_value_report(raw)
    print(f"  Raw rows: {rows_raw:,}   columns: {len(raw.columns)}   (train + test files)")
    print(f"  Missing values: {missing or 'none'}")

    duplicate_ids = rows_raw - raw.select("trans_num").distinct().count()
    print(f"  Duplicate transaction ids: {duplicate_ids}")

    clean = (
        raw.dropna(subset=["trans_date_trans_time", "cc_num", "category", "amt", "is_fraud"])
        .dropDuplicates(["trans_num"])
        .filter(F.col("amt") > 0)
        # Types: text -> timestamp / date
        .withColumn("trans_ts", F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("dob", F.to_date("dob", "yyyy-MM-dd"))
        # Tidy text columns
        .withColumn("merchant", F.regexp_replace("merchant", "^fraud_", ""))
        .withColumn("category", F.lower(F.trim("category")))
        .withColumn("gender", F.upper(F.trim("gender")))
        .withColumnRenamed("is_fraud", config.LABEL)
        .drop("trans_date_trans_time", *SPARKOV_PII_COLUMNS)
    )

    rows_clean = clean.count()
    fraud = clean.filter(F.col(config.LABEL) == 1).count()
    print(f"  Removed {rows_raw - rows_clean:,} duplicate/invalid rows -> {rows_clean:,} rows")
    print(f"  Dropped personal columns (privacy): {', '.join(SPARKOV_PII_COLUMNS)}")
    print(f"  Fraud: {fraud:,} ({fraud / rows_clean:.4%})")
    clean.groupBy("split").agg(F.count("*").alias("rows"), F.sum(config.LABEL).alias("fraud")).show()

    clean.write.mode("overwrite").parquet(config.spark_path(config.SPARKOV_CLEAN))
    print(f"  Saved -> {config.SPARKOV_CLEAN.relative_to(config.PROJECT_ROOT)}")
    return {"rows_raw": rows_raw, "rows_clean": rows_clean, "fraud": fraud, "missing": missing}


def main() -> None:
    start = time.time()
    config.ensure_dirs()
    spark = get_spark("CreditGuard-Ingest")
    print(f"Spark {spark.version} | cores: {spark.sparkContext.defaultParallelism}")

    summary = {"ulb": ingest_ulb(spark), "sparkov": ingest_sparkov(spark)}
    (config.RESULTS_DIR / "ingest_summary.json").write_text(json.dumps(summary, indent=2))

    spark.stop()
    print(f"\nIngestion finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
