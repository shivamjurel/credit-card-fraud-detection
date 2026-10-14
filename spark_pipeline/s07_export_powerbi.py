"""
Step 7 - Export Analytics Tables for Power BI
=============================================

Run:  python spark_pipeline/s07_export_powerbi.py   (after steps 1-5)

Spark prepares clean, report-ready tables; Power BI Desktop loads them
(Home -> Get data -> Text/CSV) and builds the dashboard. See powerbi/POWERBI_GUIDE.md.

Output folder: powerbi/data/
  fact_transactions.csv      all 1.85M Sparkov transactions with business columns
                             + model fraud score / prediction for the test period
  model_comparison.csv       5 models x 2 datasets, all metrics
  feature_importance.csv     top features of each selected model
  ulb_hourly.csv             Dataset 1 transactions & fraud per hour
  stream_scored.csv          transactions scored by the real-time stream (if the demo was run)
"""

import json
import shutil
import time

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import config
from spark_session import get_spark

OUT = config.POWERBI_DIR


def write_single_csv(df, name: str) -> None:
    """Spark writes a folder of part-files; merge into ONE CSV file for Power BI."""
    temporary = OUT / f"_{name}_tmp"
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(config.spark_path(temporary))
    part = next(temporary.glob("part-*.csv"))
    target = OUT / f"{name}.csv"
    if target.exists():
        target.unlink()
    shutil.move(str(part), target)
    shutil.rmtree(temporary)
    print(f"  {target.name:<24} {target.stat().st_size / 1e6:8.1f} MB")


def export_fact_table(spark) -> None:
    metadata = json.loads((config.SPARK_MODELS_DIR / "sparkov_best.json").read_text())
    model = PipelineModel.load(config.spark_path(config.SPARK_MODELS_DIR / "sparkov_best"))
    features = spark.read.parquet(config.spark_path(config.SPARKOV_FEATURES))

    # Model score is kept only for the TEST period (train rows were seen during training,
    # so their scores would look unrealistically good)
    score_column = "probability" if metadata["has_probability"] else "rawPrediction"
    is_test = F.col("split") == "test"
    raw_score = vector_to_array(score_column)[1]

    fact = (
        model.transform(features)
        .withColumn("fraud_score", F.when(is_test, raw_score))
        .withColumn("predicted_fraud", F.when(is_test, (raw_score >= metadata["threshold"]).cast("int")))
        .withColumn("transaction_id", F.monotonically_increasing_id())
        .select(
            "transaction_id",
            F.date_format("trans_ts", "yyyy-MM-dd HH:mm:ss").alias("transaction_datetime"),
            F.to_date("trans_ts").alias("transaction_date"),
            F.col("hour").cast("int").alias("hour"),
            F.date_format("trans_ts", "EEEE").alias("day_name"),
            F.date_format("trans_ts", "yyyy-MM").alias("year_month"),
            "category", "merchant", F.round("amt", 2).alias("amount"), "gender",
            F.col("age").cast("int").alias("age"),
            F.when(F.col("age") < 25, "18-24").when(F.col("age") < 35, "25-34").when(F.col("age") < 45, "35-44")
            .when(F.col("age") < 55, "45-54").when(F.col("age") < 65, "55-64").otherwise("65+").alias("age_group"),
            "job", "city", "state", "lat", "long", "city_pop",
            F.round("distance_km", 1).alias("distance_km"),
            F.col("txn_count_24h").cast("int").alias("txn_count_24h"),
            F.round("amt_to_avg_ratio", 2).alias("amount_to_card_avg"),
            F.col("is_night").cast("int").alias("is_night"),
            F.col(config.LABEL).alias("is_fraud"),
            F.initcap("split").alias("data_split"),
            F.round("fraud_score", 4).alias("fraud_score"),
            "predicted_fraud",
            "card_id",
        )
    )
    write_single_csv(fact, "fact_transactions")


def export_small_tables(spark) -> None:
    comparison = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv")
    comparison["dataset_name"] = comparison["dataset"].map({k: v["name"] for k, v in config.DATASETS.items()})
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    print(f"  {'model_comparison.csv':<24} {len(comparison)} rows")

    rows = []
    for dataset in config.DATASETS:
        metadata = json.loads((config.SPARK_MODELS_DIR / f"{dataset}_best.json").read_text())
        for rank, (feature, importance) in enumerate(metadata.get("feature_importances", {}).items(), start=1):
            rows.append({"dataset": dataset, "model": metadata["selected_model"], "rank": rank,
                         "feature": feature, "importance": importance})
    pd.DataFrame(rows).to_csv(OUT / "feature_importance.csv", index=False)
    print(f"  {'feature_importance.csv':<24} {len(rows)} rows")

    ulb_hourly = (
        spark.read.parquet(config.spark_path(config.ULB_FEATURES))
        .groupBy(F.col("hour_of_day").cast("int").alias("hour"))
        .agg(F.count("*").alias("transactions"), F.sum(config.LABEL).alias("fraud"),
             F.round(F.sum("Amount"), 2).alias("amount"))
        .orderBy("hour")
        .toPandas()
    )
    ulb_hourly.to_csv(OUT / "ulb_hourly.csv", index=False)
    print(f"  {'ulb_hourly.csv':<24} {len(ulb_hourly)} rows")

    scored_stream = config.STREAM_DIR / "scored"
    if scored_stream.exists() and any(scored_stream.glob("*.parquet")):
        stream = spark.read.parquet(config.spark_path(scored_stream)).withColumnRenamed(config.LABEL, "is_fraud")
        write_single_csv(stream, "stream_scored")


def main() -> None:
    start = time.time()
    config.ensure_dirs()
    spark = get_spark("CreditGuard-PowerBI-Export", driver_memory="6g")
    print(f"Exporting Power BI tables to {OUT.relative_to(config.PROJECT_ROOT)}")
    export_fact_table(spark)
    export_small_tables(spark)
    spark.stop()
    print(f"Export finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
