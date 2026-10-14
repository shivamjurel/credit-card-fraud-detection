"""
Step 3 - Feature Engineering (PySpark)
======================================

Run:  python spark_pipeline/s03_features.py

Dataset 1 (ULB): V1-V28 are already PCA features, so we only add
  hour_of_day, log_amount.

Dataset 2 (Sparkov): raw columns are turned into behavioural features.
  * Time features      : hour, day_of_week, is_night, is_weekend
  * Customer features  : age (from date of birth), gender
  * Location feature   : distance_km between customer home and merchant (Haversine formula)
  * Amount features    : log_amt
  * Per-card history   : built with Spark WINDOW FUNCTIONS over each card's past transactions
        - txn_count_24h      how many times this card was used in the previous 24 hours
        - amt_sum_24h        how much was spent in the previous 24 hours
        - hours_since_last   hours since this card's previous transaction
        - card_avg_amt       average amount of this card's earlier transactions
        - amt_to_avg_ratio   this amount / card's usual amount  (fraud is often unusually large)
  Only PAST transactions are used, so no information from the future leaks into a feature.

Output: data/processed/*_features.parquet
"""

import time

from pyspark.sql import Window, functions as F

import config
from spark_session import get_spark

LABEL = config.LABEL
SECONDS_PER_DAY = 86_400

# Model input columns (the backend uses the same lists)
ULB_NUMERIC_FEATURES = ["Time", "Amount", "hour_of_day", "log_amount"] + [f"V{i}" for i in range(1, 29)]

SPARKOV_NUMERIC_FEATURES = [
    "amt", "log_amt", "hour", "day_of_week", "is_night", "is_weekend", "age", "log_city_pop",
    "distance_km", "txn_count_24h", "amt_sum_24h", "hours_since_last", "card_avg_amt", "amt_to_avg_ratio",
]
SPARKOV_CATEGORICAL_FEATURES = ["category", "gender"]


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth, as a Spark column expression."""
    lat1, lon1, lat2, lon2 = [F.radians(c) for c in (lat1, lon1, lat2, lon2)]
    a = F.sin((lat2 - lat1) / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin((lon2 - lon1) / 2) ** 2
    return F.lit(2 * 6371.0) * F.asin(F.sqrt(a))


def build_ulb_features(spark) -> None:
    section("FEATURES - DATASET 1 (ULB)")
    df = spark.read.parquet(config.spark_path(config.ULB_CLEAN))
    features = (
        df.withColumn("hour_of_day", (F.floor(F.col("Time") / 3600) % 24).cast("double"))
        .withColumn("log_amount", F.log1p("Amount"))
        .withColumn("row_id", F.monotonically_increasing_id())
    )
    features.write.mode("overwrite").parquet(config.spark_path(config.ULB_FEATURES))
    print(f"  {len(ULB_NUMERIC_FEATURES)} model features -> {config.ULB_FEATURES.name}")


def build_sparkov_features(spark) -> None:
    section("FEATURES - DATASET 2 (Sparkov)")
    df = spark.read.parquet(config.spark_path(config.SPARKOV_CLEAN)).withColumn(
        "ts_sec", F.unix_timestamp("trans_ts")
    )

    # ---- Window over each card's transactions, ordered by time ----
    card_by_time = Window.partitionBy("cc_num").orderBy("ts_sec")
    last_24h = card_by_time.rangeBetween(-SECONDS_PER_DAY, -1)          # previous 24 hours, excluding this one
    all_before = card_by_time.rowsBetween(Window.unboundedPreceding, -1)  # every earlier transaction

    features = (
        df
        # time features
        .withColumn("hour", F.hour("trans_ts"))
        .withColumn("day_of_week", F.dayofweek("trans_ts"))                    # 1 = Sunday ... 7 = Saturday
        .withColumn("is_night", F.when((F.col("hour") >= 22) | (F.col("hour") < 4), 1).otherwise(0))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0))
        # customer & location features
        .withColumn("age", F.floor(F.months_between("trans_ts", "dob") / 12).cast("int"))
        .withColumn("log_city_pop", F.log1p("city_pop"))
        .withColumn("distance_km", haversine_km(F.col("lat"), F.col("long"), F.col("merch_lat"), F.col("merch_long")))
        .withColumn("log_amt", F.log1p("amt"))
        # per-card history (window functions)
        .withColumn("txn_count_24h", F.count("*").over(last_24h))
        .withColumn("amt_sum_24h", F.coalesce(F.sum("amt").over(last_24h), F.lit(0.0)))
        .withColumn("prev_ts", F.lag("ts_sec").over(card_by_time))
        .withColumn("card_avg_amt", F.avg("amt").over(all_before))
    )

    features = (
        features
        # first transaction of a card has no history -> sensible defaults
        .withColumn("hours_since_last", F.coalesce((F.col("ts_sec") - F.col("prev_ts")) / 3600.0, F.lit(720.0)))
        .withColumn("card_avg_amt", F.coalesce("card_avg_amt", "amt"))
        .withColumn("amt_to_avg_ratio", F.col("amt") / F.col("card_avg_amt"))
        # anonymous card key for reporting; the real card number is dropped
        .withColumn("card_id", F.substring(F.sha2(F.col("cc_num").cast("string"), 256), 1, 12))
        .drop("cc_num", "prev_ts")
    )

    for column in ["txn_count_24h", "hour", "day_of_week", "is_night", "is_weekend", "age"]:
        features = features.withColumn(column, F.col(column).cast("double"))

    features.write.mode("overwrite").parquet(config.spark_path(config.SPARKOV_FEATURES))
    saved = spark.read.parquet(config.spark_path(config.SPARKOV_FEATURES))
    print(f"  Rows: {saved.count():,} | {len(SPARKOV_NUMERIC_FEATURES)} numeric + "
          f"{len(SPARKOV_CATEGORICAL_FEATURES)} categorical features -> {config.SPARKOV_FEATURES.name}")

    # Show how the engineered features differ between fraud and legitimate transactions
    print("\n  Average of engineered features (legitimate vs fraud):")
    compare = ["amt", "is_night", "distance_km", "txn_count_24h", "amt_sum_24h", "hours_since_last", "amt_to_avg_ratio"]
    saved.groupBy(LABEL).agg(*[F.round(F.avg(c), 2).alias(c) for c in compare]).orderBy(LABEL).show(truncate=False)


def main() -> None:
    start = time.time()
    spark = get_spark("CreditGuard-Features")
    build_ulb_features(spark)
    build_sparkov_features(spark)
    spark.stop()
    print(f"Feature engineering finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
