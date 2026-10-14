"""
Step 6b - Real-Time Fraud Scoring with Spark Structured Streaming
=================================================================

Run (terminal 2):  python spark_pipeline/s06_streaming.py
Then start the producer in another terminal (s06_stream_producer.py).
Stop with Ctrl+C.

How it works:
  1. readStream watches data/stream/incoming/ for new JSON files (micro-batches)
  2. every new batch is scored by the saved best Sparkov PipelineModel
  3. foreachBatch:
       * prints fraud ALERTS for the batch
       * appends scored rows to data/stream/scored/ (Parquet - for Power BI / analysis)
       * inserts rows into the website's SQLite database (source = 'stream'),
         so the React dashboard updates live
  4. A checkpoint folder remembers which files were processed (fault tolerance:
     after a restart Spark continues where it stopped, without duplicates).

With Kafka the only change is the source:
    spark.readStream.format("kafka").option("subscribe", "transactions")...
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

import config
from s03_features import SPARKOV_CATEGORICAL_FEATURES, SPARKOV_NUMERIC_FEATURES
from spark_session import get_spark

INCOMING = config.STREAM_DIR / "incoming"
SCORED = config.STREAM_DIR / "scored"
CHECKPOINT = config.STREAM_DIR / "checkpoint"
DATABASE = config.PROJECT_ROOT / "backend" / "fraud_detection.db"
MEDIUM_RISK_FACTOR = 0.5  # same default as backend/config.py

STREAM_SCHEMA = StructType(
    [StructField("event_time", StringType()), StructField("card_id", StringType()),
     StructField("merchant", StringType()), StructField("city", StringType()), StructField("state", StringType())]
    + [StructField(c, DoubleType()) for c in SPARKOV_NUMERIC_FEATURES]
    + [StructField(c, StringType()) for c in SPARKOV_CATEGORICAL_FEATURES]
    + [StructField(config.LABEL, IntegerType())]
)


def risk_level(score: float, threshold: float) -> str:
    if score >= threshold:
        return "High"
    return "Medium" if score >= threshold * MEDIUM_RISK_FACTOR else "Low"


def save_to_website_database(rows, threshold: float, model_name: str) -> None:
    """Insert scored transactions into the backend's SQLite table (source = 'stream')."""
    if not DATABASE.exists():
        return
    now = datetime.now().isoformat(sep=" ")
    records = []
    for row in rows:
        features = {
            "amount": row["amt"], "category": row["category"], "gender": row["gender"], "age": row["age"],
            "hour": row["hour"], "day_of_week": row["day_of_week"], "distance_km": row["distance_km"],
            "txn_count_24h": row["txn_count_24h"], "amount_last_24h": row["amt_sum_24h"],
            "hours_since_last": row["hours_since_last"], "card_avg_amount": row["card_avg_amt"],
            "merchant": row["merchant"], "city": row["city"], "state": row["state"],
            "event_time": row["event_time"],
        }
        records.append((
            "sparkov", "stream", row["amt"], row["category"],
            "Fraud" if row["score"] >= threshold else "Legitimate",
            round(row["score"], 4), risk_level(row["score"], threshold), model_name, now, json.dumps(features),
        ))
    with sqlite3.connect(DATABASE) as connection:
        connection.executemany(
            "INSERT INTO transactions (dataset, source, amount, category, prediction, fraud_probability, "
            "risk_level, model_name, created_at, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )


def main() -> None:
    metadata = json.loads((config.SPARK_MODELS_DIR / "sparkov_best.json").read_text())
    threshold, model_name = metadata["threshold"], metadata["selected_model"]
    if not metadata["has_probability"]:
        sys.exit("Streaming demo expects a probability model (selected model has none).")

    spark = get_spark("CreditGuard-Streaming", driver_memory="3g", cores="4")
    model = PipelineModel.load(config.spark_path(config.SPARK_MODELS_DIR / "sparkov_best"))
    INCOMING.mkdir(parents=True, exist_ok=True)
    if not DATABASE.exists():
        print(f"NOTE: {DATABASE.name} not found - start the backend once so the website table exists.")

    stream = (
        spark.readStream.schema(STREAM_SCHEMA)
        .option("maxFilesPerTrigger", 1)          # one micro-batch file per trigger
        .json(config.spark_path(INCOMING))
    )
    scored = model.transform(stream).withColumn("score", vector_to_array("probability")[1])

    totals = {"transactions": 0, "alerts": 0, "true_fraud_caught": 0, "fraud_seen": 0}

    def process_batch(batch_df, batch_id: int) -> None:
        output = batch_df.select(
            "event_time", "card_id", "merchant", "city", "state", "category", "gender", "amt", "age", "hour",
            "day_of_week", "distance_km", "txn_count_24h", "amt_sum_24h", "hours_since_last", "card_avg_amt",
            config.LABEL, "score",
        ).withColumn("predicted_fraud", (F.col("score") >= threshold).cast("int"))
        rows = output.collect()
        if not rows:
            return

        output.write.mode("append").parquet(config.spark_path(SCORED))
        save_to_website_database(rows, threshold, model_name)

        alerts = [r for r in rows if r["predicted_fraud"]]
        totals["transactions"] += len(rows)
        totals["alerts"] += len(alerts)
        totals["fraud_seen"] += sum(r[config.LABEL] for r in rows)
        totals["true_fraud_caught"] += sum(r[config.LABEL] for r in alerts)
        print(f"[batch {batch_id:>3}] {len(rows):>3} transactions scored | {len(alerts)} fraud alert(s) | "
              f"running: {totals['transactions']:,} txns, {totals['alerts']} alerts, "
              f"{totals['true_fraud_caught']}/{totals['fraud_seen']} real frauds caught")
        for alert in alerts:
            print(f"      ALERT  ${alert['amt']:>8.2f}  {alert['category']:<14} {alert['city']}, {alert['state']}  "
                  f"score {alert['score']:.3f}  (card {alert['card_id']})")

    query = (
        scored.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", config.spark_path(CHECKPOINT))
        .trigger(processingTime="2 seconds")
        .start()
    )
    print(f"Streaming started with {model_name} (threshold {threshold:.3f}). Watching {INCOMING} ... Ctrl+C to stop.")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping stream...")
        query.stop()
    finally:
        spark.stop()
        print(f"Summary: {totals}")


if __name__ == "__main__":
    main()
