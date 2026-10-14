"""
Step 6a - Transaction Stream Producer (simulates a payment gateway)
===================================================================

Run (terminal 1):  python spark_pipeline/s06_stream_producer.py
Options:           --rows 3000  --batch 50  --interval 2  --fraud-share 0.03

Takes REAL transactions from the Sparkov TEST period (unseen by the model), in
time order, and drops them into data/stream/incoming/ as small JSON files -
one "micro-batch" every few seconds - exactly like new transactions arriving.

Fraud is rare (~0.4%), so a short demo might show no fraud at all. --fraud-share
mixes in extra REAL fraud transactions from the test period so alerts are visible;
use --fraud-share 0 for the natural rate. The share used is printed.

In production this role is played by a message queue such as Apache Kafka;
Spark Structured Streaming can read from Kafka with a one-line change.
"""

import argparse
import json
import shutil
import time

from pyspark.sql import functions as F

import config
from s03_features import SPARKOV_CATEGORICAL_FEATURES, SPARKOV_NUMERIC_FEATURES
from spark_session import get_spark

INCOMING = config.STREAM_DIR / "incoming"
STREAM_COLUMNS = (
    ["event_time", "card_id", "merchant", "city", "state"]
    + SPARKOV_NUMERIC_FEATURES + SPARKOV_CATEGORICAL_FEATURES + [config.LABEL]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated real-time transaction producer")
    parser.add_argument("--rows", type=int, default=3000, help="total transactions to send")
    parser.add_argument("--batch", type=int, default=50, help="transactions per micro-batch file")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between files")
    parser.add_argument("--fraud-share", type=float, default=0.03, help="share of real fraud rows mixed in (0 = natural)")
    args = parser.parse_args()

    spark = get_spark("CreditGuard-Producer", driver_memory="2g")
    test = (
        spark.read.parquet(config.spark_path(config.SPARKOV_FEATURES))
        .filter(F.col("split") == "test")
        .withColumn("event_time", F.date_format("trans_ts", "yyyy-MM-dd HH:mm:ss"))
    )
    n_fraud = int(args.rows * args.fraud_share)
    base = test.orderBy("trans_ts").limit(args.rows - n_fraud)
    extra_fraud = test.filter(F.col(config.LABEL) == 1).orderBy(F.rand(config.SEED)).limit(n_fraud)
    rows = base.unionByName(extra_fraud).orderBy(F.rand(config.SEED)).select(STREAM_COLUMNS).toPandas()
    spark.stop()

    natural = int((rows[config.LABEL] == 1).sum()) - n_fraud
    print(f"Prepared {len(rows):,} real test transactions ({natural} natural fraud + {n_fraud} mixed-in fraud)")

    if INCOMING.exists():
        shutil.rmtree(INCOMING)
    INCOMING.mkdir(parents=True)

    # unique prefix per run: the streaming checkpoint remembers file names it already processed
    run_id = time.strftime("%Y%m%d_%H%M%S")
    for batch_number, start in enumerate(range(0, len(rows), args.batch), start=1):
        batch = rows.iloc[start:start + args.batch]
        temporary = INCOMING / f".run{run_id}_batch_{batch_number:05d}.tmp"
        final = INCOMING / f"run{run_id}_batch_{batch_number:05d}.json"
        with temporary.open("w") as handle:
            for record in batch.to_dict(orient="records"):
                handle.write(json.dumps(record) + "\n")
        temporary.rename(final)  # atomic rename -> Spark never reads a half-written file
        print(f"  sent batch {batch_number:>3}: {len(batch)} transactions")
        time.sleep(args.interval)

    print("Producer finished.")


if __name__ == "__main__":
    main()
