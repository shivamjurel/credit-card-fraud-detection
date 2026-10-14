"""
Step 4b - Pick real demo transactions for the website
=====================================================

Run:  python spark_pipeline/s04b_demo_samples.py   (after s04_train_models.py)

Takes real rows from each dataset's TEST set (never seen in training), scores them
with the saved best model, and stores a few for the "Load sample" buttons:
  * legitimate rows the model classified correctly
  * fraud rows the model detected
(The model does NOT catch every fraud - see recall in the comparison table.)

Output: models/spark/samples.json  (field names = the website form / API fields)
"""

import json

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import config
from s04_train_models import load_splits
from spark_session import get_spark

PER_CLASS = 6


def ulb_to_api(row) -> dict:
    sample = {"time": row["Time"], "amount": row["Amount"]}
    sample.update({f"v{i}": round(row[f"V{i}"], 6) for i in range(1, 29)})
    return sample


def sparkov_to_api(row) -> dict:
    return {
        "amount": row["amt"],
        "category": row["category"],
        "gender": row["gender"],
        "age": int(row["age"]),
        "hour": int(row["hour"]),
        "day_of_week": int(row["day_of_week"]),
        "city_population": int(row["city_pop"]),
        "distance_km": round(row["distance_km"], 2),
        "txn_count_24h": int(row["txn_count_24h"]),
        "amount_last_24h": round(row["amt_sum_24h"], 2),
        "hours_since_last": round(row["hours_since_last"], 3),
        "card_avg_amount": round(row["card_avg_amt"], 2),
        # context only (shown in the UI, not used by the model)
        "merchant": row["merchant"],
        "city": row["city"],
        "state": row["state"],
    }


def main() -> None:
    spark = get_spark("CreditGuard-Samples")
    samples = {}
    for dataset, to_api in [("ulb", ulb_to_api), ("sparkov", sparkov_to_api)]:
        metadata = json.loads((config.SPARK_MODELS_DIR / f"{dataset}_best.json").read_text())
        model = PipelineModel.load(config.spark_path(config.SPARK_MODELS_DIR / f"{dataset}_best"))
        _, _, test = load_splits(spark, dataset)

        score_column = "probability" if metadata["has_probability"] else "rawPrediction"
        scored = model.transform(test).withColumn("score", vector_to_array(score_column)[1])
        detected = F.col("score") >= metadata["threshold"]

        legit = scored.filter((F.col(config.LABEL) == 0) & ~detected).orderBy(F.rand(config.SEED)).limit(PER_CLASS)
        fraud = scored.filter((F.col(config.LABEL) == 1) & detected).orderBy(F.rand(config.SEED)).limit(PER_CLASS)
        samples[dataset] = {
            "legitimate": [{**to_api(r), "actual_class": "Legitimate"} for r in legit.collect()],
            "fraud": [{**to_api(r), "actual_class": "Fraud"} for r in fraud.collect()],
        }
        print(f"{dataset}: {len(samples[dataset]['legitimate'])} legit + {len(samples[dataset]['fraud'])} fraud samples")

    (config.SPARK_MODELS_DIR / "samples.json").write_text(json.dumps(samples, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
