"""
Prediction logic - connects the trained PySpark models to the backend.

At startup:  a small local SparkSession is started and the two saved Spark
             PipelineModels (one per dataset) are loaded from models/spark/.
Per request: input dict -> derived features -> 1-row Spark DataFrame
             -> PipelineModel.transform() (indexing, scaling, classifier)
             -> fraud score -> label + risk level
"""

import json
import math
import sys

import config


class ModelNotLoadedError(Exception):
    """Raised when a prediction is requested but the Spark model is not available."""


# ---------------------------------------------------------------------------
# Feature preparation (must match spark_pipeline/s03_features.py)
# ---------------------------------------------------------------------------
def ulb_features(data: dict) -> dict:
    """API fields (time, amount, v1..v28) -> model columns."""
    features = {
        "Time": float(data["time"]),
        "Amount": float(data["amount"]),
        "hour_of_day": float(math.floor(data["time"] / 3600) % 24),
        "log_amount": math.log1p(data["amount"]),
    }
    features.update({f"V{i}": float(data[f"v{i}"]) for i in range(1, 29)})
    return features


def sparkov_features(data: dict) -> dict:
    """API fields (readable transaction details) -> model columns."""
    hour = int(data["hour"])
    day_of_week = int(data["day_of_week"])  # 1 = Sunday ... 7 = Saturday (Spark's dayofweek)
    return {
        "amt": float(data["amount"]),
        "log_amt": math.log1p(data["amount"]),
        "hour": float(hour),
        "day_of_week": float(day_of_week),
        "is_night": 1.0 if hour >= 22 or hour < 4 else 0.0,
        "is_weekend": 1.0 if day_of_week in (1, 7) else 0.0,
        "age": float(data["age"]),
        "log_city_pop": math.log1p(data["city_population"]),
        "distance_km": float(data["distance_km"]),
        "txn_count_24h": float(data["txn_count_24h"]),
        "amt_sum_24h": float(data["amount_last_24h"]),
        "hours_since_last": float(data["hours_since_last"]),
        "card_avg_amt": float(data["card_avg_amount"]),
        "amt_to_avg_ratio": float(data["amount"]) / float(data["card_avg_amount"]),
        "category": data["category"],
        "gender": data["gender"],
    }


FEATURE_BUILDERS = {"ulb": ulb_features, "sparkov": sparkov_features}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def classify_risk(score: float, threshold: float) -> str:
    if score >= threshold:
        return "High"
    if score >= threshold * config.MEDIUM_RISK_FACTOR:
        return "Medium"
    return "Low"


class SparkFraudPredictor:
    """Starts Spark ONCE and keeps both models in memory for every request."""

    def __init__(self):
        self.spark = None
        self.models = {}
        self.metadata = {}
        self.load_error = None

    def is_loaded(self, dataset: str | None = None) -> bool:
        if dataset:
            return dataset in self.models
        return len(self.models) == len(config.DATASETS)

    def load(self) -> None:
        try:
            sys.path.insert(0, str(config.PIPELINE_DIR))
            from spark_session import get_spark  # the same Windows-safe helper the pipeline uses
            from pyspark.ml import PipelineModel

            self.spark = get_spark("CreditGuard-API", driver_memory=config.SPARK_API_MEMORY,
                                   cores=config.SPARK_API_CORES)
            missing = []
            for dataset in config.DATASETS:
                model_dir = config.SPARK_MODELS_DIR / f"{dataset}_best"
                metadata_path = config.SPARK_MODELS_DIR / f"{dataset}_best.json"
                if not model_dir.exists() or not metadata_path.exists():
                    missing.append(dataset)
                    continue
                self.models[dataset] = PipelineModel.load(model_dir.as_posix())
                self.metadata[dataset] = json.loads(metadata_path.read_text())
            self.load_error = (
                f"Spark model missing for: {', '.join(missing)}. Run the pipeline: python spark_pipeline/s04_train_models.py"
                if missing else None
            )
        except Exception as error:  # Java missing, corrupted model, etc.
            self.load_error = f"Could not start Spark / load models: {error}"

    def stop(self) -> None:
        if self.spark is not None:
            self.spark.stop()

    def thresholds(self, dataset: str) -> dict:
        """Score cut-offs for the Medium and High risk bands, on the 0-1 scale shown in the UI."""
        metadata = self.metadata[dataset]
        threshold = metadata["threshold"] if metadata["has_probability"] else sigmoid(metadata["threshold"])
        return {"medium": round(threshold * config.MEDIUM_RISK_FACTOR, 4), "high": round(threshold, 4)}

    def predict(self, dataset: str, data: dict) -> dict:
        if not self.is_loaded(dataset):
            raise ModelNotLoadedError(self.load_error or f"Model for '{dataset}' is not loaded.")

        from pyspark.ml.functions import vector_to_array

        metadata = self.metadata[dataset]
        row = FEATURE_BUILDERS[dataset](data)
        frame = self.spark.createDataFrame([row])

        score_column = "probability" if metadata["has_probability"] else "rawPrediction"
        result = (
            self.models[dataset].transform(frame)
            .select(vector_to_array(score_column)[1].alias("score"))
            .first()
        )
        raw_score = float(result["score"])
        # LinearSVC gives a margin, not a probability -> squash it to 0..1 with the sigmoid
        score = raw_score if metadata["has_probability"] else sigmoid(raw_score)
        high = self.thresholds(dataset)["high"]

        return {
            "prediction": "Fraud" if score >= high else "Legitimate",
            "fraud_probability": round(score, 4),
            "risk_level": classify_risk(score, high),
            "model_name": metadata["selected_model"],
        }


# One shared instance used by the whole app
predictor = SparkFraudPredictor()
