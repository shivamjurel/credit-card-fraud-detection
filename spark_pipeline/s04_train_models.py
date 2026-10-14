"""
Step 4 - Train & Compare 5 PySpark MLlib Models on 2 Datasets
=============================================================

Run:  python spark_pipeline/s04_train_models.py            (both datasets)
      python spark_pipeline/s04_train_models.py sparkov    (one dataset)

Models (all from pyspark.ml.classification):
  1. Logistic Regression       - linear baseline, outputs probabilities
  2. Decision Tree             - simple, explainable if/else rules
  3. Random Forest             - many trees on random samples, majority vote
  4. Gradient-Boosted Trees    - trees built one after another, each fixing previous errors
  5. Linear SVM (LinearSVC)    - finds the widest separating boundary

Method (avoids "cheating" on the test set):
  * Split      ULB: stratified 70% train / 10% validation / 20% test
               Sparkov: the dataset's own TIME-BASED split (train = 2019..mid-2020, test = later)
                        + 10% of train held out for validation
  * Imbalance  class weights (weightCol): each fraud row counts ~as much as ~190-600 legit rows
  * Threshold  for each model, the probability cut-off that gives the best F1 on the VALIDATION set
  * Selection  best model = highest validation F1 (the test set is only used for the final report)

Outputs:
  models/spark/<dataset>_best/            saved Spark PipelineModel (features + model)
  models/spark/<dataset>_best.json        metadata (threshold, features, metrics)
  reports/results/model_comparison.json/.csv
  reports/results/scores/<dataset>_<model>.npz   test-set scores for ROC / PR curves (step 5)
"""

import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

import config
from s03_features import SPARKOV_CATEGORICAL_FEATURES, SPARKOV_NUMERIC_FEATURES, ULB_NUMERIC_FEATURES
from spark_session import get_spark

LABEL = config.LABEL
SEED = config.SEED
SCORES_DIR = config.RESULTS_DIR / "scores"


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------------
# Data splitting & class weights
# --------------------------------------------------------------------------
def stratified_split(df: DataFrame, weights: list[float]) -> list[DataFrame]:
    """randomSplit done separately per class, so every part keeps the same fraud %."""
    legit_parts = df.filter(F.col(LABEL) == 0).randomSplit(weights, seed=SEED)
    fraud_parts = df.filter(F.col(LABEL) == 1).randomSplit(weights, seed=SEED)
    return [legit.unionByName(fraud) for legit, fraud in zip(legit_parts, fraud_parts)]


def add_class_weights(df: DataFrame) -> tuple[DataFrame, dict]:
    """Balanced weights: weight(class) = total / (2 * rows_in_class)."""
    counts = {row[LABEL]: row["count"] for row in df.groupBy(LABEL).count().collect()}
    total = sum(counts.values())
    weights = {label: total / (2.0 * count) for label, count in counts.items()}
    weighted = df.withColumn(
        "class_weight", F.when(F.col(LABEL) == 1, F.lit(weights[1])).otherwise(F.lit(weights[0]))
    )
    return weighted, weights


def load_splits(spark, dataset: str):
    df = spark.read.parquet(config.spark_path(config.DATASETS[dataset]["features"]))
    if dataset == "ulb":
        train, validation, test = stratified_split(df, [0.7, 0.1, 0.2])
    else:
        train_all = df.filter(F.col("split") == "train")
        test = df.filter(F.col("split") == "test")
        train, validation = stratified_split(train_all, [0.9, 0.1])
    return train, validation, test


# --------------------------------------------------------------------------
# Feature pipeline + models
# --------------------------------------------------------------------------
def feature_stages(dataset: str) -> list:
    """Stages that turn raw columns into one scaled 'features' vector."""
    if dataset == "ulb":
        assembler = VectorAssembler(inputCols=ULB_NUMERIC_FEATURES, outputCol="raw_features")
        return [assembler, StandardScaler(inputCol="raw_features", outputCol="features", withMean=True)]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in SPARKOV_CATEGORICAL_FEATURES
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in SPARKOV_CATEGORICAL_FEATURES],
        outputCols=[f"{c}_vec" for c in SPARKOV_CATEGORICAL_FEATURES],
    )
    assembler = VectorAssembler(
        inputCols=SPARKOV_NUMERIC_FEATURES + [f"{c}_vec" for c in SPARKOV_CATEGORICAL_FEATURES],
        outputCol="raw_features",
    )
    return indexers + [encoder, assembler, StandardScaler(inputCol="raw_features", outputCol="features")]


def build_models() -> dict:
    common = {"featuresCol": "features", "labelCol": LABEL, "weightCol": "class_weight"}
    return {
        "Logistic Regression": LogisticRegression(maxIter=100, regParam=0.001, **common),
        "Decision Tree": DecisionTreeClassifier(maxDepth=8, minInstancesPerNode=20, seed=SEED, **common),
        "Random Forest": RandomForestClassifier(numTrees=100, maxDepth=10, minInstancesPerNode=10,
                                                featureSubsetStrategy="sqrt", seed=SEED, **common),
        "Gradient-Boosted Trees": GBTClassifier(maxIter=80, maxDepth=5, stepSize=0.1, seed=SEED, **common),
        "Linear SVM": LinearSVC(maxIter=100, regParam=0.001, **common),
    }


# --------------------------------------------------------------------------
# Scores, threshold tuning, metrics
# --------------------------------------------------------------------------
def with_score(predictions: DataFrame) -> DataFrame:
    """
    A single fraud score per row:
      probability[1] for models that give probabilities,
      rawPrediction[1] (distance from the boundary) for LinearSVC.
    """
    if "probability" in predictions.columns:
        return predictions.withColumn("score", vector_to_array("probability")[1])
    return predictions.withColumn("score", vector_to_array("rawPrediction")[1])


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Try every possible cut-off on the validation scores; return the one with the best F1."""
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    true_positives = np.cumsum(sorted_labels)
    predicted_positives = np.arange(1, len(sorted_labels) + 1)
    precision = true_positives / predicted_positives
    recall = true_positives / max(sorted_labels.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    # only consider positions where the score changes (a real threshold)
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    best = np.argmax(np.where(distinct, f1, -1))
    return float(sorted_scores[best]), float(f1[best])


def confusion_metrics(scored: DataFrame, threshold: float) -> dict:
    """Confusion matrix + precision/recall/F1/accuracy, counted by Spark."""
    predicted = F.when(F.col("score") >= threshold, 1).otherwise(0)
    row = scored.select(
        F.sum(((predicted == 1) & (F.col(LABEL) == 1)).cast("int")).alias("tp"),
        F.sum(((predicted == 1) & (F.col(LABEL) == 0)).cast("int")).alias("fp"),
        F.sum(((predicted == 0) & (F.col(LABEL) == 0)).cast("int")).alias("tn"),
        F.sum(((predicted == 0) & (F.col(LABEL) == 1)).cast("int")).alias("fn"),
    ).first()
    tp, fp, tn, fn = row.tp, row.fp, row.tn, row.fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / (tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
    }


def feature_importances(pipeline_model, dataset: str) -> dict:
    """Top feature importances for tree models (names recovered from the assembler metadata)."""
    model = pipeline_model.stages[-1]
    if not hasattr(model, "featureImportances"):
        return {}
    assembler = next(stage for stage in pipeline_model.stages if isinstance(stage, VectorAssembler))
    names = list(assembler.getInputCols())
    if dataset == "sparkov":  # expand one-hot columns into their category names
        names = SPARKOV_NUMERIC_FEATURES.copy()
        for stage in pipeline_model.stages:
            if hasattr(stage, "labels") and hasattr(stage, "getInputCol"):
                column = stage.getInputCol()
                names += [f"{column}={label}" for label in stage.labels]  # dropLast removes the 'invalid' slot
    importances = model.featureImportances.toArray()
    names = names[: len(importances)] + [f"f{i}" for i in range(len(names), len(importances))]
    top = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:15]
    return {name: round(float(value), 4) for name, value in top}


# --------------------------------------------------------------------------
# Main training loop for one dataset
# --------------------------------------------------------------------------
def train_dataset(spark, dataset: str) -> dict:
    section(f"DATASET: {config.DATASETS[dataset]['name']}")
    train, validation, test = load_splits(spark, dataset)
    train, weights = add_class_weights(train)
    train, validation, test = train.cache(), validation.cache(), test.cache()

    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        counts = {r[LABEL]: r["count"] for r in part.groupBy(LABEL).count().collect()}
        print(f"  {name:<11}: {sum(counts.values()):>9,} rows | {counts.get(1, 0):>6,} fraud")
    print(f"  Class weights: legit={weights[0]:.3f}, fraud={weights[1]:.1f}")

    # Fit the feature stages ONCE on training data, reuse for every model
    feature_model = Pipeline(stages=feature_stages(dataset)).fit(train)
    train_f = feature_model.transform(train).select("features", LABEL, "class_weight").cache()
    validation_f = feature_model.transform(validation).select("features", LABEL).cache()
    test_f = feature_model.transform(test).select("features", LABEL).cache()
    train_f.count()

    auc_roc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_pr = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    results, fitted = {}, {}
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, estimator in build_models().items():
        print(f"\n  >> {model_name}")
        start = time.time()
        model = estimator.fit(train_f)
        train_seconds = time.time() - start

        # 1) tune the decision threshold on the VALIDATION set
        validation_scores = with_score(model.transform(validation_f)).select("score", LABEL).toPandas()
        threshold, validation_f1 = best_f1_threshold(
            validation_scores["score"].to_numpy(), validation_scores[LABEL].to_numpy()
        )

        # 2) final, one-time evaluation on the TEST set
        test_predictions = with_score(model.transform(test_f)).cache()
        metrics = confusion_metrics(test_predictions, threshold)
        metrics["roc_auc"] = auc_roc.evaluate(test_predictions)
        metrics["pr_auc"] = auc_pr.evaluate(test_predictions)
        metrics.update(
            threshold=threshold,
            validation_f1=validation_f1,
            train_seconds=round(train_seconds, 1),
            has_probability="probability" in test_predictions.columns,
        )

        test_scores = test_predictions.select("score", LABEL).toPandas()
        np.savez_compressed(
            SCORES_DIR / f"{dataset}_{model_name.lower().replace(' ', '_').replace('-', '_')}.npz",
            score=test_scores["score"].to_numpy(),
            label=test_scores[LABEL].to_numpy(),
            threshold=threshold,
        )
        test_predictions.unpersist()

        results[model_name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
        fitted[model_name] = model
        cm = metrics["confusion_matrix"]
        print(f"     trained in {train_seconds:.1f}s | threshold {threshold:.4f} | val F1 {validation_f1:.4f}")
        print(f"     TEST -> precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
              f"F1 {metrics['f1_score']:.4f}  ROC-AUC {metrics['roc_auc']:.4f}  PR-AUC {metrics['pr_auc']:.4f}")
        print(f"     TP {cm['true_positive']}  FP {cm['false_positive']}  FN {cm['false_negative']}  TN {cm['true_negative']}")

    # ---- Select best model on VALIDATION F1 ----
    best_name = max(results, key=lambda name: results[name]["validation_f1"])
    print(f"\n  SELECTED: {best_name} (highest validation F1 = {results[best_name]['validation_f1']:.4f})")

    # One PipelineModel = feature stages + chosen classifier -> takes raw columns, outputs predictions
    best_pipeline = PipelineModel(stages=feature_model.stages + [fitted[best_name]])
    model_dir = config.SPARK_MODELS_DIR / f"{dataset}_best"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    best_pipeline.write().overwrite().save(config.spark_path(model_dir))

    importances = feature_importances(best_pipeline, dataset)
    metadata = {
        "dataset": dataset,
        "dataset_name": config.DATASETS[dataset]["name"],
        "selected_model": best_name,
        "selection_metric": "validation F1",
        "threshold": results[best_name]["threshold"],
        "has_probability": results[best_name]["has_probability"],
        "numeric_features": ULB_NUMERIC_FEATURES if dataset == "ulb" else SPARKOV_NUMERIC_FEATURES,
        "categorical_features": [] if dataset == "ulb" else SPARKOV_CATEGORICAL_FEATURES,
        "test_metrics": results[best_name],
        "feature_importances": importances,
        "class_weights": {"legitimate": weights[0], "fraud": weights[1]},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "spark_version": spark.version,
    }
    (config.SPARK_MODELS_DIR / f"{dataset}_best.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Saved best pipeline -> models/spark/{dataset}_best")
    if importances:
        print(f"  Top features: {list(importances)[:6]}")

    for frame in (train, validation, test, train_f, validation_f, test_f):
        frame.unpersist()
    return {"selected_model": best_name, "models": results, "feature_importances": importances}


def main() -> None:
    start = time.time()
    config.ensure_dirs()
    datasets = sys.argv[1:] or ["ulb", "sparkov"]
    spark = get_spark("CreditGuard-Train", driver_memory="8g")

    results_path = config.RESULTS_DIR / "model_comparison.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for dataset in datasets:
        all_results[dataset] = train_dataset(spark, dataset)
        results_path.write_text(json.dumps(all_results, indent=2))

    # Flat comparison table (CSV) for the report and Power BI
    rows = []
    for dataset, data in all_results.items():
        for model_name, m in data["models"].items():
            cm = m["confusion_matrix"]
            rows.append({
                "dataset": dataset, "model": model_name, "selected": model_name == data["selected_model"],
                "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                "f1_score": m["f1_score"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                "threshold": m["threshold"], "train_seconds": m["train_seconds"],
                "tp": cm["true_positive"], "fp": cm["false_positive"], "fn": cm["false_negative"], "tn": cm["true_negative"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n" + table[["dataset", "model", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "train_seconds"]].to_string(index=False))

    spark.stop()
    print(f"\nTraining finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
