"""
Phase 3 - Model Evaluation
==========================

Run:  python ml/evaluate_model.py

This script loads the SAVED model + scaler from models/, recreates the exact
same test set used during training (same cleaning, same random_state), and
prints a full evaluation report.

It also contains the metric helpers that train_model.py reuses to compare models.

WHY NOT ONLY ACCURACY?
----------------------
In this dataset only ~0.17% of transactions are fraud. A "dumb" model that
ALWAYS says "Legitimate" would be ~99.83% accurate but would catch ZERO fraud.
So for fraud detection we focus on:
  * Precision - of all transactions we flagged as fraud, how many were really fraud?
  * Recall    - of all real frauds, how many did we catch?
  * F1-score  - harmonic mean of precision and recall (balances both)
  * ROC-AUC   - how well the model ranks fraud above legitimate, across ALL thresholds
"""

import json

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing import (
    METADATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    clean_dataset,
    load_dataset,
    split_features_target,
    split_train_test,
)


def compute_metrics(y_true, y_pred, y_probability) -> dict:
    """Calculate all the fraud-detection metrics for one model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probability)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),   # legit correctly predicted legit
            "false_positive": int(fp),  # legit wrongly flagged as fraud (false alarm)
            "false_negative": int(fn),  # fraud we MISSED (most costly mistake)
            "true_positive": int(tp),   # fraud correctly caught
        },
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    """Pretty-print the metrics and confusion matrix for one model."""
    cm = metrics["confusion_matrix"]
    print(f"\n  --- {model_name} ---")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("  Confusion matrix:")
    print("                     Predicted Legit   Predicted Fraud")
    print(f"    Actual Legit     {cm['true_negative']:>15}   {cm['false_positive']:>15}")
    print(f"    Actual Fraud     {cm['false_negative']:>15}   {cm['true_positive']:>15}")


def print_accuracy_trap(y_test) -> None:
    """Show why accuracy alone is misleading on imbalanced data."""
    always_legit_predictions = np.zeros(len(y_test), dtype=int)
    baseline_accuracy = accuracy_score(y_test, always_legit_predictions)
    print("\n  Accuracy trap demo - a model that ALWAYS predicts 'Legitimate':")
    print(f"    Accuracy = {baseline_accuracy:.4%}  but Recall = 0.00 (catches no fraud at all!)")


def main() -> None:
    print("=" * 70)
    print("EVALUATING SAVED MODEL")
    print("=" * 70)

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise SystemExit("Model or scaler not found. Run `python ml/train_model.py` first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    decision_threshold = metadata.get("decision_threshold", 0.5)
    print(f"\nLoaded model : {metadata.get('selected_model', type(model).__name__)}")
    print(f"Decision threshold: {decision_threshold}")

    # Recreate the same test set as training (same cleaning + same random_state)
    print("\nPreparing test data ...")
    df = clean_dataset(load_dataset(), verbose=False)
    X, y = split_features_target(df)
    _, X_test, _, y_test = split_train_test(X, y)

    # Scale the test features with the scaler fitted on the TRAINING data
    X_test_scaled = scaler.transform(X_test)

    fraud_probability = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (fraud_probability >= decision_threshold).astype(int)

    print(f"Test set: {len(y_test)} transactions ({int(y_test.sum())} fraud)")
    metrics = compute_metrics(y_test, predictions, fraud_probability)
    print_metrics(metadata.get("selected_model", "Saved model"), metrics)
    print_accuracy_trap(y_test)

    print("\n  Detailed classification report:")
    print(classification_report(y_test, predictions, target_names=["Legitimate", "Fraud"], digits=4))

    # How precision/recall change when we move the decision threshold.
    # Lower threshold -> catch more fraud (recall up) but more false alarms (precision down).
    print("  Threshold analysis (trade-off between precision and recall):")
    print("    Threshold   Precision   Recall   F1-score")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        threshold_predictions = (fraud_probability >= threshold).astype(int)
        print(
            f"    {threshold:>9.1f}   "
            f"{precision_score(y_test, threshold_predictions, zero_division=0):>9.4f}   "
            f"{recall_score(y_test, threshold_predictions):>6.4f}   "
            f"{f1_score(y_test, threshold_predictions):>8.4f}"
        )


if __name__ == "__main__":
    main()
