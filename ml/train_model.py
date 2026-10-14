"""
Phases 1-4 - Credit Card Fraud Detection: Model Training
========================================================

Run:  python ml/train_model.py

Pipeline:
  1. Load dataset                 (Pandas)
  2. Explore dataset              (rows, columns, missing values, class distribution)
  3. Clean data                   (missing values, duplicates)
  4. Handle class imbalance       (class_weight="balanced")
  5. Train/test split             (80/20, stratified)
  6. Feature scaling              (StandardScaler, fitted on TRAIN data only)
  7. Train models                 (Logistic Regression, Random Forest, XGBoost if installed)
  8. Compare models               (Precision, Recall, F1, ROC-AUC - NOT just accuracy)
  9. Select the final model       (highest F1-score on the fraud class)
 10. Save model + scaler          (models/fraud_model.pkl, models/scaler.pkl)
"""

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate_model import compute_metrics, print_accuracy_trap, print_metrics
from preprocessing import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SAMPLES_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    clean_dataset,
    load_dataset,
    split_features_target,
    split_train_test,
)

# A transaction is labelled "Fraud" when fraud probability >= this value.
DECISION_THRESHOLD = 0.5


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def explore_dataset(df) -> dict:
    """Step 2: Print basic facts about the dataset."""
    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    legit_count = int((df[TARGET_COLUMN] == 0).sum())
    total = len(df)

    print(f"  Number of rows    : {total}")
    print(f"  Number of columns : {df.shape[1]}")
    print(f"  Missing values    : {int(df.isnull().sum().sum())}")
    print(f"  Legitimate (0)    : {legit_count}")
    print(f"  Fraud (1)         : {fraud_count}")
    print(f"  Class distribution: {legit_count / total:.4%} legitimate, {fraud_count / total:.4%} fraud")
    print(f"  Amount range      : {df['Amount'].min():.2f} to {df['Amount'].max():.2f}")
    print(f"  Time range (sec)  : {df['Time'].min():.0f} to {df['Time'].max():.0f}")

    return {"rows": total, "columns": int(df.shape[1]), "fraud": fraud_count, "legitimate": legit_count}


def build_models() -> dict:
    """
    Step 4 + 7: Create the candidate models.

    class_weight="balanced" tells the model: "a mistake on a fraud example
    costs MORE than a mistake on a legit example" (weight = inverse of class
    frequency). This way the model cannot ignore the tiny fraud class.
    """
    models = {
        # Simple, fast, interpretable linear model -> a good baseline
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        # Ensemble of many decision trees; captures non-linear patterns
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    # XGBoost is OPTIONAL - it is only trained if the package is installed
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("  (XGBoost not installed - skipping the optional XGBoost model)")

    return models


def pick_sample_transactions(model, X_test, X_test_scaled, y_test, per_class: int = 5) -> dict:
    """
    Save a few REAL rows from the test set (never seen during training) so the
    web app can offer "load sample" buttons and the automated tests have real inputs.

    Note: we deliberately pick fraud rows that the model DID detect and legit rows
    it classified correctly, so that the demo buttons show both outcomes. The model
    does not catch every fraud (see recall), which is explained in the README.
    """
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    rng = np.random.default_rng(RANDOM_STATE)

    def rows_to_json(mask, label):
        candidate_positions = np.flatnonzero(mask)
        chosen = rng.choice(candidate_positions, size=min(per_class, len(candidate_positions)), replace=False)
        samples = []
        for position in chosen:
            row = X_test.iloc[position]
            sample = {column.lower(): round(float(row[column]), 6) for column in FEATURE_COLUMNS}
            sample["actual_class"] = label
            samples.append(sample)
        return samples

    y_values = y_test.to_numpy()
    return {
        "legitimate": rows_to_json((y_values == 0) & (predictions == 0), "Legitimate"),
        "fraud": rows_to_json((y_values == 1) & (predictions == 1), "Fraud"),
    }


def main() -> None:
    start_time = time.time()

    # ------------------------------------------------------------------ Step 1
    section("STEP 1: LOAD DATASET")
    df = load_dataset()
    print(f"  Loaded data/creditcard.csv -> shape {df.shape}")

    # ------------------------------------------------------------------ Step 2
    section("STEP 2: DATA EXPLORATION")
    dataset_stats = explore_dataset(df)

    # ------------------------------------------------------------------ Step 3
    section("STEP 3: HANDLE MISSING VALUES & DUPLICATES")
    df = clean_dataset(df)

    # ------------------------------------------------------------------ Step 4
    section("STEP 4: CLASS IMBALANCE")
    fraud_ratio = df[TARGET_COLUMN].mean()
    print(f"  Only {fraud_ratio:.4%} of transactions are fraud (about 1 in {int(round(1 / fraud_ratio))}).")
    print("  A model could get ~99.8% accuracy by ALWAYS saying 'Legitimate' - and catch zero fraud.")
    print("  Fix used: class_weight='balanced' -> errors on the rare fraud class are penalised more.")

    # ------------------------------------------------------------------ Step 5/6
    section("STEP 5: TRAIN / TEST SPLIT (80/20, stratified)")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"  Training set: {len(X_train)} rows ({int(y_train.sum())} fraud)")
    print(f"  Test set    : {len(X_test)} rows ({int(y_test.sum())} fraud)")

    section("STEP 6: FEATURE SCALING (StandardScaler)")
    # IMPORTANT: fit the scaler on the TRAINING data only. Fitting on all data
    # would "leak" information about the test set into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaled all 30 features to mean = 0, standard deviation = 1.")
    print(f"  Example - 'Amount' mean before scaling: {X_train['Amount'].mean():.2f}, "
          f"after: {X_train_scaled[:, FEATURE_COLUMNS.index('Amount')].mean():.4f}")
    print(f"  Example - 'Time'   mean before scaling: {X_train['Time'].mean():.2f}, "
          f"after: {X_train_scaled[:, FEATURE_COLUMNS.index('Time')].mean():.4f}")

    # ------------------------------------------------------------------ Step 7
    section("STEP 7: TRAIN MODELS")
    models = build_models()
    results = {}
    for model_name, model in models.items():
        print(f"\n  Training {model_name} ...", end=" ", flush=True)
        model_start = time.time()
        model.fit(X_train_scaled, y_train)
        print(f"done in {time.time() - model_start:.1f}s")

        fraud_probability = model.predict_proba(X_test_scaled)[:, 1]
        predictions = (fraud_probability >= DECISION_THRESHOLD).astype(int)
        results[model_name] = compute_metrics(y_test, predictions, fraud_probability)

    # ------------------------------------------------------------------ Step 8
    section("STEP 8: MODEL EVALUATION & COMPARISON")
    print_accuracy_trap(y_test)
    for model_name, metrics in results.items():
        print_metrics(model_name, metrics)

    print("\n  Summary table:")
    print(f"  {'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'ROC-AUC':>10}")
    for model_name, m in results.items():
        print(f"  {model_name:<22}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1_score']:>9.4f}{m['roc_auc']:>10.4f}")

    # ------------------------------------------------------------------ Step 9
    section("STEP 9: SELECT FINAL MODEL")
    # We choose the model with the best F1-score for the fraud class, because F1
    # balances catching fraud (recall) with not raising too many false alarms
    # (precision). ROC-AUC is used as a tie-breaker. Accuracy is NOT used.
    selected_name = max(results, key=lambda name: (results[name]["f1_score"], results[name]["roc_auc"]))
    selected_model = models[selected_name]
    print(f"  Selected: {selected_name}")
    print(f"  Reason  : highest F1-score ({results[selected_name]['f1_score']:.4f}) "
          "- best balance of precision and recall.")

    # ------------------------------------------------------------------ Step 10
    section("STEP 10: SAVE MODEL, SCALER & METADATA")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH, compress=3)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "selected_model": selected_name,
        "selection_metric": "f1_score",
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": results,
        "dataset": {
            **dataset_stats,
            "rows_after_cleaning": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": TEST_SIZE,
        },
        "imbalance_handling": "class_weight='balanced'",
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    samples = pick_sample_transactions(selected_model, X_test, X_test_scaled, y_test)
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2))

    print(f"  Model    -> {MODEL_PATH.relative_to(MODELS_DIR.parent)}")
    print(f"  Scaler   -> {SCALER_PATH.relative_to(MODELS_DIR.parent)}")
    print(f"  Metadata -> {METADATA_PATH.relative_to(MODELS_DIR.parent)}")
    print(f"  Samples  -> {SAMPLES_PATH.relative_to(MODELS_DIR.parent)}")
    print(f"\nTraining finished in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
