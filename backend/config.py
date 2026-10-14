"""
Central configuration for the backend.

Every value can be overridden with an environment variable, e.g.
    set MEDIUM_RISK_FACTOR=0.4   (Windows)
    export MEDIUM_RISK_FACTOR=0.4 (Linux/Mac)

NOTE: Risk levels are for DEMONSTRATION purposes only - they are NOT real banking rules.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# ---- Spark pipeline + trained Spark models (produced by spark_pipeline/) ----
PIPELINE_DIR = PROJECT_ROOT / "spark_pipeline"
SPARK_MODELS_DIR = Path(os.getenv("SPARK_MODELS_DIR", PROJECT_ROOT / "models" / "spark"))
SAMPLES_PATH = SPARK_MODELS_DIR / "samples.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "results" / "model_comparison.json"

DATASETS = {
    "ulb": "ULB European Credit Card (PCA)",
    "sparkov": "Sparkov Simulated Card Transactions",
}

# ---- SQLite database file ----
DATABASE_PATH = Path(os.getenv("FRAUD_DB_PATH", BACKEND_DIR / "fraud_detection.db"))

# ---- Risk levels (demo) ----
# Each model has its own tuned decision threshold T (chosen on validation data, see s04_train_models.py).
#   score >= T                        -> High   (predicted Fraud)
#   T * MEDIUM_RISK_FACTOR <= score   -> Medium (suspicious, but below the fraud threshold)
#   otherwise                         -> Low
MEDIUM_RISK_FACTOR = float(os.getenv("MEDIUM_RISK_FACTOR", 0.5))
if not 0 < MEDIUM_RISK_FACTOR < 1:
    raise ValueError("MEDIUM_RISK_FACTOR must be between 0 and 1")

# ---- Spark resources for the API (small: it only scores one transaction at a time) ----
SPARK_API_CORES = os.getenv("SPARK_API_CORES", "4")
SPARK_API_MEMORY = os.getenv("SPARK_API_MEMORY", "2g")

# ---- Frontend origins allowed to call this API (CORS) ----
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
