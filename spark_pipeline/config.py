"""Paths and settings shared by all Spark pipeline scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Raw input data ----
DATA_DIR = PROJECT_ROOT / "data"
ULB_RAW = DATA_DIR / "creditcard.csv"                  # Dataset 1: European cardholders (PCA features)
SPARKOV_TRAIN_RAW = DATA_DIR / "sparkov" / "fraudTrain.csv"  # Dataset 2: simulated US card transactions
SPARKOV_TEST_RAW = DATA_DIR / "sparkov" / "fraudTest.csv"

# ---- Processed data (Parquet = compressed, columnar, the standard Big Data file format) ----
PROCESSED_DIR = DATA_DIR / "processed"
ULB_CLEAN = PROCESSED_DIR / "ulb_clean.parquet"
SPARKOV_CLEAN = PROCESSED_DIR / "sparkov_clean.parquet"
ULB_FEATURES = PROCESSED_DIR / "ulb_features.parquet"
SPARKOV_FEATURES = PROCESSED_DIR / "sparkov_features.parquet"

# ---- Outputs ----
SPARK_MODELS_DIR = PROJECT_ROOT / "models" / "spark"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
POWERBI_DIR = PROJECT_ROOT / "powerbi" / "data"
STREAM_DIR = DATA_DIR / "stream"

DATASETS = {
    "ulb": {"name": "ULB European Credit Card (PCA)", "features": ULB_FEATURES},
    "sparkov": {"name": "Sparkov Simulated Card Transactions", "features": SPARKOV_FEATURES},
}

LABEL = "label"          # 0 = legitimate, 1 = fraud (renamed from Class / is_fraud)
SEED = 42


def ensure_dirs() -> None:
    for folder in [PROCESSED_DIR, SPARK_MODELS_DIR, FIGURES_DIR, RESULTS_DIR, POWERBI_DIR, STREAM_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def spark_path(path: Path) -> str:
    """Path string for Spark (forward slashes work on every OS)."""
    return path.as_posix()
