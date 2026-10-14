"""
Shared data-preparation helpers used by BOTH train_model.py and evaluate_model.py.

Keeping these steps in one place guarantees that training and evaluation
load, clean and split the data in exactly the same way.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, so scripts work from anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SAMPLES_PATH = MODELS_DIR / "sample_transactions.json"

# The 30 input features, in the exact order the model expects them.
# The backend uses the same order when it builds a feature vector.
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"  # 0 = Legitimate, 1 = Fraud

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Step 1: Load the CSV dataset with Pandas."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'creditcard.csv' (Kaggle: mlg-ulb/creditcardfraud) and place it in the data/ folder."
        )
    df = pd.read_csv(path)

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 3: Handle missing values and duplicate rows.

    - Missing values: this dataset normally has none, but if any appear we
      drop those rows (there are very few, and inventing values for anonymised
      PCA features would not be meaningful).
    - Duplicates: exact duplicate rows add no new information and can leak
      between train and test sets, so we remove them.
    """
    rows_before = len(df)

    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        df = df.dropna()
        if verbose:
            print(f"  Dropped rows with missing values ({missing_total} missing cells found).")
    elif verbose:
        print("  No missing values found - nothing to fix.")

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"  Removed {duplicate_count} duplicate rows.")
        print(f"  Rows: {rows_before} -> {len(df)}")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate the input features (X) from the label (y)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(X, y):
    """
    Step 6: 80% training / 20% testing split.

    stratify=y keeps the same fraud percentage in both sets. Without it, the
    tiny fraud class could end up very unevenly split.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
