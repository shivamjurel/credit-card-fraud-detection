"""
Pydantic schemas = the SHAPE of data going in and out of the API.

FastAPI uses them to:
  * validate incoming JSON automatically (wrong type / missing field -> HTTP 422)
  * document the API at http://localhost:8000/docs

Privacy note: we NEVER accept card numbers, CVV, PIN, OTP, expiry dates or
bank credentials - only anonymised / behavioural transaction features.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# V1..V28 are anonymised PCA components. In the dataset they lie roughly
# between -115 and +120, so +/-500 is a generous sanity limit.
PCAFeature = Annotated[float, Field(ge=-500, le=500, allow_inf_nan=False)]

MERCHANT_CATEGORIES = Literal[
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home",
    "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]


class TransactionInput(BaseModel):
    """Body of POST /predict/ulb - Dataset 1 (ULB, anonymised PCA features)."""

    time: float = Field(..., ge=0, le=1_000_000, allow_inf_nan=False,
                        description="Seconds since the first transaction in the dataset (0 to 172,792).")
    amount: float = Field(..., ge=0, le=1_000_000, allow_inf_nan=False, description="Transaction amount (>= 0).")
    v1: PCAFeature
    v2: PCAFeature
    v3: PCAFeature
    v4: PCAFeature
    v5: PCAFeature
    v6: PCAFeature
    v7: PCAFeature
    v8: PCAFeature
    v9: PCAFeature
    v10: PCAFeature
    v11: PCAFeature
    v12: PCAFeature
    v13: PCAFeature
    v14: PCAFeature
    v15: PCAFeature
    v16: PCAFeature
    v17: PCAFeature
    v18: PCAFeature
    v19: PCAFeature
    v20: PCAFeature
    v21: PCAFeature
    v22: PCAFeature
    v23: PCAFeature
    v24: PCAFeature
    v25: PCAFeature
    v26: PCAFeature
    v27: PCAFeature
    v28: PCAFeature

    model_config = ConfigDict(
        json_schema_extra={"example": {"time": 10000, "amount": 250.50, **{f"v{i}": 0.0 for i in range(1, 29)}}}
    )


class SparkovTransactionInput(BaseModel):
    """Body of POST /predict/sparkov - Dataset 2 (readable transaction + card-history features)."""

    amount: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False, description="Transaction amount (> 0).")
    category: MERCHANT_CATEGORIES = Field(..., description="Merchant category.")
    gender: Literal["M", "F"]
    age: int = Field(..., ge=14, le=110, description="Customer age in years.")
    hour: int = Field(..., ge=0, le=23, description="Hour of the transaction (0-23).")
    day_of_week: int = Field(..., ge=1, le=7, description="1 = Sunday ... 7 = Saturday.")
    city_population: int = Field(..., ge=0, le=50_000_000)
    distance_km: float = Field(..., ge=0, le=20_000, allow_inf_nan=False,
                               description="Distance between customer's home and the merchant.")
    txn_count_24h: int = Field(..., ge=0, le=10_000, description="Card's transactions in the previous 24 hours.")
    amount_last_24h: float = Field(..., ge=0, le=10_000_000, allow_inf_nan=False,
                                   description="Amount spent with the card in the previous 24 hours.")
    hours_since_last: float = Field(..., ge=0, le=100_000, allow_inf_nan=False,
                                    description="Hours since the card's previous transaction.")
    card_avg_amount: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False,
                                   description="The card's usual (average) transaction amount.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 845.2, "category": "shopping_net", "gender": "F", "age": 52, "hour": 23, "day_of_week": 6,
                "city_population": 12000, "distance_km": 80.5, "txn_count_24h": 6, "amount_last_24h": 1900.0,
                "hours_since_last": 0.4, "card_avg_amount": 70.0,
            }
        }
    )


class RiskThresholds(BaseModel):
    medium: float
    high: float


class PredictionResponse(BaseModel):
    """Returned by POST /predict/*."""

    transaction_id: int
    dataset: str
    model_name: str
    prediction: Literal["Fraud", "Legitimate"]
    fraud_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    risk_thresholds: RiskThresholds
    amount: float
    category: str | None = None
    created_at: datetime


class TransactionOut(BaseModel):
    """One row from the transactions table."""

    model_config = ConfigDict(from_attributes=True)  # allow building from ORM objects

    id: int
    dataset: str
    source: str
    amount: float
    category: str | None = None
    prediction: str
    fraud_probability: float
    risk_level: str
    created_at: datetime


class TransactionDetail(TransactionOut):
    """Single transaction including all input features."""

    model_name: str | None = None
    features: dict[str, float | str] | None = None
    risk_thresholds: RiskThresholds | None = None


class TimelinePoint(BaseModel):
    period: str
    fraud: int
    legitimate: int
    total: int


class DashboardStats(BaseModel):
    """Returned by GET /dashboard - all numbers come from the SQLite database."""

    total_transactions: int
    fraud_transactions: int
    legitimate_transactions: int
    fraud_percentage: float
    average_fraud_probability: float
    total_amount: float
    fraud_amount: float
    risk_distribution: dict[str, int]
    by_dataset: dict[str, int]
    by_source: dict[str, int]
    fraud_by_category: dict[str, int]
    timeline_granularity: Literal["minute", "hour", "day"]
    timeline: list[TimelinePoint]
