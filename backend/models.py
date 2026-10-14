"""
Database table definitions (SQLAlchemy ORM).

Note: these are DATABASE models (tables), not ML models.
The ML models live in models/spark/ and are used by predictor.py.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)        # "ulb" / "sparkov"
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="web")     # "web" / "stream"
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str | None] = mapped_column(String(40), nullable=True)            # Sparkov only
    prediction: Mapped[str] = mapped_column(String(20), nullable=False, index=True)     # "Fraud" / "Legitimate"
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)                 # "Low" / "Medium" / "High"
    model_name: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False, index=True)

    # All input features stored as JSON text, so a record can be viewed in full later
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)
