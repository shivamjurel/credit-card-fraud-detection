"""
CRUD = Create, Read, Update, Delete.

All database operations live here, separate from the API routes in main.py.
Routes call these functions; they never write SQL themselves.
"""

import json
from datetime import timedelta

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from models import Transaction


def create_transaction(db: Session, dataset: str, transaction_input: dict, prediction_result: dict,
                       source: str = "web") -> Transaction:
    """Save one prediction (input + model output) to the database."""
    record = Transaction(
        dataset=dataset,
        source=source,
        amount=transaction_input["amount"],
        category=transaction_input.get("category"),
        prediction=prediction_result["prediction"],
        fraud_probability=prediction_result["fraud_probability"],
        risk_level=prediction_result["risk_level"],
        model_name=prediction_result.get("model_name"),
        features_json=json.dumps(transaction_input),
    )
    db.add(record)
    db.commit()
    db.refresh(record)  # reload to get the auto-generated id and created_at
    return record


def _filtered(query, dataset: str | None, source: str | None = None):
    if dataset:
        query = query.where(Transaction.dataset == dataset)
    if source:
        query = query.where(Transaction.source == source)
    return query


def get_transactions(db: Session, limit: int = 100, offset: int = 0, dataset: str | None = None,
                     source: str | None = None) -> list[Transaction]:
    """Most recent transactions first."""
    query = _filtered(select(Transaction), dataset, source)
    query = query.order_by(Transaction.created_at.desc(), Transaction.id.desc())
    return list(db.scalars(query.offset(offset).limit(limit)))


def get_transaction(db: Session, transaction_id: int) -> Transaction | None:
    return db.get(Transaction, transaction_id)


def delete_transaction(db: Session, transaction_id: int) -> bool:
    """Returns True if a row was deleted, False if the id did not exist."""
    record = db.get(Transaction, transaction_id)
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True


def _count_by(db: Session, column, dataset: str | None, only_fraud: bool = False) -> dict:
    query = _filtered(select(column, func.count(Transaction.id)), dataset)
    if only_fraud:
        query = query.where(Transaction.prediction == "Fraud")
    return {key: count for key, count in db.execute(query.group_by(column)) if key is not None}


def get_dashboard_stats(db: Session, dataset: str | None = None) -> dict:
    """Compute dashboard statistics from the stored predictions (optionally for one dataset)."""
    is_fraud = case((Transaction.prediction == "Fraud", 1), else_=0)
    fraud_amount = case((Transaction.prediction == "Fraud", Transaction.amount), else_=0.0)

    total, fraud, average_probability, total_amount, fraud_total_amount, first_time, last_time = db.execute(
        _filtered(
            select(
                func.count(Transaction.id),
                func.coalesce(func.sum(is_fraud), 0),
                func.coalesce(func.avg(Transaction.fraud_probability), 0.0),
                func.coalesce(func.sum(Transaction.amount), 0.0),
                func.coalesce(func.sum(fraud_amount), 0.0),
                func.min(Transaction.created_at),
                func.max(Transaction.created_at),
            ),
            dataset,
        )
    ).one()

    risk_distribution = {"Low": 0, "Medium": 0, "High": 0}
    risk_distribution.update(_count_by(db, Transaction.risk_level, dataset))

    # Timeline: choose the bucket size from how much time the data covers
    #   <= 2 hours -> per minute,  <= 2 days -> per hour,  otherwise -> per day
    granularity = "day"
    if first_time and last_time:
        time_span = last_time - first_time
        if time_span <= timedelta(hours=2):
            granularity = "minute"
        elif time_span <= timedelta(days=2):
            granularity = "hour"
    period_format = {"minute": "%Y-%m-%d %H:%M", "hour": "%Y-%m-%d %H:00", "day": "%Y-%m-%d"}[granularity]
    period = func.strftime(period_format, Transaction.created_at)

    timeline_query = _filtered(
        select(period.label("period"), func.sum(is_fraud).label("fraud"), func.count(Transaction.id).label("total")),
        dataset,
    ).group_by(period).order_by(period)
    timeline = [
        {"period": row.period, "fraud": row.fraud, "legitimate": row.total - row.fraud, "total": row.total}
        for row in db.execute(timeline_query)
    ]

    return {
        "total_transactions": total,
        "fraud_transactions": fraud,
        "legitimate_transactions": total - fraud,
        "fraud_percentage": round(fraud / total * 100, 2) if total else 0.0,
        "average_fraud_probability": round(float(average_probability), 4),
        "total_amount": round(float(total_amount), 2),
        "fraud_amount": round(float(fraud_total_amount), 2),
        "risk_distribution": risk_distribution,
        "by_dataset": _count_by(db, Transaction.dataset, dataset),
        "by_source": _count_by(db, Transaction.source, dataset),
        "fraud_by_category": _count_by(db, Transaction.category, dataset, only_fraud=True),
        "timeline_granularity": granularity,
        "timeline": timeline,
    }
