"""
CreditGuard - FastAPI backend (API routes only).

Run from the backend/ folder:
    uvicorn main:app --reload

Interactive API docs: http://localhost:8000/docs

Request flow for POST /predict/{dataset}:
    React (Axios)  ->  Pydantic validation (schemas.py)
                   ->  predictor.py (feature preparation + Spark PipelineModel)
                   ->  crud.py (save to SQLite)
                   ->  JSON response back to React
"""

import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import config
import crud
from database import get_db, init_db
from predictor import ModelNotLoadedError, predictor
from schemas import (
    DashboardStats,
    PredictionResponse,
    RiskThresholds,
    SparkovTransactionInput,
    TransactionDetail,
    TransactionInput,
    TransactionOut,
)

DatasetName = Literal["ulb", "sparkov"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: create DB tables, start Spark, load the ML models."""
    init_db()
    print("[startup] Starting Spark and loading models (takes ~15 seconds)...")
    predictor.load()
    for dataset in config.DATASETS:
        if predictor.is_loaded(dataset):
            print(f"[startup] {dataset}: {predictor.metadata[dataset]['selected_model']} loaded")
    if predictor.load_error:
        print(f"[startup] WARNING - {predictor.load_error}")
    yield
    predictor.stop()


app = FastAPI(
    title="CreditGuard API",
    description="Academic Big Data / ML demo for credit card fraud detection (PySpark models). "
                "Not for real financial decisions.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: allow the React dev server (a different port) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Error handlers -> meaningful JSON errors instead of raw stack traces
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"] if part not in ("body", "query", "path"))
        errors.append({"field": field or "body", "message": error["msg"]})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid input. Please check the highlighted fields.", "errors": errors},
    )


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Database error. Please try again."},
    )


# --------------------------------------------------------------------------
# General endpoints
# --------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {"message": "CreditGuard API is running", "docs": "/docs"}


@app.get("/health", tags=["General"])
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
        "models": {dataset: predictor.is_loaded(dataset) for dataset in config.DATASETS},
        "model_error": predictor.load_error,
    }


@app.get("/model-info", tags=["General"])
def model_info():
    """Selected Spark models + the full 5-model comparison for both datasets."""
    comparison = json.loads(config.MODEL_COMPARISON_PATH.read_text()) if config.MODEL_COMPARISON_PATH.exists() else {}
    datasets = {}
    for dataset, name in config.DATASETS.items():
        metadata = predictor.metadata.get(dataset, {})
        datasets[dataset] = {
            "dataset_name": name,
            "selected_model": metadata.get("selected_model"),
            "risk_thresholds": predictor.thresholds(dataset) if predictor.is_loaded(dataset) else None,
            "feature_importances": metadata.get("feature_importances", {}),
            "models": comparison.get(dataset, {}).get("models", {}),
            "trained_at": metadata.get("trained_at"),
        }
    return {"spark_version": predictor.spark.version if predictor.spark else None,
            "medium_risk_factor": config.MEDIUM_RISK_FACTOR, "datasets": datasets}


@app.get("/samples", tags=["General"])
def sample_transactions():
    """Real rows from each dataset's held-out TEST set, used by the 'Load sample' buttons."""
    if not config.SAMPLES_PATH.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="No samples found. Run python spark_pipeline/s04b_demo_samples.py")
    return json.loads(config.SAMPLES_PATH.read_text())


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def run_prediction(dataset: str, transaction_data: dict, db: Session) -> PredictionResponse:
    """Shared logic: Spark model -> save to SQLite -> response."""
    try:
        result = predictor.predict(dataset, transaction_data)
    except ModelNotLoadedError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))

    record = crud.create_transaction(db, dataset, transaction_data, result, source="web")
    return PredictionResponse(
        transaction_id=record.id,
        dataset=dataset,
        model_name=result["model_name"],
        prediction=record.prediction,
        fraud_probability=record.fraud_probability,
        risk_level=record.risk_level,
        risk_thresholds=predictor.thresholds(dataset),
        amount=record.amount,
        category=record.category,
        created_at=record.created_at,
    )


@app.post("/predict/sparkov", response_model=PredictionResponse, tags=["Prediction"])
def predict_sparkov(transaction: SparkovTransactionInput, db: Session = Depends(get_db)):
    """Dataset 2: readable transaction details + card history."""
    return run_prediction("sparkov", transaction.model_dump(), db)


@app.post("/predict/ulb", response_model=PredictionResponse, tags=["Prediction"])
def predict_ulb(transaction: TransactionInput, db: Session = Depends(get_db)):
    """Dataset 1: anonymised PCA features (Time, Amount, V1-V28)."""
    return run_prediction("ulb", transaction.model_dump(), db)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], include_in_schema=False)
def predict_legacy(transaction: TransactionInput, db: Session = Depends(get_db)):
    """Backwards-compatible alias for /predict/ulb."""
    return run_prediction("ulb", transaction.model_dump(), db)


# --------------------------------------------------------------------------
# Transaction history
# --------------------------------------------------------------------------
@app.get("/transactions", response_model=list[TransactionOut], tags=["Transactions"])
def list_transactions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    dataset: DatasetName | None = None,
    source: Literal["web", "stream"] | None = None,
    db: Session = Depends(get_db),
):
    return crud.get_transactions(db, limit=limit, offset=offset, dataset=dataset, source=source)


@app.get("/transactions/{transaction_id}", response_model=TransactionDetail, tags=["Transactions"])
def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    record = crud.get_transaction(db, transaction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Transaction {transaction_id} not found")

    detail = TransactionDetail.model_validate(record)
    detail.features = json.loads(record.features_json) if record.features_json else None
    if predictor.is_loaded(record.dataset):
        detail.risk_thresholds = RiskThresholds(**predictor.thresholds(record.dataset))
    return detail


@app.delete("/transactions/{transaction_id}", tags=["Transactions"])
def delete_transaction(transaction_id: int, db: Session = Depends(get_db)):
    if not crud.delete_transaction(db, transaction_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Transaction {transaction_id} not found")
    return {"message": f"Transaction {transaction_id} deleted"}


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
@app.get("/dashboard", response_model=DashboardStats, tags=["Dashboard"])
def dashboard(dataset: DatasetName | None = None, db: Session = Depends(get_db)):
    return crud.get_dashboard_stats(db, dataset=dataset)
