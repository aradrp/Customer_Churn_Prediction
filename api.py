"""
api.py
======
Lightweight FastAPI deployment for the Telco Churn model.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints
---------
GET  /health              -> liveness check + model version
POST /predict              -> single-customer churn prediction
POST /predict/batch        -> batch prediction for a list of customers

Example request body for /predict:
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 427.5
}
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import (
    InputValidationError, ModelArtifact, load_model,
    predict_batch, validate_input_records,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts churn probability and recommends retention actions.",
    version="1.0.0",
)

_artifact: Optional[ModelArtifact] = None


class CustomerRecord(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    risk_group: str
    recommended_action: str
    model_version: str
    threshold_used: float


class BatchRequest(BaseModel):
    customers: List[CustomerRecord]
    customer_ids: Optional[List[str]] = None


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the trained pipeline once at startup rather than per-request."""
    global _artifact
    try:
        _artifact = load_model()
        logger.info("Model loaded: version=%s, threshold=%.2f", _artifact.model_version, _artifact.threshold)
    except FileNotFoundError as exc:
        logger.error("Model artifact not found: %s. Run `python -m src.train --synthetic` first.", exc)
        _artifact = None


def _require_model() -> ModelArtifact:
    if _artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train and save a pipeline with `python -m src.train` first.",
        )
    return _artifact


@app.get("/health")
def health():
    """Liveness check; reports whether a model is loaded and its version."""
    return {
        "status": "ok" if _artifact is not None else "model_not_loaded",
        "model_version": _artifact.model_version if _artifact else None,
        "threshold": _artifact.threshold if _artifact else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRecord):
    """Score a single customer and return probability, risk group, and action."""
    artifact = _require_model()
    try:
        df = validate_input_records([customer.model_dump()])
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = predict_batch(artifact, df)
    row = result.iloc[0]
    return PredictionResponse(
        probability=float(row["probability"]),
        prediction=int(row["prediction"]),
        risk_group=row["risk_group"],
        recommended_action=row["recommended_action"],
        model_version=artifact.model_version,
        threshold_used=artifact.threshold,
    )


@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchRequest):
    """Score a batch of customers in one request."""
    artifact = _require_model()
    try:
        df = validate_input_records([c.model_dump() for c in request.customers])
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = predict_batch(artifact, df, customer_ids=request.customer_ids)
    return result.to_dict(orient="records")
