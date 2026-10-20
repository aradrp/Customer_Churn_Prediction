"""
predict.py
==========
Inference utilities: load the saved production pipeline and score new
customer records, returning churn probability, thresholded prediction, risk
group, and a recommended business action.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.evaluate import recommended_action, risk_group

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/churn_pipeline.joblib")
DEFAULT_METADATA_PATH = Path("models/model_metadata.joblib")

# The exact raw feature columns the model was trained on (excludes
# customerID and Churn). Used to validate incoming JSON/DataFrame requests.
REQUIRED_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


class InputValidationError(Exception):
    """Raised when an inference request does not match the expected schema."""


@dataclass
class ModelArtifact:
    """Bundle of the fitted pipeline plus metadata needed for safe inference."""
    pipeline: Any
    threshold: float
    model_version: str
    trained_at: str
    metrics_at_threshold: Dict[str, float]


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> ModelArtifact:
    """
    Load the full pipeline (preprocessing + classifier) and its metadata
    (frozen decision threshold, version, training metrics) from disk.
    """
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained pipeline found at {model_path}. Run `python src/train.py` first."
        )
    pipeline = joblib.load(model_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}

    return ModelArtifact(
        pipeline=pipeline,
        threshold=metadata.get("threshold", 0.5),
        model_version=metadata.get("model_version", "unknown"),
        trained_at=metadata.get("trained_at", "unknown"),
        metrics_at_threshold=metadata.get("metrics_at_threshold", {}),
    )


def validate_input_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Validate a list of raw JSON customer records and convert to a DataFrame
    ready for the pipeline.

    Raises
    ------
    InputValidationError
        If required fields are missing or of an unusable type.
    """
    if not records:
        raise InputValidationError("Request contained no customer records.")

    df = pd.DataFrame(records)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(f"Missing required fields: {missing}")

    # Coerce TotalCharges/MonthlyCharges to numeric; do not silently accept
    # garbage strings that would break the imputer/scaler at transform time.
    for col in ["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[REQUIRED_INPUT_COLUMNS]


def predict_batch(
    artifact: ModelArtifact,
    df: pd.DataFrame,
    customer_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Score a batch of customers with the loaded model artifact.

    Returns
    -------
    pd.DataFrame with columns:
        customerID (if provided), probability, prediction, risk_group,
        recommended_action
    """
    probabilities = artifact.pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= artifact.threshold).astype(int)
    risk_groups = [risk_group(p) for p in probabilities]
    actions = [recommended_action(r) for r in risk_groups]

    result = pd.DataFrame({
        "probability": probabilities.round(4),
        "prediction": predictions,
        "risk_group": risk_groups,
        "recommended_action": actions,
    })
    if customer_ids is not None:
        result.insert(0, "customerID", customer_ids)
    return result


def predict_single(artifact: ModelArtifact, record: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single customer record (used by the FastAPI endpoint)."""
    df = validate_input_records([record])
    result = predict_batch(artifact, df)
    row = result.iloc[0]
    return {
        "probability": float(row["probability"]),
        "prediction": int(row["prediction"]),
        "risk_group": row["risk_group"],
        "recommended_action": row["recommended_action"],
        "model_version": artifact.model_version,
        "threshold_used": artifact.threshold,
    }


def generate_final_risk_report(
    artifact: ModelArtifact,
    X: pd.DataFrame,
    y_true: np.ndarray,
    customer_ids: pd.Series,
    output_path: str | Path = "predictions.csv",
) -> pd.DataFrame:
    """
    Build the final `predictions.csv` deliverable: customerID, actual label,
    probability, thresholded prediction, risk group, recommended action.
    """
    scored = predict_batch(artifact, X, customer_ids=customer_ids.tolist())
    scored.insert(1, "actual_churn", np.asarray(y_true))
    scored = scored.sort_values("probability", ascending=False).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    logger.info("Final risk report written to %s (%d rows)", output_path, len(scored))
    return scored


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score customer records from a JSON file via the CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to a JSON file: a list of customer records")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA_PATH))
    args = parser.parse_args()

    art = load_model(args.model, args.metadata)
    with open(args.input) as f:
        recs = json.load(f)
    valid_df = validate_input_records(recs)
    out = predict_batch(art, valid_df)
    print(out.to_json(orient="records", indent=2))
