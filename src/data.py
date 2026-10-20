"""
data.py
=======
Data loading, schema validation, and leakage-safe cleaning utilities for the
Telco Customer Churn project.

Design principles
------------------
1. Every function that "learns" something from data (e.g. imputation values)
   must be expressed as a scikit-learn compatible transformer so it can live
   inside a Pipeline and be fit ONLY on training folds.
2. Functions in this module that run before the train/test split are strictly
   descriptive / deterministic (type coercion, dropping obviously broken
   columns) and never depend on statistics computed across the whole dataset.
3. customerID is never used as a model feature. It is retained separately so
   it can be re-attached to predictions for business reporting.

Author: ML Engineering Team
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RANDOM_SEED = 42
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Columns expected in the raw Kaggle Telco Customer Churn dataset.
EXPECTED_COLUMNS: List[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


class SchemaValidationError(Exception):
    """Raised when the input data does not match the expected schema."""


@dataclass
class DataQualityReport:
    """Container summarizing data quality findings for the audit notebook."""
    n_rows: int
    n_cols: int
    missing_by_column: dict
    duplicate_row_count: int
    duplicate_id_count: int
    total_charges_coerced_nulls: int
    zero_tenure_rows: int
    target_prevalence: float
    issues: List[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return the `issues` log as a tidy DataFrame for reporting."""
        return pd.DataFrame(self.issues, columns=["issue", "affected_rows", "decision", "rationale"])

    def add_issue(self, issue: str, affected_rows: int, decision: str, rationale: str) -> None:
        self.issues.append(
            {"issue": issue, "affected_rows": affected_rows, "decision": decision, "rationale": rationale}
        )


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV from disk.

    Parameters
    ----------
    path : str or Path
        Path to the raw CSV file (as downloaded from Kaggle, unmodified).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    SchemaValidationError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'. Download the Kaggle Telco "
            "Customer Churn CSV and place it at data/raw/telco_churn.csv, "
            "or run `python src/data.py --make-synthetic` to generate a "
            "structurally identical demo dataset."
        )

    df = pd.read_csv(path)
    validate_schema(df)
    logger.info("Loaded raw data: %d rows x %d columns from %s", df.shape[0], df.shape[1], path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure the dataframe contains the columns this project expects."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaValidationError(f"Missing expected columns: {sorted(missing)}")


def audit_data(df: pd.DataFrame) -> DataQualityReport:
    """
    Produce a structured data-quality report WITHOUT mutating the dataframe.
    This is intended to run on the full raw dataset (before splitting) since
    it only describes data quality; it does not learn any statistic that
    will be reused at inference time (that belongs in the sklearn Pipeline).
    """
    n_rows, n_cols = df.shape
    missing_by_column = df.isna().sum().to_dict()

    duplicate_row_count = int(df.duplicated().sum())
    duplicate_id_count = int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else -1

    coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
    total_charges_coerced_nulls = int(coerced.isna().sum() - df["TotalCharges"].isna().sum())

    zero_tenure_rows = int((df["tenure"] == 0).sum())

    target_prevalence = float((df[TARGET_COLUMN] == "Yes").mean()) if TARGET_COLUMN in df.columns else float("nan")

    report = DataQualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        missing_by_column=missing_by_column,
        duplicate_row_count=duplicate_row_count,
        duplicate_id_count=duplicate_id_count,
        total_charges_coerced_nulls=total_charges_coerced_nulls,
        zero_tenure_rows=zero_tenure_rows,
        target_prevalence=target_prevalence,
    )

    report.add_issue(
        issue="TotalCharges stored as string with blank entries",
        affected_rows=total_charges_coerced_nulls,
        decision="Coerce to numeric with errors='coerce'; impute inside pipeline (median, train-fold only)",
        rationale="Blank values correspond to customers with zero tenure (brand-new accounts); "
                   "coercion prevents the whole column being treated as categorical.",
    )
    report.add_issue(
        issue="Duplicate rows",
        affected_rows=duplicate_row_count,
        decision="Drop exact duplicates before splitting" if duplicate_row_count else "None found",
        rationale="Exact duplicate rows would let the same customer's information leak across "
                   "train/validation/test partitions.",
    )
    report.add_issue(
        issue="customerID uniqueness",
        affected_rows=duplicate_id_count,
        decision="Retain customerID for reporting only; excluded from model features",
        rationale="customerID is an arbitrary identifier with no predictive signal and would "
                   "act as a lookup key that overfits with high-cardinality encoders.",
    )
    report.add_issue(
        issue="Zero-tenure customers",
        affected_rows=zero_tenure_rows,
        decision="Keep, but flag as a small, plausible edge case",
        rationale="New customers naturally have tenure=0 and blank TotalCharges; this is a real "
                   "business segment, not a data error.",
    )
    return report


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Apply deterministic, leakage-free cleaning steps that do not require any
    statistic learned from the data (those live in the sklearn Pipeline).

    Steps
    -----
    1. Coerce TotalCharges to numeric.
    2. Drop exact duplicate rows.
    3. Normalize the target to {0, 1}.
    4. Strip whitespace from string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (post `load_raw_data`).
    drop_duplicates : bool
        Whether to drop exact duplicate rows.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe. TotalCharges may still contain NaN by design —
        imputation happens inside the modeling Pipeline so it is fit only on
        training folds.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d exact duplicate rows", dropped)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype("Int64")
        if df[TARGET_COLUMN].isna().any():
            raise SchemaValidationError("Target column contains values other than 'Yes'/'No' after mapping.")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if "SeniorCitizen" in df.columns:
        # Comes in as 0/1 int already in the Kaggle dataset; keep as-is but
        # ensure dtype is consistent regardless of source formatting.
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """
    Split feature columns into numerical and categorical groups.
    Excludes customerID and the target column.

    Returns
    -------
    (numerical_columns, categorical_columns)
    """
    exclude = {ID_COLUMN, TARGET_COLUMN}
    feature_df = df.drop(columns=[c for c in exclude if c in df.columns])

    numerical_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # SeniorCitizen is encoded as 0/1 but is semantically categorical/binary;
    # keep it numeric here since it is already a clean binary indicator and
    # scaling a binary column is harmless for linear models.
    return numerical_columns, categorical_columns


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED,
):
    """
    Create stratified train / validation / test splits.

    The split is performed in two stages so that all three partitions are
    stratified by the target:
        1. train+val vs test
        2. train vs val (from the remaining train+val)

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    from sklearn.model_selection import train_test_split

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df[target_col], random_state=random_state
    )

    logger.info(
        "Split sizes -> train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
        len(train_df), 100 * len(train_df) / len(df),
        len(val_df), 100 * len(val_df) / len(df),
        len(test_df), 100 * len(test_df) / len(df),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_synthetic_telco_data(n_rows: int = 3000, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the SCHEMA and rough statistical
    shape of the Kaggle Telco Customer Churn dataset.

    This is a fallback/demo utility ONLY. It exists so that:
      - The full pipeline (audit -> EDA -> modeling -> decisions -> reporting)
        can be run end-to-end in environments without internet access to
        download the real Kaggle CSV.
      - Unit tests do not depend on an external file.

    For an actual portfolio submission, replace data/raw/telco_churn.csv with
    the real dataset from:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

    Returns
    -------
    pd.DataFrame with the same columns as the real dataset.
    """
    rng = np.random.default_rng(random_state)
    n = n_rows

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n)
    phone_service = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n)
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def dependent_service(base_choices):
        out = rng.choice(base_choices, n)
        out = np.where(internet_service == "No", "No internet service", out)
        return out

    online_security = dependent_service(["Yes", "No"])
    online_backup = dependent_service(["Yes", "No"])
    device_protection = dependent_service(["Yes", "No"])
    tech_support = dependent_service(["Yes", "No"])
    streaming_tv = dependent_service(["Yes", "No"])
    streaming_movies = dependent_service(["Yes", "No"])

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )

    base_charge = rng.normal(65, 20, n).clip(18, 118)
    internet_bonus = np.where(internet_service == "Fiber optic", 25, np.where(internet_service == "DSL", 10, 0))
    monthly_charges = (base_charge + internet_bonus).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 50, n)).clip(min=0).round(2)
    # Real dataset encodes blanks for tenure == 0 as a single space " ",
    # NOT an empty string -- this matters because pandas' CSV reader treats
    # a truly empty field as NaN automatically, which would let the
    # TotalCharges coercion issue go unnoticed. A literal " " forces the
    # column to stay object/string dtype until we explicitly coerce it,
    # faithfully reproducing the real dataset's data-quality issue.
    total_charges_str = np.where(tenure == 0, " ", total_charges.astype(str))

    # Construct a churn probability with realistic, explainable drivers.
    logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.02 * (monthly_charges - 65)
        - 0.045 * tenure
        + 0.5 * (internet_service == "Fiber optic")
        + 0.35 * (payment_method == "Electronic check")
        - 0.3 * (tech_support == "Yes")
        + 0.2 * senior
        - 0.15 * (partner == "Yes")
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob_churn)
    churn_label = np.where(churn == 1, "Yes", "No")

    customer_id = [f"{rng.integers(1000,9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(n)]

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn_label,
    })
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data utilities CLI")
    parser.add_argument("--make-synthetic", action="store_true", help="Generate a synthetic demo dataset")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument(
        "--out", type=str, default="data/raw/telco_churn.csv",
        help="Output path for the synthetic CSV",
    )
    args = parser.parse_args()

    if args.make_synthetic:
        synthetic = make_synthetic_telco_data(n_rows=args.rows)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(out_path, index=False)
        logger.info("Synthetic dataset written to %s (%d rows)", out_path, len(synthetic))
