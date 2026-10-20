"""
features.py
============
Custom scikit-learn-compatible transformers and pipeline factories for the
Telco Customer Churn project.

Everything here is designed to be fit ONLY inside cross-validation folds
(i.e. as steps of a Pipeline / ColumnTransformer) so nothing is learned from
data the model will later be evaluated on.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLUMNS_YES_NO = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds a handful of domain-specific engineered features on top of the raw
    columns: how much a customer pays relative to how long they have stayed,
    how many add-on services they have bundled, and which tenure "life stage"
    they are in.

    New features
    ------------
    - avg_charge_per_tenure_month : TotalCharges / max(tenure, 1)
    - num_subscribed_services     : count of "Yes" across service columns
    - has_internet                : binary indicator, InternetService != "No"
    - is_new_customer             : binary indicator, tenure <= 3 months
    - tenure_bucket               : categorical bucket of tenure

    This transformer is stateless (no statistics are learned from `fit`), so
    it is inherently leakage-safe and can be applied identically to any fold
    or to new data at inference time. It is still implemented as a proper
    Transformer (with fit/transform) so it composes cleanly inside a
    Pipeline and survives cross-validation / grid search cloning.

    Edge cases handled
    -------------------
    - tenure == 0 -> avoid division by zero by flooring the denominator at 1.
    - Missing/NaN TotalCharges -> propagated as NaN (handled later by the
      imputer in the numeric branch of the ColumnTransformer).
    - Unseen categorical values in service columns -> anything not exactly
      "Yes" is treated as not-subscribed (conservative, avoids KeyErrors).
    """

    def __init__(self, service_columns: Optional[List[str]] = None):
        self.service_columns = service_columns or SERVICE_COLUMNS_YES_NO

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        # Stateless: nothing to learn. Validate expected columns exist.
        missing = [c for c in ["tenure", "TotalCharges", "InternetService"] if c not in X.columns]
        if missing:
            raise ValueError(f"DomainFeatureEngineer missing required columns: {missing}")
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()

        safe_tenure = X["tenure"].clip(lower=1)
        X["avg_charge_per_tenure_month"] = (X["TotalCharges"] / safe_tenure).replace([np.inf, -np.inf], np.nan)

        present_service_cols = [c for c in self.service_columns if c in X.columns]
        X["num_subscribed_services"] = (X[present_service_cols] == "Yes").sum(axis=1)

        X["has_internet"] = (X["InternetService"] != "No").astype(int)
        X["is_new_customer"] = (X["tenure"] <= 3).astype(int)

        bins = [-1, 3, 12, 24, 48, np.inf]
        labels = ["0-3mo", "4-12mo", "13-24mo", "25-48mo", "49mo+"]
        X["tenure_bucket"] = pd.cut(X["tenure"], bins=bins, labels=labels).astype(str)

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + [
            "avg_charge_per_tenure_month", "num_subscribed_services",
            "has_internet", "is_new_customer", "tenure_bucket",
        ]


def build_preprocessor(
    numerical_columns: List[str],
    categorical_columns: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build the ColumnTransformer used inside every model Pipeline.

    Numerical branch:  SimpleImputer(median) [-> StandardScaler]
    Categorical branch: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')

    Parameters
    ----------
    numerical_columns, categorical_columns : list of column names
    scale_numeric : bool
        Tree-based models do not require scaling, but a shared preprocessor
        keeps the pipeline definitions simple and scaling a well-behaved
        numeric column is harmless for trees. Set False to skip scaling
        (e.g. if you want a leaner tree-only pipeline).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": at inference time an unseen category
        # (e.g. a new PaymentMethod added by the business later) is encoded
        # as all-zeros instead of raising an error, keeping the service
        # available rather than crashing on novel input.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(
    estimator,
    numerical_columns: List[str],
    categorical_columns: List[str],
    use_domain_features: bool = True,
    scale_numeric: bool = True,
) -> Pipeline:
    """
    Compose the complete leakage-safe pipeline:
        [optional DomainFeatureEngineer] -> ColumnTransformer -> estimator

    All fitting (imputation statistics, one-hot categories, scaler mean/std)
    happens inside `.fit()`, so calling this pipeline's `.fit()` only on a
    training fold guarantees no information from validation/test leaks in.
    """
    steps = []
    numerical = list(numerical_columns)
    categorical = list(categorical_columns)

    if use_domain_features:
        steps.append(("domain_features", DomainFeatureEngineer()))
        # engineered columns need to be added to the ColumnTransformer's column lists
        numerical = numerical + ["avg_charge_per_tenure_month", "num_subscribed_services",
                                  "has_internet", "is_new_customer"]
        categorical = categorical + ["tenure_bucket"]

    preprocessor = build_preprocessor(numerical, categorical, scale_numeric=scale_numeric)
    steps.append(("preprocessor", preprocessor))
    steps.append(("classifier", estimator))

    return Pipeline(steps)


def build_imbalanced_pipeline(
    estimator,
    numerical_columns: List[str],
    categorical_columns: List[str],
    resampler: str = "none",
    use_domain_features: bool = True,
    scale_numeric: bool = True,
    random_state: int = 42,
):
    """
    Same as `build_full_pipeline` but optionally inserts a resampler
    (SMOTE / RandomOverSampler) BETWEEN preprocessing and the classifier,
    using an imbalanced-learn Pipeline so resampling only ever touches
    training folds during cross-validation (never the held-out fold).

    Parameters
    ----------
    resampler : {"none", "class_weight", "smote", "random_oversample"}
        "class_weight" does not add a resampling step; instead the caller
        should pass an estimator already configured with
        class_weight="balanced".

    Returns
    -------
    A `sklearn.pipeline.Pipeline` (if resampler in {"none", "class_weight"})
    or an `imblearn.pipeline.Pipeline` (if resampler in {"smote",
    "random_oversample"}). If imbalanced-learn is not installed, raises a
    clear ImportError with installation instructions.
    """
    steps = []
    numerical = list(numerical_columns)
    categorical = list(categorical_columns)

    if use_domain_features:
        steps.append(("domain_features", DomainFeatureEngineer()))
        numerical = numerical + ["avg_charge_per_tenure_month", "num_subscribed_services",
                                  "has_internet", "is_new_customer"]
        categorical = categorical + ["tenure_bucket"]

    preprocessor = build_preprocessor(numerical, categorical, scale_numeric=scale_numeric)
    steps.append(("preprocessor", preprocessor))

    if resampler in ("none", "class_weight"):
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "imbalanced-learn is required for resampler != 'none'/'class_weight'. "
            "Install it with `pip install imbalanced-learn` and re-run."
        ) from exc

    if resampler == "smote":
        sampler = SMOTE(random_state=random_state)
    elif resampler == "random_oversample":
        sampler = RandomOverSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampler '{resampler}'")

    steps.append(("resample", sampler))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps)
