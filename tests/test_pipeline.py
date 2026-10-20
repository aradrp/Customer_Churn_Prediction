"""
Unit tests for src/features.py and src/evaluate.py.

Leakage is covered separately in tests/test_leakage.py.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import TARGET_COLUMN, clean_data, get_feature_columns, make_synthetic_telco_data, split_data
from src.evaluate import (
    evaluate_at_threshold, expected_business_cost, find_optimal_threshold,
    recommended_action, risk_group, threshold_search,
)
from src.features import DomainFeatureEngineer, build_full_pipeline


@pytest.fixture(scope="module")
def prepared_data():
    df = clean_data(make_synthetic_telco_data(n_rows=800, random_state=2))
    train_df, val_df, test_df = split_data(df, random_state=2)
    num_cols, cat_cols = get_feature_columns(train_df)
    feature_cols = num_cols + cat_cols
    return train_df, val_df, test_df, num_cols, cat_cols, feature_cols


def test_domain_feature_engineer_adds_expected_columns():
    df = clean_data(make_synthetic_telco_data(n_rows=50, random_state=3))
    engineer = DomainFeatureEngineer()
    transformed = engineer.fit_transform(df)
    for col in ["avg_charge_per_tenure_month", "num_subscribed_services", "has_internet",
                "is_new_customer", "tenure_bucket"]:
        assert col in transformed.columns


def test_domain_feature_engineer_handles_zero_tenure_without_error():
    df = clean_data(make_synthetic_telco_data(n_rows=50, random_state=3))
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = np.nan
    engineer = DomainFeatureEngineer()
    transformed = engineer.fit_transform(df)

    nonzero_tenure_mask = df["tenure"] > 0
    assert np.isfinite(
        transformed.loc[nonzero_tenure_mask, "avg_charge_per_tenure_month"]
    ).all()
    assert transformed.loc[~nonzero_tenure_mask, "avg_charge_per_tenure_month"].isna().all()


def test_pipeline_fits_and_predicts_probabilities(prepared_data):
    train_df, val_df, _, num_cols, cat_cols, feature_cols = prepared_data
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500), num_cols, cat_cols)
    pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMN])
    proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    assert proba.shape[0] == len(val_df)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_pipeline_handles_unseen_category_gracefully(prepared_data):
    train_df, val_df, _, num_cols, cat_cols, feature_cols = prepared_data
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500), num_cols, cat_cols)
    pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMN])

    novel_row = val_df[feature_cols].iloc[[0]].copy()
    novel_row["PaymentMethod"] = "Cryptocurrency (new)"  # unseen category
    # handle_unknown="ignore" should prevent an exception
    proba = pipeline.predict_proba(novel_row)[:, 1]
    assert proba.shape == (1,)


def test_threshold_search_returns_expected_columns_and_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_proba = rng.random(300)
    table = threshold_search(y_true, y_proba)
    assert set(["threshold", "precision", "recall", "f1", "expected_cost"]).issubset(table.columns)
    assert table["threshold"].min() >= 0.05
    assert table["threshold"].max() <= 0.95


def test_find_optimal_threshold_minimizes_cost():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    best_t, table = find_optimal_threshold(y_true, y_proba)
    assert table.iloc[0]["expected_cost"] == table["expected_cost"].min()
    assert 0.05 <= best_t <= 0.95


def test_expected_business_cost_formula():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])  # 1 FN, 1 FP
    cost = expected_business_cost(y_true, y_pred, fn_cost=500, fp_cost=30)
    assert cost == 500 + 30


@pytest.mark.parametrize("proba,expected", [(0.9, "High"), (0.5, "Medium"), (0.1, "Low")])
def test_risk_group_buckets(proba, expected):
    assert risk_group(proba) == expected


def test_recommended_action_is_nonempty_for_all_groups():
    for group in ["High", "Medium", "Low"]:
        assert len(recommended_action(group)) > 0
