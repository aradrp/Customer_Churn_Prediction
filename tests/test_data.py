"""Unit tests for src/data.py"""
import numpy as np
import pandas as pd
import pytest

from src.data import (
    ID_COLUMN, TARGET_COLUMN, SchemaValidationError, audit_data, clean_data,
    get_feature_columns, make_synthetic_telco_data, split_data, validate_schema,
)


@pytest.fixture
def raw_df():
    return make_synthetic_telco_data(n_rows=500, random_state=1)


def test_synthetic_data_has_expected_shape(raw_df):
    assert len(raw_df) == 500
    assert TARGET_COLUMN in raw_df.columns
    assert ID_COLUMN in raw_df.columns


def test_validate_schema_passes_on_synthetic(raw_df):
    validate_schema(raw_df)  # should not raise


def test_validate_schema_raises_on_missing_columns(raw_df):
    broken = raw_df.drop(columns=["tenure"])
    with pytest.raises(SchemaValidationError):
        validate_schema(broken)


def test_clean_data_converts_total_charges_to_numeric(raw_df):
    cleaned = clean_data(raw_df)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_clean_data_maps_target_to_binary(raw_df):
    cleaned = clean_data(raw_df)
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})


def test_clean_data_zero_tenure_produces_missing_total_charges(raw_df):
    cleaned = clean_data(raw_df)
    zero_tenure_mask = cleaned["tenure"] == 0
    if zero_tenure_mask.any():
        assert cleaned.loc[zero_tenure_mask, "TotalCharges"].isna().all()


def test_audit_data_reports_prevalence_between_0_and_1(raw_df):
    report = audit_data(raw_df)
    assert 0.0 <= report.target_prevalence <= 1.0


def test_get_feature_columns_excludes_id_and_target(raw_df):
    cleaned = clean_data(raw_df)
    num_cols, cat_cols = get_feature_columns(cleaned)
    all_cols = set(num_cols) | set(cat_cols)
    assert ID_COLUMN not in all_cols
    assert TARGET_COLUMN not in all_cols


def test_split_data_is_stratified_and_disjoint(raw_df):
    cleaned = clean_data(raw_df)
    train_df, val_df, test_df = split_data(cleaned, test_size=0.15, val_size=0.15, random_state=1)

    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)

    train_ids = set(train_df[ID_COLUMN])
    val_ids = set(val_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    overall_rate = cleaned[TARGET_COLUMN].mean()
    for part in (train_df, val_df, test_df):
        assert abs(part[TARGET_COLUMN].mean() - overall_rate) < 0.1
