"""
Leakage tests.

The point of these tests is that each one can actually FAIL. A leakage test
that refits the same pipeline on the same training data twice and compares
the results passes no matter what the code does, so it protects nothing.

Strategy used here:

1. Make the validation split statistically very different from the training
   split (shifted TotalCharges, shifted tenure, a payment method that exists
   only in validation). If anything ever learns from validation rows, the
   fitted medians / means / category lists change in a visible way.
2. Compare fitted preprocessing state against statistics computed
   independently with pandas, on the training rows only.
3. Include a negative control: a deliberately leaky fit (pooled train + val)
   must be flagged by the very same check. That proves the check is
   sensitive and not just agreeing with whatever the code produces.
4. Exercise the real code paths in src/train.py (compare_models,
   tune_finalist), not only hand-built pipelines.
5. Use a spy step inside cross-validation to record which rows each fold's
   preprocessing was actually fit on.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import TARGET_COLUMN, clean_data, get_feature_columns, make_synthetic_telco_data, split_data
from src.features import build_full_pipeline, build_imbalanced_pipeline
from src.train import compare_models, tune_finalist

VAL_ONLY_PAYMENT_METHOD = "Payment method that only exists in validation"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shifted_splits():
    """Train/val/test where validation looks very different from training."""
    df = clean_data(make_synthetic_telco_data(n_rows=900, random_state=7))
    train_df, val_df, test_df = split_data(df, random_state=7)
    num_cols, cat_cols = get_feature_columns(train_df)
    feature_cols = num_cols + cat_cols

    val_df = val_df.copy()
    val_df["TotalCharges"] = val_df["TotalCharges"] * 10 + 10_000  # NaN stays NaN
    val_df["tenure"] = val_df["tenure"] + 100
    val_df["PaymentMethod"] = VAL_ONLY_PAYMENT_METHOD
    # Make sure validation has missing values to impute, so that any imputer
    # that learns from the batch it is scoring behaves differently from one
    # that uses training statistics.
    val_df.loc[val_df.index[::5], "TotalCharges"] = np.nan

    return {
        "X_train": train_df[feature_cols].reset_index(drop=True),
        "y_train": train_df[TARGET_COLUMN].to_numpy(),
        "X_val": val_df[feature_cols].reset_index(drop=True),
        "y_val": val_df[TARGET_COLUMN].to_numpy(),
        "X_test": test_df[feature_cols].reset_index(drop=True),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fitted_state(pipeline):
    """Pull the learned preprocessing state out of a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    _, num_pipe, num_columns = preprocessor.transformers_[0]
    _, cat_pipe, cat_columns = preprocessor.transformers_[1]
    imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]
    onehot = cat_pipe.named_steps["onehot"]
    return {
        "median_total_charges": imputer.statistics_[num_columns.index("TotalCharges")],
        "mean_tenure": scaler.mean_[num_columns.index("tenure")],
        "mean_total_charges": scaler.mean_[num_columns.index("TotalCharges")],
        "payment_methods": set(onehot.categories_[cat_columns.index("PaymentMethod")]),
    }


def _expected_state(X):
    """The same statistics, computed independently with pandas on rows X."""
    median_tc = X["TotalCharges"].median()
    return {
        "median_total_charges": median_tc,
        "mean_tenure": X["tenure"].mean(),
        "mean_total_charges": X["TotalCharges"].fillna(median_tc).mean(),
        "payment_methods": set(X["PaymentMethod"].unique()),
    }


def _states_equal(a, b):
    return (
        np.isclose(a["median_total_charges"], b["median_total_charges"], rtol=1e-9)
        and np.isclose(a["mean_tenure"], b["mean_tenure"], rtol=1e-9)
        and np.isclose(a["mean_total_charges"], b["mean_total_charges"], rtol=1e-9)
        and a["payment_methods"] == b["payment_methods"]
    )


def _learned_only_from_train(pipeline, X_train, X_val):
    """True if the fitted state matches the train-only statistics, False if it
    matches anything else (for example train + validation pooled)."""
    return _states_equal(_fitted_state(pipeline), _expected_state(X_train))


class _FitIndexSpy(BaseEstimator, TransformerMixin):
    """Pass-through step that remembers which row labels it was fit on."""

    def fit(self, X, y=None):
        self.fit_index_ = frozenset(X.index)
        return self

    def transform(self, X):
        return X


class _SampleCountingClassifier(DummyClassifier):
    """Classifier that records how many rows it was fit on."""

    def fit(self, X, y, sample_weight=None):
        self.n_fit_rows_ = X.shape[0]
        return super().fit(X, y, sample_weight)


# ---------------------------------------------------------------------------
# The test data itself must be able to expose leakage
# ---------------------------------------------------------------------------
def test_fixture_makes_train_only_and_pooled_statistics_clearly_different(shifted_splits):
    X_train, X_val = shifted_splits["X_train"], shifted_splits["X_val"]
    pooled = pd.concat([X_train, X_val], ignore_index=True)
    train_state, pooled_state = _expected_state(X_train), _expected_state(pooled)

    assert not np.isclose(train_state["median_total_charges"], pooled_state["median_total_charges"], rtol=0.05)
    assert not np.isclose(train_state["mean_tenure"], pooled_state["mean_tenure"], rtol=0.05)
    assert VAL_ONLY_PAYMENT_METHOD in pooled_state["payment_methods"]
    assert VAL_ONLY_PAYMENT_METHOD not in train_state["payment_methods"]


# ---------------------------------------------------------------------------
# Fitted state: train only
# ---------------------------------------------------------------------------
def test_pipeline_fit_on_train_learns_only_training_statistics(shifted_splits):
    s = shifted_splits
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500), s["num_cols"], s["cat_cols"])
    pipeline.fit(s["X_train"], s["y_train"])
    assert _learned_only_from_train(pipeline, s["X_train"], s["X_val"])


def test_negative_control_leaky_fit_is_detected(shifted_splits):
    """If someone fits on train + validation, the check above must say so."""
    s = shifted_splits
    pooled_X = pd.concat([s["X_train"], s["X_val"]], ignore_index=True)
    pooled_y = np.concatenate([s["y_train"], s["y_val"]])

    leaky = build_full_pipeline(LogisticRegression(max_iter=500), s["num_cols"], s["cat_cols"])
    leaky.fit(pooled_X, pooled_y)

    assert not _learned_only_from_train(leaky, s["X_train"], s["X_val"])
    assert VAL_ONLY_PAYMENT_METHOD in _fitted_state(leaky)["payment_methods"]


def test_predicting_on_new_data_does_not_change_fitted_state(shifted_splits):
    """transform/predict must never re-learn anything from the rows it scores."""
    s = shifted_splits
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500), s["num_cols"], s["cat_cols"])
    pipeline.fit(s["X_train"], s["y_train"])
    before = _fitted_state(pipeline)

    pipeline.predict_proba(s["X_val"])
    pipeline.predict_proba(s["X_test"])

    assert _states_equal(before, _fitted_state(pipeline))


def test_each_prediction_is_independent_of_the_other_rows_in_the_batch(shifted_splits):
    """Scoring a row alone or inside a batch must give the same probability.

    This catches "learning at transform time", e.g. imputing or scaling with
    statistics computed from the batch being scored instead of the training
    data. Such a bug is invisible in the fitted state but changes predictions.
    """
    s = shifted_splits
    pipeline = build_full_pipeline(LogisticRegression(max_iter=500), s["num_cols"], s["cat_cols"])
    pipeline.fit(s["X_train"], s["y_train"])

    batch = s["X_val"].iloc[:25]
    in_batch = pipeline.predict_proba(batch)[:, 1]
    alone = np.array([pipeline.predict_proba(batch.iloc[[i]])[0, 1] for i in range(len(batch))])

    np.testing.assert_allclose(in_batch, alone, rtol=1e-9)


# ---------------------------------------------------------------------------
# The real code paths in src/train.py
# ---------------------------------------------------------------------------
def test_compare_models_fits_on_train_only(shifted_splits):
    s = shifted_splits
    models = {"logistic_regression": LogisticRegression(max_iter=500)}
    _, fitted = compare_models(
        models, s["X_train"], s["y_train"], s["X_val"], s["y_val"], s["num_cols"], s["cat_cols"]
    )
    assert _learned_only_from_train(fitted["logistic_regression"], s["X_train"], s["X_val"])


def test_tune_finalist_refits_best_model_on_train_only(shifted_splits):
    s = shifted_splits
    search = tune_finalist(
        "logistic_regression", LogisticRegression(max_iter=500), {"C": [0.1, 1.0]},
        s["X_train"], s["y_train"], s["num_cols"], s["cat_cols"], n_iter=2,
    )
    assert _learned_only_from_train(search.best_estimator_, s["X_train"], s["X_val"])


# ---------------------------------------------------------------------------
# Cross-validation: every fold refits preprocessing on its own training rows
# ---------------------------------------------------------------------------
def test_cross_validation_never_fits_preprocessing_on_heldout_rows(shifted_splits):
    s = shifted_splits
    base = build_full_pipeline(LogisticRegression(max_iter=500), s["num_cols"], s["cat_cols"])
    pipeline = Pipeline([("spy", _FitIndexSpy())] + list(base.steps))

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(s["X_train"], s["y_train"]))
    result = cross_validate(
        pipeline, s["X_train"], s["y_train"], cv=folds, scoring="roc_auc", return_estimator=True
    )

    assert len(result["estimator"]) == 5
    for (train_idx, test_idx), fold_estimator in zip(folds, result["estimator"]):
        seen = fold_estimator.named_steps["spy"].fit_index_
        assert seen == frozenset(train_idx)
        assert seen.isdisjoint(test_idx)


# ---------------------------------------------------------------------------
# Resampling (SMOTE / oversampling) happens inside the pipeline, train rows only
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("resampler", ["smote", "random_oversample"])
def test_resampling_only_touches_rows_the_pipeline_is_fit_on(shifted_splits, resampler):
    pytest.importorskip("imblearn")
    s = shifted_splits
    n_train = len(s["X_train"])
    n_minority = int(s["y_train"].sum())
    n_majority = n_train - n_minority

    plain = build_imbalanced_pipeline(
        _SampleCountingClassifier(strategy="prior"), s["num_cols"], s["cat_cols"], resampler="none"
    ).fit(s["X_train"], s["y_train"])
    resampled = build_imbalanced_pipeline(
        _SampleCountingClassifier(strategy="prior"), s["num_cols"], s["cat_cols"], resampler=resampler
    ).fit(s["X_train"], s["y_train"])

    # No resampling: classifier sees exactly the training rows.
    assert plain.named_steps["classifier"].n_fit_rows_ == n_train
    # Resampling: classes balanced to the majority count, using training rows only.
    assert resampled.named_steps["classifier"].n_fit_rows_ == 2 * n_majority

    # At inference nothing is resampled: one prediction per input row.
    assert resampled.predict_proba(s["X_val"]).shape[0] == len(s["X_val"])
