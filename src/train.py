"""
train.py
========
End-to-end training entry point. Running this script reproduces the entire
core modeling pipeline:

    1. Load + clean data, stratified split (train/val/test)
    2. Build leakage-safe preprocessing pipeline
    3. Train + compare Dummy, LogisticRegression, DecisionTree,
       RandomForest, GradientBoosting
    4. Cross-validate and lightly tune the strongest finalists
    5. Compare imbalance strategies (class_weight; SMOTE/RandomOverSampler
       if imbalanced-learn is installed)
    6. Select the best model by validation ROC-AUC, then optimize the
       decision threshold on the validation set by expected business cost
    7. Evaluate ONCE on the held-out test set at the frozen threshold
    8. Save the pipeline + metadata, generate predictions.csv,
       feature-importance / permutation-importance report

Usage
-----
    python -m src.train --data data/raw/telco_churn.csv
    python -m src.train --synthetic   # no Kaggle download needed (demo mode)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.data import (
    RANDOM_SEED, TARGET_COLUMN, ID_COLUMN,
    audit_data, clean_data, get_feature_columns, load_raw_data,
    make_synthetic_telco_data, split_data,
)
from src.evaluate import (
    DEFAULT_FN_COST, DEFAULT_FP_COST,
    cost_sensitivity_analysis, evaluate_at_threshold, find_optimal_threshold,
    permutation_importance_report,
)
from src.features import build_full_pipeline
from src.predict import ModelArtifact, generate_final_risk_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

np.random.seed(RANDOM_SEED)


def get_candidate_models(random_state: int = RANDOM_SEED) -> dict:
    """Model zoo: dummy baseline + 4 required model families."""
    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=random_state),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def compare_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    numerical_columns: list,
    categorical_columns: list,
) -> pd.DataFrame:
    """Fit each candidate on train, evaluate on validation. Returns a comparison table."""
    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = build_full_pipeline(estimator, numerical_columns, categorical_columns)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = evaluate_at_threshold(y_val, val_proba, threshold=0.5)

        rows.append({
            "model": name,
            "val_roc_auc": metrics.roc_auc,
            "val_pr_auc": metrics.pr_auc,
            "val_f1_at_0.5": metrics.f1,
            "val_precision_at_0.5": metrics.precision,
            "val_recall_at_0.5": metrics.recall,
            "fit_seconds": round(fit_seconds, 2),
        })
        fitted[name] = pipeline
        logger.info("%-20s | val ROC-AUC=%.4f | val PR-AUC=%.4f", name, metrics.roc_auc, metrics.pr_auc)

    comparison = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
    return comparison, fitted


def tune_finalist(
    estimator_name: str,
    estimator,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numerical_columns: list,
    categorical_columns: list,
    n_iter: int = 15,
    random_state: int = RANDOM_SEED,
) -> RandomizedSearchCV:
    """RandomizedSearchCV over a finalist pipeline using StratifiedKFold."""
    pipeline = build_full_pipeline(estimator, numerical_columns, categorical_columns)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    prefixed_params = {f"classifier__{k}": v for k, v in param_distributions.items()}
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=prefixed_params,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info(
        "%s tuned | best CV ROC-AUC=%.4f +/- %.4f | best params=%s",
        estimator_name, search.best_score_,
        search.cv_results_["std_test_score"][search.best_index_], search.best_params_,
    )
    return search


def compare_imbalance_strategies(
    base_estimator_factory,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    numerical_columns: list,
    categorical_columns: list,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compare: no treatment, class_weight='balanced', SMOTE, RandomOverSampler
    (the latter two skipped gracefully if imbalanced-learn is unavailable).
    """
    from src.features import build_imbalanced_pipeline

    strategies = {
        "none": (base_estimator_factory(class_weight=None), "none"),
        "class_weight_balanced": (base_estimator_factory(class_weight="balanced"), "class_weight"),
    }

    try:
        import imblearn  # noqa: F401
        strategies["smote"] = (base_estimator_factory(class_weight=None), "smote")
        strategies["random_oversample"] = (base_estimator_factory(class_weight=None), "random_oversample")
        imblearn_available = True
    except ImportError:
        imblearn_available = False
        logger.warning("imbalanced-learn not installed; skipping SMOTE / RandomOverSampler comparison. "
                        "Install with `pip install imbalanced-learn` to enable.")

    rows = []
    for strategy_name, (estimator, resampler_kind) in strategies.items():
        pipeline = build_imbalanced_pipeline(
            estimator, numerical_columns, categorical_columns,
            resampler=resampler_kind, random_state=random_state,
        )
        pipeline.fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = evaluate_at_threshold(y_val, val_proba, threshold=0.5)
        rows.append({
            "strategy": strategy_name,
            "val_roc_auc": metrics.roc_auc,
            "val_pr_auc": metrics.pr_auc,
            "val_recall_at_0.5": metrics.recall,
            "val_precision_at_0.5": metrics.precision,
        })

    if not imblearn_available:
        rows.append({
            "strategy": "smote", "val_roc_auc": np.nan, "val_pr_auc": np.nan,
            "val_recall_at_0.5": np.nan, "val_precision_at_0.5": np.nan,
        })
        rows.append({
            "strategy": "random_oversample", "val_roc_auc": np.nan, "val_pr_auc": np.nan,
            "val_recall_at_0.5": np.nan, "val_precision_at_0.5": np.nan,
        })

    return pd.DataFrame(rows)


def run_training(data_path: str, use_synthetic: bool, output_dir: str = ".") -> None:
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    processed_dir = output_dir / "data" / "processed"
    for d in (models_dir, reports_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load & clean -------------------------------------------------
    if use_synthetic:
        logger.info("Generating synthetic demo dataset (no Kaggle download required)...")
        raw_df = make_synthetic_telco_data(n_rows=5000, random_state=RANDOM_SEED)
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(data_path, index=False)
    raw_df = load_raw_data(data_path)

    quality_report = audit_data(raw_df)
    quality_report.to_frame().to_csv(reports_dir / "data_quality_report.csv", index=False)
    logger.info("Target prevalence (raw): %.2f%%", 100 * quality_report.target_prevalence)

    df = clean_data(raw_df)

    # ---- 2. Split ----------------------------------------------------------
    train_df, val_df, test_df = split_data(df, random_state=RANDOM_SEED)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    numerical_columns, categorical_columns = get_feature_columns(train_df)
    logger.info("Numerical columns: %s", numerical_columns)
    logger.info("Categorical columns: %s", categorical_columns)

    feature_cols = numerical_columns + categorical_columns
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN].values
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN].values
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN].values

    # ---- 3. Compare baseline models -----------------------------------
    models = get_candidate_models()
    comparison_table, fitted_models = compare_models(
        models, X_train, y_train, X_val, y_val, numerical_columns, categorical_columns
    )
    comparison_table.to_csv(reports_dir / "model_comparison.csv", index=False)
    logger.info("\n%s", comparison_table.to_string(index=False))

    # ---- 4. Tune the top two finalists (excluding dummy) ------------------
    finalists = [m for m in comparison_table["model"] if m != "dummy"][:2]
    logger.info("Finalists selected for tuning: %s", finalists)

    param_grids = {
        "random_forest": {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [1, 2, 4, 8],
        },
        "gradient_boosting": {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        },
        "logistic_regression": {
            "C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "decision_tree": {
            "max_depth": [3, 4, 5, 6, 8, 10],
            "min_samples_leaf": [1, 2, 4, 8, 16],
        },
    }

    tuned_results = {}
    for name in finalists:
        base_estimator = get_candidate_models()[name]
        search = tune_finalist(
            name, base_estimator, param_grids.get(name, {}),
            X_train, y_train, numerical_columns, categorical_columns,
        )
        tuned_results[name] = search

    # ---- 5. Imbalance strategy comparison for the single best finalist ----
    best_finalist_name = finalists[0]

    def estimator_factory(class_weight=None, _name=best_finalist_name):
        base = get_candidate_models()[_name]
        if "class_weight" in base.get_params():
            base.set_params(class_weight=class_weight)
        elif class_weight is not None:
            logger.warning(
            "%s has no class_weight parameter — the 'class_weight_balanced' "
            "row will be identical to 'none' for this model.", _name
        )
        return base

    imbalance_table = compare_imbalance_strategies(
        estimator_factory, X_train, y_train, X_val, y_val, numerical_columns, categorical_columns
    )
    imbalance_table.to_csv(reports_dir / "imbalance_strategy_comparison.csv", index=False)
    logger.info("\n%s", imbalance_table.to_string(index=False))

    # ---- 6. Select final model = best tuned finalist by validation ROC-AUC
    best_name, best_search = max(tuned_results.items(), key=lambda kv: kv[1].best_score_)
    final_pipeline = best_search.best_estimator_
    logger.info("Selected final model: %s (CV ROC-AUC=%.4f)", best_name, best_search.best_score_)

    # ---- 7. Threshold optimization on validation set ----------------------
    val_proba = final_pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold, threshold_table = find_optimal_threshold(
        y_val, val_proba, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST
    )
    threshold_table.to_csv(reports_dir / "threshold_search.csv", index=False)

    sensitivity_table = cost_sensitivity_analysis(y_val, val_proba)
    sensitivity_table.to_csv(reports_dir / "cost_sensitivity_analysis.csv", index=False)

    # ---- 8. Final, single evaluation on the held-out TEST set -------------
    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(
        y_test, test_proba, threshold=optimal_threshold,
        measure_latency_model=final_pipeline, X_for_latency=X_test.iloc[:200],
    )
    logger.info("FINAL TEST METRICS @ threshold=%.2f: %s", optimal_threshold, test_metrics.as_dict())
    with open(reports_dir / "final_test_metrics.json", "w") as f:
        json.dump(test_metrics.as_dict(), f, indent=2)

    # ---- 9. Interpretability: permutation importance -----------------------
    perm_importance = permutation_importance_report(final_pipeline, X_val, y_val, n_repeats=10)
    perm_importance.to_csv(reports_dir / "permutation_importance.csv", index=False)

    # Native feature importance / coefficients where available
    classifier = final_pipeline.named_steps["classifier"]
    try:
        preproc = final_pipeline.named_steps["preprocessor"]
        feature_names = preproc.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            native_importance = pd.DataFrame({
                "feature": feature_names, "importance": classifier.feature_importances_,
            }).sort_values("importance", ascending=False)
            native_importance.to_csv(reports_dir / "native_feature_importance.csv", index=False)
        elif hasattr(classifier, "coef_"):
            native_importance = pd.DataFrame({
                "feature": feature_names, "coefficient": classifier.coef_.ravel(),
            }).sort_values("coefficient", key=abs, ascending=False)
            native_importance.to_csv(reports_dir / "native_feature_importance.csv", index=False)
    except Exception as exc:
        logger.warning("Could not compute native feature importance: %s", exc)

    # ---- 10. Save pipeline + metadata --------------------------------------
    model_path = models_dir / "churn_pipeline.joblib"
    metadata_path = models_dir / "model_metadata.joblib"
    joblib.dump(final_pipeline, model_path)

    metadata = {
        "model_version": f"{best_name}-v1-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": optimal_threshold,
        "fn_cost_assumption": DEFAULT_FN_COST,
        "fp_cost_assumption": DEFAULT_FP_COST,
        "metrics_at_threshold": test_metrics.as_dict(),
        "random_seed": RANDOM_SEED,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "model_family": best_name,
    }
    joblib.dump(metadata, metadata_path)
    with open(reports_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # ---- 11. Final risk report / predictions.csv ---------------------------
    artifact = ModelArtifact(
        pipeline=final_pipeline, threshold=optimal_threshold,
        model_version=metadata["model_version"], trained_at=metadata["trained_at"],
        metrics_at_threshold=test_metrics.as_dict(),
    )
    generate_final_risk_report(
        artifact, X_test, y_test, test_df[ID_COLUMN],
        output_path=output_dir / "predictions.csv",
    )

    logger.info("Training complete. Artifacts saved to %s and %s", models_dir, reports_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Telco Churn model end-to-end")
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic demo dataset first")
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    run_training(args.data, use_synthetic=args.synthetic, output_dir=args.output_dir)
