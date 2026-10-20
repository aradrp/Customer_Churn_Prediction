"""
evaluate.py
===========
Metrics, threshold optimization, business-cost analysis, calibration, and
explainability utilities for the Telco Customer Churn project.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

logger = logging.getLogger(__name__)

DEFAULT_FN_COST = 500.0  # cost of missing a true churner (business default assumption)
DEFAULT_FP_COST = 30.0   # cost of unnecessarily contacting a non-churner


@dataclass
class ClassificationMetrics:
    """Standard metric bundle at a fixed decision threshold."""
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int
    prediction_latency_ms: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    measure_latency_model=None,
    X_for_latency=None,
) -> ClassificationMetrics:
    """
    Compute the full metric bundle required by the project spec at a given
    decision threshold: accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
    Brier score, and the confusion matrix.

    ROC-AUC and PR-AUC are threshold-independent (computed on probabilities)
    and reported alongside threshold-dependent metrics because:
      - ROC-AUC summarizes ranking quality across all thresholds but can look
        optimistic under class imbalance.
      - PR-AUC focuses on the positive (churn) class and is more sensitive to
        prevalence, which is informative in an imbalanced business problem.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    latency_ms = 0.0
    if measure_latency_model is not None and X_for_latency is not None:
        start = time.perf_counter()
        measure_latency_model.predict_proba(X_for_latency)
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(len(X_for_latency), 1)

    return ClassificationMetrics(
        threshold=threshold,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
        brier=brier_score_loss(y_true, y_proba),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        prediction_latency_ms=latency_ms,
    )


def expected_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> float:
    """
    expected_cost = false_negatives * fn_cost + false_positives * fp_cost

    This intentionally ignores true positives/negatives: correctly predicting
    a non-churner costs nothing extra, and correctly flagging a churner is
    assumed to cost the same regardless of the model (the retention offer
    cost is a business constant, not a modeling decision, and is netted out
    of this comparison since it is identical across models/thresholds).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * fn_cost + fp * fp_cost)


def threshold_search(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> pd.DataFrame:
    """
    Evaluate thresholds from 0.05 to 0.95 (default step 0.01) and compute
    precision, recall, F1, confusion-matrix counts, and expected business
    cost at each one.

    Returns
    -------
    pd.DataFrame sorted by expected_cost ascending, indexed by threshold.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * fn_cost + fp * fp_cost
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "expected_cost": cost,
            "cost_per_customer": cost / len(y_true),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("expected_cost").reset_index(drop=True)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> Tuple[float, pd.DataFrame]:
    """
    Return (optimal_threshold, full_search_table) minimizing expected
    business cost on the VALIDATION set. The threshold must be frozen before
    it is ever applied to the test set.
    """
    table = threshold_search(y_true, y_proba, fn_cost=fn_cost, fp_cost=fp_cost)
    best_threshold = float(table.iloc[0]["threshold"])
    logger.info("Optimal threshold = %.2f (expected cost = %.2f)", best_threshold, table.iloc[0]["expected_cost"])
    return best_threshold, table


def cost_sensitivity_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_costs: List[float] = (200, 500, 1000, 2000),
    fp_cost: float = DEFAULT_FP_COST,
) -> pd.DataFrame:
    """
    Repeat threshold selection across several false-negative cost
    assumptions and report the resulting optimal threshold and total cost
    for each, satisfying the "Business Cost Exercise" sensitivity challenge.
    """
    rows = []
    for fn_cost in fn_costs:
        best_t, table = find_optimal_threshold(y_true, y_proba, fn_cost=fn_cost, fp_cost=fp_cost)
        best_row = table.iloc[0]
        rows.append({
            "fn_cost_assumption": fn_cost,
            "fp_cost_assumption": fp_cost,
            "optimal_threshold": best_t,
            "expected_cost": best_row["expected_cost"],
            "precision": best_row["precision"],
            "recall": best_row["recall"],
        })
    return pd.DataFrame(rows)


def risk_group(probability: float) -> str:
    """Bucket a churn probability into a business-friendly risk group."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    return "Low"


def recommended_action(risk: str) -> str:
    """Map a risk group to a recommended retention action."""
    return {
        "High": "Priority outreach: personal call + retention offer within 48h",
        "Medium": "Targeted email campaign with loyalty discount",
        "Low": "No action; include in standard satisfaction survey",
    }.get(risk, "Review manually")


def permutation_importance_report(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """
    Compute model-agnostic permutation importance on the validation set.
    Works for any fitted pipeline (linear, tree, boosting).
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=-1,
    )
    report = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return report


def shap_explanation(model, X_sample: pd.DataFrame, max_display: int = 15):
    """
    Compute SHAP values for a fitted pipeline on a sample of rows.

    Gracefully degrades if `shap` is not installed: returns None and logs a
    warning instead of crashing the pipeline. SHAP is a nice-to-have on top
    of permutation importance, not something the rest of the system depends
    on, so a missing optional dependency shouldn't take down training or the
    notebooks.
    """
    try:
        import shap
    except ImportError:
        logger.warning(
            "shap is not installed; skipping SHAP explanation. "
            "Install with `pip install shap` to enable this analysis. "
            "Falling back to permutation importance is recommended."
        )
        return None

    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        logger.warning("Model does not expose 'preprocessor'/'classifier' steps; skipping SHAP.")
        return None

    domain_features = model.named_steps.get("domain_features")
    X_for_preprocessor = domain_features.transform(X_sample) if domain_features is not None else X_sample
    X_transformed = preprocessor.transform(X_for_preprocessor)
    
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.Explainer(classifier, X_transformed, feature_names=feature_names)
    shap_values = explainer(X_transformed)
    return shap_values


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap a metric to estimate a confidence interval.

    Point estimates alone invite over-reading small differences between
    models (e.g. "0.842 vs 0.838 ROC-AUC" can be pure resampling noise).
    Bootstrapping the validation set shows roughly how much the metric would
    move on a different sample of the same population.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_t, y_p = y_true[idx], y_proba[idx]
        if len(np.unique(y_t)) < 2:
            continue
        scores.append(metric_fn(y_t, y_p))

    scores = np.array(scores)
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    return (
        float(metric_fn(y_true, y_proba)),
        float(np.percentile(scores, lower_pct)),
        float(np.percentile(scores, upper_pct)),
    )


def subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_proba_col: str,
    group_col: str,
    min_group_size: int = 20,
) -> pd.DataFrame:
    """
    Compute precision, recall, false-positive rate, false-negative rate, and
    average predicted probability (a proxy for calibration) for each
    subgroup of `group_col`. Groups smaller than `min_group_size` are
    flagged as unreliable rather than silently reported.
    """
    rows = []
    for group_value, sub in df.groupby(group_col):
        n = len(sub)
        y_t = sub[y_true_col].values
        y_p = sub[y_pred_col].values
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        rows.append({
            "group": group_value,
            "n": n,
            "reliable_sample_size": n >= min_group_size,
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall": recall_score(y_t, y_p, zero_division=0),
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "mean_predicted_probability": sub[y_proba_col].mean(),
            "actual_churn_rate": y_t.mean(),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
