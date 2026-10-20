# Telco Customer Churn — Leakage-Safe Prediction & Retention Decisions

Predicts which telecom customers are likely to cancel, explains why, and
recommends an action based on **estimated business cost, not accuracy**.

Trained and evaluated on the real
[Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
**7,043 customers, 21 columns, 26.5% churn rate.** Every number below is
read from the files in `reports/`, produced by `python -m src.train` on that data.

---

## 1. What this project does

1. **Audits and cleans** the raw data (schema check, `TotalCharges` coercion,
   duplicate / missing-value checks, `customerID` kept out of the features).
2. **Compares five models** (Dummy, Logistic Regression, Decision Tree, Random
   Forest, Gradient Boosting) on a held-out validation set, tunes the two best
   with `RandomizedSearchCV` over a 5-fold `StratifiedKFold`, and compares four
   imbalance strategies (none, `class_weight`, SMOTE, random oversampling).
3. **Chooses the decision threshold by business cost**
   (`false negative = $500`, `false positive = $30`) on the validation set, then
   evaluates once on the test set.
4. **Explains the model** with logistic-regression coefficients, permutation
   importance and SHAP, plus a look at true-positive / false-positive /
   false-negative examples.
5. **Serves predictions** through a FastAPI app and writes `predictions.csv`
   with probability, risk group and a recommended action per customer.

---

## 2. Results (real data)

### Model comparison — validation set, threshold 0.5

| Model | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|
| Gradient Boosting | 0.8369 | 0.6434 | 0.6827 | 0.5053 |
| Logistic Regression | 0.8360 | 0.6409 | 0.6698 | 0.5125 |
| Random Forest | 0.8310 | 0.6354 | 0.6602 | 0.4840 |
| Decision Tree | 0.8136 | 0.5774 | 0.5913 | 0.4377 |
| Dummy | 0.5020 | 0.2666 | 0.2688 | 0.2669 |

Gradient Boosting and Logistic Regression are effectively tied
(0.8369 vs 0.8360). After tuning, 5-fold CV ROC-AUC was
0.8477 ± 0.0037 for Gradient Boosting and 0.8489 ± 0.0050 for Logistic
Regression (`C=0.1`) — again within noise. **Logistic Regression was selected**
because it scored marginally higher in CV and is simpler and directly
interpretable.

### Final model on the held-out test set (threshold = 0.05)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8544 |
| PR-AUC | 0.6689 |
| Recall | 0.9929 |
| Precision | 0.3592 |
| F1 | 0.5275 |
| Accuracy | 0.5289 |
| Confusion matrix (TN / FP / FN / TP) | 281 / 496 / 2 / 278 |
| Expected cost on test (2×$500 + 496×$30) | $15,880 |

**How to read this.** ROC-AUC (~0.85) is the honest measure of how well the
model ranks customers. Accuracy is low and recall is very high *by design*: with
a missed churner assumed to cost about 16.7× an unnecessary contact, the
cost-optimal policy flags 73% of customers and accepts many false alarms.

### Threshold selection

Thresholds from 0.05 to 0.95 (step 0.01) were scored on the **validation**
set with `cost = FN × $500 + FP × $30`; the minimum
($19,770 on validation) is at **0.05**, then frozen before touching the test set.

Sensitivity to the cost assumption (validation set):

| Assumed FN cost | Optimal threshold | Expected cost | Precision | Recall |
|---|---|---|---|---|
| $200 | 0.11 | 15,870 | 0.411 | 0.915 |
| $500 | 0.05 | 19,770 | 0.348 | 0.968 |
| $1000 | 0.05 | 24,270 | 0.348 | 0.968 |
| $2000 | 0.05 | 33,270 | 0.348 | 0.968 |

Two caveats: the optimum sits at the **lower edge of the search grid**
(0.05), so the true optimum may be lower; and the $500 / $30 figures are
illustrative assumptions, not this business's real costs.

### Imbalance strategies (run on Gradient Boosting, the pre-tuning leader; validation, threshold 0.5)

| Strategy | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|
| none | 0.8369 | 0.6434 | 0.6827 | 0.5053 |
| class_weight_balanced | 0.8369 | 0.6434 | 0.6827 | 0.5053 |
| smote | 0.8357 | 0.6318 | 0.6054 | 0.6441 |
| random_oversample | 0.8374 | 0.6342 | 0.5303 | 0.7794 |

`class_weight_balanced` is identical to `none` because
`GradientBoostingClassifier` has no `class_weight` parameter — that row is
a no-op, not evidence that class weights don't help. SMOTE and random
oversampling raise recall at the cost of precision, as expected; neither
improves ROC-AUC meaningfully.

### What drives churn

Permutation importance (validation set, ROC-AUC drop, 10 repeats):

| Feature | Importance |
|---|---|
| tenure | 0.1755 ± 0.0111 |
| Contract | 0.0366 ± 0.0046 |
| TotalCharges | 0.0257 ± 0.0051 |
| InternetService | 0.0151 ± 0.0037 |
| PaymentMethod | 0.0046 ± 0.0015 |

Largest logistic-regression coefficients (standardized numeric features,
one-hot categoricals):

| Feature | Coefficient |
|---|---|
| Contract_Month-to-month | +0.729 |
| Contract_Two year | -0.694 |
| tenure | -0.487 |
| InternetService_Fiber optic | +0.387 |
| InternetService_DSL | -0.337 |
| is_new_customer | +0.269 |

Both views agree: short tenure and month-to-month contracts raise churn
risk; two-year contracts and longer tenure lower it; fiber-optic internet
and electronic-check payment (+0.25) raise it. These are **associations in
observational data, not proven causes.**

### Data quality

11 customers have a blank `TotalCharges`; all of them have `tenure = 0`
(new accounts that have not been billed yet). They are imputed inside the
pipeline, not dropped. No duplicate rows or duplicate `customerID` values.

---

## 3. How leakage is prevented

- **Split first.** A stratified 70 / 15 / 15 train / validation / test split
  (seed 42) happens before anything is learned from the data. The only
  steps before the split are stateless (type coercion, duplicate removal).
- **Everything that learns lives in the pipeline.** Imputation medians,
  scaler mean/std and one-hot vocabularies are `Pipeline` /
  `ColumnTransformer` steps; SMOTE and random oversampling sit inside an
  `imblearn` pipeline, so they only ever touch the rows that pipeline is
  being fit on.
- **Three data roles.** Hyperparameters are tuned with CV on the training
  set only. The validation set ranks models and picks the threshold. The
  test set is used once, after the threshold is frozen.
- **`customerID` is excluded** from the features.

### Tests (`tests/test_leakage.py`)

The leakage tests are designed so that they can fail:

- The validation split is deliberately made very different from training
  (shifted `TotalCharges` and `tenure`, missing values, a payment method that
  exists only in validation). Fitted medians, means and category lists must
  equal statistics computed independently with pandas on the **training rows
  only**.
- A **negative control** fits on train + validation pooled and asserts the
  same check flags it.
- The real code paths `compare_models` and `tune_finalist` from `src/train.py`
  are tested, not just hand-built pipelines.
- A spy step inside cross-validation records which rows each fold's
  preprocessing was fit on: always exactly the fold's training rows, never the
  held-out ones.
- Predictions for a row must be identical alone or inside a batch (catches
  statistics learned at transform time).
- Resampling is applied only to the training rows the pipeline is fit on, and
  not at inference.

As a check on the tests themselves, deliberately injecting leaks (fitting on
train + validation inside `compare_models`; an imputer or scaler that learns
from the batch being scored) made them fail.

---

## 4. Evaluation protocol — and what it is *not*

Model selection uses `RandomizedSearchCV` with `StratifiedKFold` (5 folds) on
the training set. **This is not nested cross-validation**: the CV score used to
pick the final model comes from the same folds that tuned its
hyperparameters, so it is slightly optimistic. The honest estimate is the
single test-set evaluation above. Test-set size is 1,057
customers, so metrics carry sampling noise (see the bootstrap CI in notebook 04).

---

## 5. Repository structure

```
telco-churn-project/
├── data/raw/                 # put telco_churn.csv here
├── data/processed/           # frozen train / val / test splits
├── notebooks/                # 01 audit · 02 EDA · 03 modeling · 04 decision analysis
├── src/                      # data.py · features.py · evaluate.py · predict.py · train.py
├── tests/                    # test_data.py · test_pipeline.py · test_leakage.py
├── models/                   # fitted pipeline + metadata (joblib)
├── reports/                  # metrics tables and figures
├── api.py                    # FastAPI: /health, /predict, /predict/batch
├── predictions.csv           # per-customer risk report for the test set
├── requirements.txt
└── README.md
```

Notebook 04 loads the model saved by `src.train` rather than refitting its
own, so its numbers match `models/model_metadata.json`.

---

## 6. Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and save it as
`data/raw/telco_churn.csv`.

```bash
python -m src.train --data data/raw/telco_churn.csv   # train, tune, pick threshold, write reports
pytest tests/ -v                                      # run the tests
uvicorn api:app --reload --port 8000                  # serve predictions
jupyter notebook notebooks/01_data_audit.ipynb        # run notebooks 01 → 04 in order
```

(`python -m src.train --synthetic` generates a schema-compatible fake dataset,
only for smoke-testing without the Kaggle file; it is not used for any result here.)

Example request:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 427.5
}'
```

**Reproducibility.** Seed 42 is used for the split, models, CV and sampling.
`requirements.txt` gives version ranges, not exact pins. The reports in this
repo were produced with Python 3.12, pandas 2.3.3, scikit-learn 1.9.0,
numpy 2.4.4, imbalanced-learn 0.14.2 and shap 0.54.1; re-running
`python -m src.train` reproduced the earlier threshold search, cost
sensitivity and permutation importance exactly. A model pickled with one
scikit-learn version may not load cleanly under another, so retrain rather
than copying `models/*.joblib` across environments.

---

## 7. Limitations

- **Cost assumptions are illustrative** ($500 / $30); replace
  `DEFAULT_FN_COST` / `DEFAULT_FP_COST` in `src/evaluate.py` with real figures.
- **The cost-optimal threshold is aggressive** (recall 0.99, precision
  0.36): most contacts go to customers who would have stayed. If
  outreach capacity is limited, contacting the top-N highest-risk customers,
  or using a higher threshold from the sensitivity table, is more realistic.
- **Associations are not causal.** Acting on "month-to-month churns more"
  would need a controlled experiment.
- **Random split, not time-based.** Stability under changing offers or
  customer behaviour is untested; a deployment needs monitoring and retraining.
- **CV scores are slightly optimistic** (no nested CV); the test set is the
  reference.
- **Fairness is not audited.** `subgroup_metrics` is demonstrated for contract
  type only; a full audit by gender, senior-citizen status, etc. is still needed.
