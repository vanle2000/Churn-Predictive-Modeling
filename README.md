# Customer Churn Predictive Modeling

**Retention ROI Optimization System** — identifies at-risk customers, quantifies revenue impact, and produces per-customer SHAP explanations for CRM integration.

---

## Case Study

### Introduction
Customer acquisition costs 5–7× more than retention. Yet most companies only act after a customer is already gone. This project builds a production-grade churn prediction pipeline that ranks customers by churn risk and — critically — simulates the **net ROI** of intervening, so retention teams know not just *who* to contact but *whether it's worth it*.

### Problem
Standard churn models optimize for accuracy or AUC on imbalanced datasets. That's the wrong objective. A model that flags the top 10% of customers by risk and achieves 60% precision at that threshold recovers far more revenue than a 95%-accurate model that never identifies the right segment. The business questions are:
- Which customers are in the top decile of churn risk right now?
- What is the expected revenue impact of reaching out to them?
- Which product signals are driving their risk?

### Solution
End-to-end binary classification pipeline with business-first evaluation:

1. **Data:** IBM Telco Customer Churn dataset (7,043 customers, 21 features)
2. **Features:** Tenure buckets, charge volatility, service adoption breadth, billing behavior
3. **Models:** Logistic Regression → Random Forest → XGBoost (SMOTE-augmented)
4. **Evaluation:** ROC-AUC, Precision@10%, Precision@20%, calibration curves, net ROI simulation
5. **Explainability:** SHAP global summary + per-customer waterfall plots
6. **Business output:** ROI curve across targeting thresholds (intervention cost = $50/customer, revenue saved = $500/churner retained)

### Results
> Run `make data && make train` to reproduce.

| Model | ROC-AUC | Precision@10% | Net ROI (top 20%) |
|-------|---------|--------------|-------------------|
| Logistic Regression | ~0.84 | ~0.62 | ~positive |
| Random Forest | ~0.86 | ~0.65 | ~positive |
| XGBoost + SMOTE | ~0.87 | ~0.67 | ~highest |

*Exact results vary by random seed. Run training to get current numbers.*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| ML modeling | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Evaluation | Stratified 5-Fold CV, calibration curves, Precision@K |
| Visualization | Matplotlib, Seaborn |
| Testing | pytest, pytest-cov |
| Reproducibility | Makefile, joblib model serialization |

---

## Data Architecture

```
data/
├── raw/
│   └── telco_churn.csv          ← Downloaded by `make data`
└── processed/
    └── churn_processed.parquet  ← Output of preprocessing pipeline

src/
├── data/
│   ├── download.py              ← Fetches raw dataset
│   └── preprocessing.py        ← clean() → encode() → engineer_features()
├── models/
│   ├── train.py                 ← CV evaluation + final model training
│   └── evaluate.py             ← SHAP, ROI curves, calibration plots
└── visualization/
    └── eda.py                   ← Reusable EDA plots with statistical tests

models/
└── {model_name}.joblib          ← Persisted best model

reports/
├── model_comparison.csv         ← CV results table
└── figures/
    ├── shap_summary_*.png
    ├── precision_at_k_*.png
    ├── roi_curve_*.png
    └── calibration_*.png
```

---

## Key Insights & Analytics

1. **Month-to-month contract customers churn at ~42%** vs ~11% for two-year contracts — the single highest-signal feature, confirmed by SHAP
2. **Tenure under 12 months is the highest-risk window** — 55%+ churn rate in the 0–6 month band, suggesting onboarding friction as root cause
3. **Fiber optic customers churn more than DSL** despite higher spend — a product quality signal worth investigating with an A/B test on service reliability
4. **Electronic check payers churn at 2× the rate** of auto-pay customers — a payment friction signal that could be addressed with a UX intervention
5. **Precision@10% of ~67%** means targeting the top 10% of customers by model score catches 67% true churners — vs 26% for random outreach

---

## Experiment Design (A/B Test Framing)
The model scores customers for *likelihood to churn*. The next step is measuring whether the intervention *works*:

- **Control group:** High-risk customers who receive no outreach (holdout)
- **Treatment group:** High-risk customers who receive a retention offer
- **Primary metric:** 90-day retention rate
- **Secondary metric:** Average revenue retained per customer contacted
- **Guardrail metric:** Customer satisfaction score (don't over-contact)
- **Expected MDE:** 5pp lift in 90-day retention, powered at 80% with α=0.05 requires ~600 customers per arm

---

## How to Run

```bash
git clone https://github.com/vanle2000/Churn-Predictive-Modeling.git
cd Churn-Predictive-Modeling
pip install -r requirements.txt

make data    # download Telco dataset → data/raw/
make train   # run 5-fold CV + save best model → models/
make test    # run unit test suite
```

**Scale to production:**
- Replace CSV with a SQL/Snowflake connector in `src/data/download.py`
- Swap Pandas for PySpark in `preprocessing.py` for datasets >10M rows
- Deploy scoring via FastAPI + the serialized `.joblib` model
- Schedule weekly retraining with Airflow; monitor PSI for feature drift

---

## Challenges & What Could Be Improved

| Challenge | Improvement Path |
|-----------|-----------------|
| Single static dataset | Connect to live CRM via API; add incremental data loading |
| SMOTE creates synthetic samples globally | Use stratified SMOTE within each CV fold to prevent data leakage |
| No uplift modeling | Add a two-model uplift estimator: P(retain \| treated) − P(retain \| untreated) |
| Calibration not enforced | Wrap best model in `CalibratedClassifierCV` before deployment |
| No drift monitoring | Add PSI calculation on feature distributions week-over-week |
| Single business scenario | Parameterize ROI curve: different intervention costs per customer segment |
