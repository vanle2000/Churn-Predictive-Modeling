# Customer Churn Predictive Modeling

---

## Case Study

### Introduction
Customer churn is one of the most costly and preventable problems in subscription-based businesses. Acquiring a new customer costs 5–7x more than retaining an existing one. Yet most companies only learn a customer has churned after they are already gone. This project is designed to reverse that dynamic: identify at-risk customers before they leave, so retention teams can act while there is still time.

### Problem
The core challenge is not just building a classifier — it is building one that is **useful for a business**. A model with 95% accuracy on a 95/5 class-imbalanced churn dataset tells you nothing useful. The real questions are:
- Which customers are in the top decile of churn risk right now?
- How confident is the model in that prediction?
- What features are driving the risk for each customer?
- What is the expected revenue impact of intervening?

A technically correct model that cannot answer these questions will not get deployed.

### Solution
An end-to-end binary classification pipeline designed for operational use:

1. **Data ingestion** from source systems: CRM records, billing history, product usage logs, support ticket interactions
2. **Feature engineering** focused on behavioral signals:
   - Tenure and engagement trends (usage frequency, session depth, feature adoption)
   - Billing behavior (late payments, plan downgrades, discount usage)
   - Support history (ticket volume, resolution time, satisfaction scores)
   - Recency/Frequency/Monetary (RFM) features
3. **Baseline models:** Logistic Regression, Random Forest, XGBoost / LightGBM
4. **Evaluation framework:** Cross-validation + holdout test set, optimized for Recall and Precision@K — not raw accuracy
5. **Interpretability layer:** SHAP values for per-customer explanation of risk drivers
6. **Deployment:** Batch scoring pipeline with monitoring for model drift

### Results
> **Status: In progress.** The modeling pipeline and project structure are defined. Dataset integration and model training are the immediate next steps. Results will be published here upon completion.

**Target metrics upon completion:**
- ROC-AUC > 0.80 on holdout test set
- Precision@10% > 60% (top 10% of predictions should be true churners)
- Recall > 70% on the churn-positive class

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| ML modeling | Scikit-learn, XGBoost, LightGBM |
| Interpretability | SHAP |
| Experiment tracking | MLflow (planned) |
| Visualization | Matplotlib, Seaborn |
| Validation | Stratified KFolds, calibration curves |

---

## Data Architecture

```
Source Systems
├── CRM (customer demographics, account age, plan tier)
├── Billing (payment history, MRR, downgrades)
├── Product Usage (logins, feature usage, session length)
└── Support (ticket count, resolution time, CSAT scores)
         │
         ▼
Feature Store (processed, labeled, point-in-time correct)
         │
         ▼
├── data/raw/          ← original source extracts
├── data/processed/    ← cleaned, feature-engineered datasets
└── data/external/     ← third-party enrichment (industry benchmarks)
         │
         ▼
Model Training → Evaluation → Batch Scoring → Dashboard
```

---

## Key Insights & Analytics

> To be populated after model training. Expected findings based on domain literature:

- **Engagement drop in the 30–60 day window** before churn is typically the strongest leading indicator — more predictive than any single demographic feature
- **Support ticket spikes with negative sentiment** signal product frustration, not just usage problems
- **Plan downgrade events** are lagging indicators — customers who downgrade have often already decided to leave
- **High-value, long-tenure customers** who churn are the most costly — model should weight these in the loss function

---

## How to Reuse / Scale

**Reuse on your own dataset:**
1. Clone the repo
2. Replace `data/raw/` with your churn-labeled dataset (minimum columns: customer_id, churn_label, and behavioral features)
3. Update feature definitions in `src/features/`
4. Run `notebooks/` for EDA, then `src/models/` for training

**Scaling to large datasets (millions of customers):**
- Replace Pandas with PySpark or Dask for feature engineering on distributed data
- Use Databricks or AWS SageMaker for model training at scale
- Implement real-time scoring via a feature store (Feast, Tecton) + REST API rather than batch scoring

**Generalizes to:**
- SaaS subscription churn
- Telecom churn
- Retail customer attrition
- Financial product abandonment

---

## Challenges & What Could Be Improved

| Challenge | Improvement Path |
|-----------|-----------------|
| No dataset integrated yet | Connect to a public churn dataset (Telco Customer Churn on Kaggle) as starting point |
| Class imbalance (churn is typically 5–20% of customers) | SMOTE oversampling, cost-sensitive learning, or threshold tuning |
| Model drift over time | Implement PSI (Population Stability Index) monitoring on feature distributions |
| Interpretability gap | Add SHAP waterfall plots per customer for CRM integration |
| No backtesting | Simulate historical intervention ROI: model's top-K predictions vs. actual retention outcomes |
| Single model approach | Build a model ensemble: short-term risk (30-day) + long-term risk (90-day) for different intervention strategies |
