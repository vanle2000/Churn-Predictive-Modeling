"""Train and evaluate churn models with proper stratified CV and business metrics."""

import pathlib
import logging
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score, brier_score_loss,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)

PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data" / "processed"
MODELS_DIR = pathlib.Path(__file__).parents[2] / "models"
REPORTS_DIR = pathlib.Path(__file__).parents[2] / "reports"

TARGET = "Churn"
DROP_COLS = ["tenure_band"]  # non-numeric engineered cols used only for EDA


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / "churn_processed.parquet")
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=[TARGET] + drop)
    y = df[TARGET]
    # Keep only numeric columns (get_dummies may leave object cols)
    X = X.select_dtypes(include=[np.number])
    return X, y


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Precision among the top-k% predicted churners — the metric that drives ROI."""
    n = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_prob)[-n:]
    return y_true[top_idx].mean()


def business_roi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    intervention_cost: float = 50.0,
    revenue_saved: float = 500.0,
    k: float = 0.20,
) -> dict:
    """
    Simulate ROI of targeting top-k% customers with a retention offer.

    intervention_cost: cost per customer contacted (dollars)
    revenue_saved: average revenue recovered per churner retained (dollars)
    k: fraction of customers targeted
    """
    n_total = len(y_true)
    n_targeted = int(n_total * k)
    top_idx = np.argsort(y_prob)[-n_targeted:]

    true_churners_caught = int(y_true[top_idx].sum())
    cost = n_targeted * intervention_cost
    revenue = true_churners_caught * revenue_saved
    roi = (revenue - cost) / cost if cost > 0 else 0.0

    # Baseline: random targeting
    baseline_caught = int(y_true.mean() * n_targeted)
    baseline_revenue = baseline_caught * revenue_saved
    baseline_roi = (baseline_revenue - cost) / cost if cost > 0 else 0.0

    return {
        "n_targeted": n_targeted,
        "true_churners_caught": true_churners_caught,
        "precision_at_k": true_churners_caught / n_targeted,
        "gross_revenue_saved": revenue,
        "intervention_cost": cost,
        "net_roi": roi,
        "vs_random_roi_lift": roi - baseline_roi,
    }


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )),
        ]),
        "xgboost_smote": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=3,  # handles imbalance
                eval_metric="logloss", use_label_encoder=False,
                random_state=42, n_jobs=-1,
            )),
        ]),
    }


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roi = business_roi(y_val.values, y_prob)

        results.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_val, y_prob),
            "avg_precision": average_precision_score(y_val, y_prob),
            "brier_score": brier_score_loss(y_val, y_prob),
            "recall": recall_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred),
            "precision_at_10pct": precision_at_k(y_val.values, y_prob, k=0.10),
            "precision_at_20pct": precision_at_k(y_val.values, y_prob, k=0.20),
            "net_roi": roi["net_roi"],
            "vs_random_roi_lift": roi["vs_random_roi_lift"],
        })

    summary = pd.DataFrame(results).mean().to_dict()
    summary["model"] = name
    logger.info(
        "[%s] ROC-AUC=%.3f | P@10%%=%.3f | ROI=%.2fx",
        name, summary["roc_auc"], summary["precision_at_10pct"], summary["net_roi"],
    )
    return summary


def train_final_model(model, X: pd.DataFrame, y: pd.Series, name: str) -> None:
    """Re-train winning model on full dataset and persist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.fit(X, y)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    X, y = load_features()
    logger.info("Dataset: %d rows, churn rate=%.1f%%", len(y), y.mean() * 100)

    models = build_models()
    all_results = []

    for name, model in models.items():
        logger.info("Evaluating %s ...", name)
        result = evaluate_model(name, model, X, y)
        all_results.append(result)

    results_df = pd.DataFrame(all_results).set_index("model").round(4)
    print("\n=== Model Comparison ===")
    print(results_df[["roc_auc", "avg_precision", "recall", "precision_at_10pct", "net_roi"]])

    best_name = results_df["roc_auc"].idxmax()
    logger.info("Best model: %s — re-training on full dataset", best_name)
    train_final_model(models[best_name], X, y, best_name)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv")
    return results_df


if __name__ == "__main__":
    run()
