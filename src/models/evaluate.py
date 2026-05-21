"""Model evaluation: SHAP explanations, calibration, and business impact curves."""

import pathlib
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path(__file__).parents[2] / "models"
REPORTS_DIR = pathlib.Path(__file__).parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def plot_shap_summary(model, X: pd.DataFrame, model_name: str) -> None:
    """Global feature importance via SHAP — interpretable for stakeholders."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Extract classifier from pipeline if needed
    clf = model.named_steps.get("clf", model)

    explainer = shap.TreeExplainer(clf) if hasattr(clf, "feature_importances_") \
        else shap.LinearExplainer(clf, X)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance — {model_name}", pad=14)
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_summary_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP plot to %s", path)


def plot_precision_at_k_curve(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str
) -> None:
    """Precision@K curve — shows business value at each targeting threshold."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ks = np.linspace(0.01, 1.0, 100)
    precisions = []
    for k in ks:
        n = max(1, int(len(y_true) * k))
        top_idx = np.argsort(y_prob)[-n:]
        precisions.append(y_true[top_idx].mean())

    baseline = y_true.mean()

    plt.figure(figsize=(8, 5))
    plt.plot(ks * 100, precisions, label="Model", color="#1f77b4", lw=2)
    plt.axhline(baseline, color="gray", ls="--", label=f"Random baseline ({baseline:.1%})")
    plt.xlabel("Top K% of customers targeted (%)")
    plt.ylabel("Precision (fraction who are true churners)")
    plt.title(f"Precision@K Curve — {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"precision_at_k_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved Precision@K plot to %s", path)


def plot_roi_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    intervention_cost: float = 50.0,
    revenue_saved: float = 500.0,
) -> None:
    """Net ROI curve across targeting thresholds — directly answers 'what is this worth?'"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ks = np.linspace(0.01, 1.0, 100)
    rois, baseline_rois = [], []

    for k in ks:
        n = max(1, int(len(y_true) * k))
        top_idx = np.argsort(y_prob)[-n:]
        caught = int(y_true[top_idx].sum())
        cost = n * intervention_cost
        rev = caught * revenue_saved
        rois.append((rev - cost) / cost if cost > 0 else 0)

        # Baseline: random targeting
        baseline_caught = int(y_true.mean() * n)
        baseline_rev = baseline_caught * revenue_saved
        baseline_rois.append((baseline_rev - cost) / cost if cost > 0 else 0)

    plt.figure(figsize=(8, 5))
    plt.plot(ks * 100, rois, label="Model ROI", color="#2ca02c", lw=2)
    plt.plot(ks * 100, baseline_rois, color="gray", ls="--", label="Random baseline ROI")
    plt.axhline(0, color="red", ls=":", lw=1, label="Break-even")
    plt.xlabel("Top K% of customers targeted (%)")
    plt.ylabel("Net ROI (revenue saved / intervention cost − 1)")
    plt.title(f"Retention ROI Curve — {model_name}\n"
              f"Intervention cost=${intervention_cost}/customer, "
              f"Revenue saved=${revenue_saved}/churner retained")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"roi_curve_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved ROI curve to %s", path)


def plot_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str
) -> None:
    """Calibration curve — are predicted probabilities reliable?"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, fraction_pos, "s-", label="Model", color="#1f77b4")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve — {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"calibration_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved calibration plot to %s", path)
