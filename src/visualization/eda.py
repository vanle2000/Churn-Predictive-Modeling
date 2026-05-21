"""Reusable EDA plotting functions for churn analysis."""

import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

FIGURES_DIR = pathlib.Path(__file__).parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def churn_rate_by_segment(df: pd.DataFrame, col: str, target: str = "Churn") -> None:
    """Bar chart of churn rate per category with sample sizes and confidence intervals."""
    groups = df.groupby(col)[target]
    rates = groups.mean()
    counts = groups.count()

    # 95% Wilson confidence interval
    z = 1.96
    lower, upper = [], []
    for p, n in zip(rates, counts):
        se = np.sqrt(p * (1 - p) / n)
        lower.append(max(0, p - z * se))
        upper.append(min(1, p + z * se))

    fig, ax = plt.subplots(figsize=(max(6, len(rates) * 0.8), 4))
    bars = ax.bar(rates.index, rates.values, color="#4878cf", alpha=0.85)
    ax.errorbar(
        range(len(rates)), rates.values,
        yerr=[rates.values - lower, np.array(upper) - rates.values],
        fmt="none", color="black", capsize=4,
    )
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={n:,}", ha="center", va="bottom", fontsize=8,
        )
    ax.axhline(df[target].mean(), color="red", ls="--", lw=1, label="Overall churn rate")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel(col)
    ax.set_title(f"Churn Rate by {col}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"churn_by_{col}.png", dpi=150)
    plt.close()


def monthly_charge_distribution(df: pd.DataFrame, target: str = "Churn") -> None:
    """KDE of MonthlyCharges split by churn — shows where churners concentrate."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for churn_val, label, color in [(0, "Retained", "#1f77b4"), (1, "Churned", "#d62728")]:
        subset = df[df[target] == churn_val]["MonthlyCharges"]
        subset.plot.kde(ax=ax, label=f"{label} (n={len(subset):,})", color=color, lw=2)
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_title("Monthly Charge Distribution: Churned vs. Retained")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_charges_by_churn.png", dpi=150)
    plt.close()


def tenure_churn_heatmap(df: pd.DataFrame, target: str = "Churn") -> None:
    """Churn rate by tenure × contract type — highlights high-risk segments."""
    if "tenure_band" not in df.columns:
        df = df.copy()
        df["tenure_band"] = pd.cut(
            df["tenure"], bins=[0, 6, 12, 24, 48, float("inf")],
            labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
        )
    pivot = df.pivot_table(values=target, index="tenure_band", columns="Contract", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn_r", ax=ax, vmin=0, vmax=1)
    ax.set_title("Churn Rate by Tenure Band × Contract Type")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tenure_contract_heatmap.png", dpi=150)
    plt.close()


def correlation_heatmap(df: pd.DataFrame, target: str = "Churn") -> None:
    """Correlation of numeric features with churn target."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corrwith(df[target]).drop(target, errors="ignore").sort_values()
    fig, ax = plt.subplots(figsize=(6, max(4, len(corr) * 0.25)))
    corr.plot.barh(ax=ax, color=["#d62728" if v > 0 else "#1f77b4" for v in corr])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Feature Correlation with {target}")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlations.png", dpi=150)
    plt.close()


def statistical_segment_test(
    df: pd.DataFrame, feature: str, target: str = "Churn"
) -> dict:
    """Two-sample t-test: is the mean of `feature` different for churned vs retained?"""
    churned = df[df[target] == 1][feature].dropna()
    retained = df[df[target] == 0][feature].dropna()
    t_stat, p_value = stats.ttest_ind(churned, retained)
    effect_size = (churned.mean() - retained.mean()) / df[feature].std()
    return {
        "feature": feature,
        "churned_mean": churned.mean(),
        "retained_mean": retained.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": effect_size,
        "significant": p_value < 0.05,
    }
