"""Data loading, cleaning and validation for churn pipeline."""

import pathlib
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

RAW_DIR = pathlib.Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data" / "processed"

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


def load_raw(path: pathlib.Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "telco_churn.csv"
    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    assert len(df) > 100, "Dataset suspiciously small — check download"
    logger.info("Schema validation passed: %d rows", len(df))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is object due to spaces for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nulls = df["TotalCharges"].isna().sum()
    if n_nulls > 0:
        logger.info("Imputing %d TotalCharges nulls with 0 (new customers)", n_nulls)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop non-informative ID
    df = df.drop(columns=["customerID"])

    assert df["Churn"].isna().sum() == 0, "Churn target has nulls after cleaning"
    assert df.shape[1] > 10, "Too many columns dropped"
    logger.info("Cleaning complete: %d rows, %d cols", *df.shape)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; keep numerics as-is."""
    df = df.copy()
    binary_map = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}

    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    df["gender"] = (df["gender"] == "Male").astype(int)

    nominal_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-meaningful derived features."""
    df = df.copy()

    # Revenue exposure: how much has this customer paid us?
    df["revenue_exposure"] = df["TotalCharges"]

    # Monthly charge volatility proxy: deviation from expected total
    df["charge_vs_expected"] = df["TotalCharges"] - (df["tenure"] * df["MonthlyCharges"])

    # Tenure segments (aligns with business lifecycle stages)
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    )

    # Service count — breadth of product adoption
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    # Only sum the ones that are already encoded as 0/1
    numeric_services = [c for c in service_cols if df[c].dtype in [int, float, "int64", "float64"]]
    if numeric_services:
        df["service_count"] = df[numeric_services].sum(axis=1)

    return df


def save_processed(df: pd.DataFrame, name: str = "churn_processed.parquet") -> pathlib.Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path, index=False)
    logger.info("Saved processed data to %s", path)
    return path


def run_pipeline() -> pd.DataFrame:
    """Full preprocessing pipeline: load → clean → encode → feature-engineer."""
    raw = load_raw()
    cleaned = clean(raw)
    encoded = encode(cleaned)
    featured = engineer_features(encoded)
    save_processed(featured)
    return featured


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = run_pipeline()
    print(df.shape)
    print(df.dtypes)
