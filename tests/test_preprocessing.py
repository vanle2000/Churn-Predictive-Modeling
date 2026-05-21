"""Unit tests for preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.data.preprocessing import clean, encode, engineer_features, _validate_schema


def _make_raw_df(n: int = 200) -> pd.DataFrame:
    """Minimal synthetic raw dataset matching Telco schema."""
    rng = np.random.default_rng(42)
    binary = ["Yes", "No"]
    return pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(binary, n),
        "Dependents": rng.choice(binary, n),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(binary, n),
        "MultipleLines": rng.choice(binary + ["No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(binary + ["No internet service"], n),
        "OnlineBackup": rng.choice(binary + ["No internet service"], n),
        "DeviceProtection": rng.choice(binary + ["No internet service"], n),
        "TechSupport": rng.choice(binary + ["No internet service"], n),
        "StreamingTV": rng.choice(binary + ["No internet service"], n),
        "StreamingMovies": rng.choice(binary + ["No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(binary, n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(18, 120, n).round(2),
        "TotalCharges": [str(round(v, 2)) if i > 0 else " " for i, v in enumerate(rng.uniform(0, 8000, n))],
        "Churn": rng.choice(binary, n),
    })


class TestClean:
    def test_removes_customer_id(self):
        df = clean(_make_raw_df())
        assert "customerID" not in df.columns

    def test_churn_is_binary_int(self):
        df = clean(_make_raw_df())
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_total_charges_numeric(self):
        df = clean(_make_raw_df())
        assert df["TotalCharges"].dtype == float

    def test_no_nulls_in_churn(self):
        df = clean(_make_raw_df())
        assert df["Churn"].isna().sum() == 0

    def test_row_count_preserved(self):
        raw = _make_raw_df(300)
        cleaned = clean(raw)
        assert len(cleaned) == 300


class TestEncode:
    def test_no_object_columns_except_tenure_band(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        encoded = encode(cleaned)
        obj_cols = [c for c in encoded.select_dtypes(include="object").columns
                    if c != "tenure_band"]
        assert obj_cols == [], f"Unexpected object columns: {obj_cols}"

    def test_gender_is_binary(self):
        cleaned = clean(_make_raw_df())
        encoded = encode(cleaned)
        assert set(encoded["gender"].unique()).issubset({0, 1})

    def test_dummy_columns_created(self):
        cleaned = clean(_make_raw_df())
        encoded = encode(cleaned)
        assert any(c.startswith("Contract_") for c in encoded.columns)


class TestFeatureEngineering:
    def test_service_count_non_negative(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        encoded = encode(cleaned)
        featured = engineer_features(encoded)
        assert (featured["service_count"] >= 0).all()

    def test_tenure_band_has_expected_categories(self):
        raw = _make_raw_df(500)
        cleaned = clean(raw)
        encoded = encode(cleaned)
        featured = engineer_features(encoded)
        expected = {"0-6m", "6-12m", "1-2y", "2-4y", "4y+"}
        actual = set(featured["tenure_band"].dropna().unique())
        assert actual.issubset(expected)

    def test_charge_vs_expected_computed(self):
        raw = _make_raw_df()
        cleaned = clean(raw)
        encoded = encode(cleaned)
        featured = engineer_features(encoded)
        assert "charge_vs_expected" in featured.columns


class TestSchemaValidation:
    def test_raises_on_missing_column(self):
        raw = _make_raw_df()
        raw = raw.drop(columns=["Churn"])
        with pytest.raises(ValueError, match="Missing expected columns"):
            _validate_schema(raw)

    def test_passes_on_valid_df(self):
        raw = _make_raw_df()
        _validate_schema(raw)  # should not raise
