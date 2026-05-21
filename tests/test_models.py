"""Unit tests for model evaluation functions."""

import pytest
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.models.train import precision_at_k, business_roi


class TestPrecisionAtK:
    def test_perfect_model(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02])
        assert precision_at_k(y_true, y_prob, k=0.30) == 1.0

    def test_random_model_near_base_rate(self):
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.15, 10_000)
        y_prob = rng.uniform(0, 1, 10_000)
        p = precision_at_k(y_true, y_prob, k=0.10)
        # Should be close to base rate (0.15) with some noise
        assert 0.05 < p < 0.30

    def test_k_of_1_equals_base_rate(self):
        rng = np.random.default_rng(0)
        y_true = rng.binomial(1, 0.20, 1_000)
        y_prob = rng.uniform(0, 1, 1_000)
        p = precision_at_k(y_true, y_prob, k=1.0)
        assert abs(p - y_true.mean()) < 0.01

    def test_returns_float_in_range(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        p = precision_at_k(y_true, y_prob, k=0.40)
        assert 0.0 <= p <= 1.0


class TestBusinessROI:
    def test_perfect_model_positive_roi(self):
        y_true = np.array([1] * 50 + [0] * 950)
        y_prob = np.array([0.99] * 50 + [0.01] * 950)
        result = business_roi(y_true, y_prob, k=0.10)
        assert result["net_roi"] > 0

    def test_roi_keys_present(self):
        y_true = np.zeros(100, dtype=int)
        y_true[:15] = 1
        y_prob = np.random.default_rng(1).uniform(0, 1, 100)
        result = business_roi(y_true, y_prob)
        expected_keys = {
            "n_targeted", "true_churners_caught", "precision_at_k",
            "gross_revenue_saved", "intervention_cost", "net_roi", "vs_random_roi_lift",
        }
        assert expected_keys.issubset(result.keys())

    def test_n_targeted_is_correct_fraction(self):
        y_true = np.zeros(1000, dtype=int)
        y_prob = np.random.default_rng(2).uniform(0, 1, 1000)
        result = business_roi(y_true, y_prob, k=0.20)
        assert result["n_targeted"] == 200

    def test_zero_churners_zero_revenue(self):
        y_true = np.zeros(100, dtype=int)
        y_prob = np.ones(100) * 0.5
        result = business_roi(y_true, y_prob, k=0.10)
        assert result["gross_revenue_saved"] == 0.0
        assert result["net_roi"] < 0  # cost with no revenue = negative ROI
