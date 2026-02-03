"""Tests for the FeatureExtractor."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.data.cross_asset import CrossAssetData
from src.ml.features import FeatureExtractor, FEATURE_NAMES


@pytest.fixture
def sample_df():
    n = 300
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": prices,
            "high": prices + np.random.rand(n) * 0.5,
            "low": prices - np.random.rand(n) * 0.5,
            "close": prices + np.random.randn(n) * 0.1,
            "volume": np.random.randint(100, 1000, n),
            "swap_long": 0.005,
            "swap_short": -0.005,
        }
    )


class TestFeatureExtractor:
    def test_extract_returns_all_features(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_extract_values_are_finite(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        for name, val in features.items():
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_rsi_in_range(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        assert 0 <= features["rsi_14"] <= 100

    def test_hour_and_day_in_range(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        assert 0 <= features["hour_of_day"] <= 23
        assert 0 <= features["day_of_week"] <= 6

    def test_regime_code_valid(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        assert features["regime_code"] in (0, 1, 2, 3)

    def test_to_array(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        arr = extractor.to_array(features)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(FEATURE_NAMES)

    def test_empty_features_on_short_data(self):
        extractor = FeatureExtractor()
        short_df = pd.DataFrame({"close": [1, 2, 3]})
        features = extractor.extract(short_df, "USD/JPY")
        assert all(v == 0.0 for v in features.values())

    def test_swap_rate_extracted(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        assert features["swap_rate"] == 0.005

    def test_default_cross_asset_values_no_fetcher(self, sample_df):
        """Without a fetcher, cross-asset features use defaults."""
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        assert features["vix"] == 20.0
        assert features["dxy"] == 100.0
        assert features["us10y_yield"] == 4.0
        assert features["sp500_change_pct"] == 0.0

    def test_cross_asset_values_with_mock_fetcher(self, sample_df):
        """With a fetcher, cross-asset features come from fetcher data."""
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = CrossAssetData(
            vix=25.0,
            dxy=105.0,
            us10y=4.5,
            sp500=5200.0,
            sp500_prev_close=5100.0,
            fetched_at=0.0,
            is_default=False,
        )
        extractor = FeatureExtractor(cross_asset_fetcher=mock_fetcher)
        features = extractor.extract(sample_df, "USD/JPY")
        assert features["vix"] == 25.0
        assert features["dxy"] == 105.0
        assert features["us10y_yield"] == 4.5
        # (5200 - 5100) / 5100 * 100 ≈ 1.9608
        assert abs(features["sp500_change_pct"] - 100 / 5100 * 100) < 0.01

    def test_feature_count_is_14(self):
        assert len(FEATURE_NAMES) == 14

    def test_to_array_returns_shape_14(self, sample_df):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_df, "USD/JPY")
        arr = extractor.to_array(features)
        assert arr.shape == (14,)
