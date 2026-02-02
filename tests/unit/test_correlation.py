"""Tests for the CorrelationMonitor."""

import numpy as np
import pandas as pd
import pytest

from src.risk.correlation import CorrelationMonitor


@pytest.fixture
def correlated_candles():
    """Two pairs with high correlation."""
    np.random.seed(42)
    n = 100
    base = np.cumsum(np.random.randn(n) * 0.1)
    df1 = pd.DataFrame({"close": 150.0 + base})
    df2 = pd.DataFrame({"close": 95.0 + base * 0.8 + np.random.randn(n) * 0.02})
    return {"USD/JPY": df1, "AUD/JPY": df2}


@pytest.fixture
def uncorrelated_candles():
    """Two pairs with low correlation."""
    np.random.seed(42)
    n = 100
    df1 = pd.DataFrame({"close": 150.0 + np.cumsum(np.random.randn(n) * 0.1)})
    df2 = pd.DataFrame({"close": 1.10 + np.cumsum(np.random.randn(n) * 0.001)})
    return {"USD/JPY": df1, "EUR/USD": df2}


class TestCorrelationMonitor:
    def test_update_with_single_pair(self):
        monitor = CorrelationMonitor()
        monitor.update({"USD/JPY": pd.DataFrame({"close": [150, 151, 152]})})
        assert monitor._correlation_matrix is None  # Need 2+ pairs

    def test_correlated_pairs_detected(self, correlated_candles):
        monitor = CorrelationMonitor(lookback=60)
        monitor.update(correlated_candles)
        corr = monitor.get_correlation("USD/JPY", "AUD/JPY")
        assert corr > 0.5

    def test_portfolio_factor_high_correlation(self, correlated_candles):
        monitor = CorrelationMonitor(lookback=60)
        monitor.update(correlated_candles)
        factor = monitor.portfolio_correlation_factor(["USD/JPY", "AUD/JPY"])
        assert factor < 1.0
        assert factor >= 0.5

    def test_portfolio_factor_single_pair(self):
        monitor = CorrelationMonitor()
        factor = monitor.portfolio_correlation_factor(["USD/JPY"])
        assert factor == 1.0

    def test_portfolio_factor_no_data(self):
        monitor = CorrelationMonitor()
        factor = monitor.portfolio_correlation_factor(["USD/JPY", "AUD/JPY"])
        assert factor == 0.7  # Default for JPY pairs

    def test_get_correlation_unknown_pair(self):
        monitor = CorrelationMonitor()
        corr = monitor.get_correlation("USD/JPY", "EUR/USD")
        assert corr == 0.0

    def test_empty_candles(self):
        monitor = CorrelationMonitor()
        monitor.update({})
        assert monitor._correlation_matrix is None

    def test_short_data(self):
        monitor = CorrelationMonitor()
        monitor.update({
            "A": pd.DataFrame({"close": [1, 2]}),
            "B": pd.DataFrame({"close": [3, 4]}),
        })
        # Too short for meaningful correlation
        assert monitor._correlation_matrix is None

    def test_portfolio_factor_non_jpy_pairs(self):
        monitor = CorrelationMonitor()
        factor = monitor.portfolio_correlation_factor(["EUR/USD", "GBP/USD"])
        assert factor == 1.0  # No JPY, no default penalty
