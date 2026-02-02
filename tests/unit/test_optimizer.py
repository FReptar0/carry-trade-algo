"""Tests for the ParamOptimizer."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.adaptive.optimizer import ParamOptimizer
from src.adaptive.param_store import ParamStore
from src.regime.adapters import RegimeAdaptedParams


@pytest.fixture
def store(tmp_path):
    return ParamStore(str(tmp_path / "test_opt.db"))


@pytest.fixture
def sample_candles():
    """Create minimal candle data for testing."""
    n = 300
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame(
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
    return {"USD/JPY": df}


class TestParamOptimizer:
    def test_default_params_fallback(self, store):
        optimizer = ParamOptimizer(store)
        params = optimizer._default_params("TREND_LOW_VOL")
        assert isinstance(params, RegimeAdaptedParams)
        assert params.should_trade is True

    def test_default_params_range_high_vol(self, store):
        optimizer = ParamOptimizer(store)
        params = optimizer._default_params("RANGE_HIGH_VOL")
        assert params.should_trade is False
        assert params.should_close_on_weakness is True

    def test_default_params_unknown_regime(self, store):
        optimizer = ParamOptimizer(store)
        params = optimizer._default_params("UNKNOWN")
        assert isinstance(params, RegimeAdaptedParams)

    def test_optimize_without_optuna(self, store):
        optimizer = ParamOptimizer(store)
        with patch.dict("sys.modules", {"optuna": None}):
            result = optimizer.optimize_regime(
                "TREND_LOW_VOL", {}, n_trials=1
            )
            assert isinstance(result, RegimeAdaptedParams)

    def test_optimize_with_empty_candles(self, store):
        optimizer = ParamOptimizer(store)
        try:
            import optuna

            result = optimizer.optimize_regime(
                "TREND_LOW_VOL", {}, n_trials=2
            )
            assert isinstance(result, RegimeAdaptedParams)
        except ImportError:
            pytest.skip("optuna not installed")

    def test_optimize_with_short_candles(self, store):
        """Data too short should still complete."""
        short_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-01-01", periods=50, freq="h"
                ),
                "open": np.ones(50) * 150,
                "high": np.ones(50) * 151,
                "low": np.ones(50) * 149,
                "close": np.ones(50) * 150,
                "volume": np.ones(50, dtype=int) * 100,
                "swap_long": 0.005,
                "swap_short": -0.005,
            }
        )
        optimizer = ParamOptimizer(store)
        try:
            import optuna

            result = optimizer.optimize_regime(
                "TREND_LOW_VOL",
                {"USD/JPY": short_df},
                n_trials=2,
            )
            assert isinstance(result, RegimeAdaptedParams)
        except ImportError:
            pytest.skip("optuna not installed")

    def test_optimize_saves_to_store(self, store, sample_candles):
        optimizer = ParamOptimizer(store)
        try:
            import optuna

            optimizer.optimize_regime(
                "TREND_LOW_VOL",
                sample_candles,
                n_trials=3,
            )
            active = store.get_active_params("TREND_LOW_VOL")
            assert active is not None
            assert active.is_active is True
        except ImportError:
            pytest.skip("optuna not installed")
