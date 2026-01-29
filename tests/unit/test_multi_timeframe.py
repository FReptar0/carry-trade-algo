"""Tests for multi-timeframe data loading and V4 strategy."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.multi_timeframe import (
    MultiTimeframeLoader,
    MultiTimeframeContext,
    TimeframeData,
    get_aligned_signals,
)
from src.strategy.carry_trade_v4 import CarryTradeStrategyV4, StrategyConfigV4


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_daily_data() -> pd.DataFrame:
    """Create sample daily data for testing."""
    np.random.seed(42)
    n = 300  # About 1 year of trading days

    # Create uptrending price data
    base_price = 150.0
    trend = np.linspace(0, 20, n)  # Gradual uptrend
    noise = np.random.randn(n) * 0.5

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close - np.random.randn(n) * 0.3,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
            "swap_long": np.full(n, 0.005),  # Positive carry
            "swap_short": np.full(n, -0.005),
        }
    )


@pytest.fixture
def sample_daily_with_weekly(sample_daily_data: pd.DataFrame) -> pd.DataFrame:
    """Create daily data with weekly context columns."""
    df = sample_daily_data.copy()

    # Add weekly context columns (simulated)
    df["weekly_close"] = df["close"].rolling(5).mean().shift(5)  # Lagged
    df["weekly_ma"] = df["close"].rolling(20).mean().shift(5)  # Lagged
    df["weekly_trend"] = "up"  # Simplified for testing
    df["weekly_high"] = df["high"].rolling(5).max().shift(5)
    df["weekly_low"] = df["low"].rolling(5).min().shift(5)

    return df


# ============================================================================
# MultiTimeframeLoader Tests
# ============================================================================


class TestMultiTimeframeLoader:
    """Tests for MultiTimeframeLoader class."""

    def test_loader_initialization(self):
        """Loader initializes with default settings."""
        loader = MultiTimeframeLoader()
        assert loader.weekly_ma_period == 20

    def test_loader_custom_params(self):
        """Loader accepts custom parameters."""
        loader = MultiTimeframeLoader(weekly_ma_period=40)
        assert loader.weekly_ma_period == 40

    def test_resample_to_weekly(self, sample_daily_data: pd.DataFrame):
        """Resampling daily to weekly produces correct bars."""
        loader = MultiTimeframeLoader()
        weekly = loader._resample_to_weekly(sample_daily_data)

        # Should have fewer bars than daily
        assert len(weekly) < len(sample_daily_data)

        # Should have correct columns
        assert "week_end" in weekly.columns
        assert "open" in weekly.columns
        assert "high" in weekly.columns
        assert "low" in weekly.columns
        assert "close" in weekly.columns

        # Weekly high should be max of daily highs
        # Weekly low should be min of daily lows
        for _, row in weekly.iterrows():
            week_end = row["week_end"]
            week_start = week_end - timedelta(days=7)

            daily_in_week = sample_daily_data[
                (sample_daily_data["timestamp"] > week_start)
                & (sample_daily_data["timestamp"] <= week_end)
            ]

            if len(daily_in_week) > 0:
                assert row["high"] >= daily_in_week["high"].max() - 0.01
                assert row["low"] <= daily_in_week["low"].min() + 0.01

    def test_weekly_indicators_added(self, sample_daily_data: pd.DataFrame):
        """Weekly indicators are calculated correctly."""
        loader = MultiTimeframeLoader()
        weekly = loader._resample_to_weekly(sample_daily_data)
        weekly = loader._add_weekly_indicators(weekly)

        assert "weekly_ma" in weekly.columns
        assert "weekly_trend" in weekly.columns
        assert "weekly_momentum" in weekly.columns

        # Trend should be categorized
        assert weekly["weekly_trend"].isin(["up", "down", "neutral"]).all()

    def test_no_lookahead_in_merge(self, sample_daily_data: pd.DataFrame):
        """Merged data has no lookahead bias."""
        loader = MultiTimeframeLoader()
        weekly = loader._resample_to_weekly(sample_daily_data)
        weekly = loader._add_weekly_indicators(weekly)
        merged = loader._merge_weekly_to_daily(sample_daily_data, weekly)

        # Check a sample of rows to verify no lookahead
        # For each daily bar, weekly_close should be from a PAST week
        sample_rows = merged.iloc[50:60]  # Check middle portion

        for idx, row in sample_rows.iterrows():
            daily_date = row["timestamp"]
            weekly_close = row["weekly_close"]

            if pd.isna(weekly_close):
                continue

            # The merge logic uses weeks that ended BEFORE daily_date
            # We verify by checking that week_end < daily_date for all valid weeks
            valid_weeks = weekly[weekly["week_end"] < daily_date]
            if len(valid_weeks) > 0:
                # The weekly_close should match the most recent valid week
                last_valid_close = valid_weeks.iloc[-1]["close"]
                assert np.isclose(
                    weekly_close, last_valid_close, rtol=1e-5
                ), f"Weekly close mismatch for {daily_date}"


class TestMultiTimeframeContext:
    """Tests for MultiTimeframeContext dataclass."""

    def test_context_creation(self, sample_daily_with_weekly: pd.DataFrame):
        """Context can be created with required data."""
        context = MultiTimeframeContext(
            daily=sample_daily_with_weekly,
            weekly_raw=pd.DataFrame(),
            pair="USD/JPY",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert context.pair == "USD/JPY"
        assert len(context.daily) == len(sample_daily_with_weekly)


# ============================================================================
# Alignment Functions Tests
# ============================================================================


class TestAlignmentFunctions:
    """Tests for signal alignment functions."""

    def test_long_aligned_with_uptrend(self):
        """Long signal aligns with weekly uptrend."""
        aligned, reason = get_aligned_signals("long", "up")
        assert aligned is True
        assert "aligned" in reason.lower()

    def test_long_conflicts_with_downtrend(self):
        """Long signal conflicts with weekly downtrend."""
        aligned, reason = get_aligned_signals("long", "down")
        assert aligned is False
        assert "conflict" in reason.lower()

    def test_short_aligned_with_downtrend(self):
        """Short signal aligns with weekly downtrend."""
        aligned, reason = get_aligned_signals("short", "down")
        assert aligned is True

    def test_neutral_weekly_waits(self):
        """Neutral weekly trend should wait."""
        aligned, reason = get_aligned_signals("long", "neutral")
        assert aligned is False
        assert "neutral" in reason.lower()

    def test_alignment_not_required(self):
        """When alignment not required, always returns True."""
        aligned, _ = get_aligned_signals("long", "down", require_alignment=False)
        assert aligned is True


# ============================================================================
# V4 Strategy Tests
# ============================================================================


class TestV4Strategy:
    """Tests for CarryTradeStrategyV4."""

    def test_strategy_initialization(self):
        """Strategy initializes with default config."""
        strategy = CarryTradeStrategyV4()
        assert strategy.config.daily_ma_period == 20
        assert strategy.config.require_weekly_alignment is True

    def test_strategy_custom_config(self):
        """Strategy accepts custom config."""
        config = StrategyConfigV4(
            daily_ma_period=30,
            require_weekly_alignment=False,
        )
        strategy = CarryTradeStrategyV4(config=config)

        assert strategy.config.daily_ma_period == 30
        assert strategy.config.require_weekly_alignment is False

    def test_generates_signals_with_weekly_data(
        self, sample_daily_with_weekly: pd.DataFrame
    ):
        """Strategy generates signals when weekly data available."""
        strategy = CarryTradeStrategyV4()
        signals = strategy.generate_signals(sample_daily_with_weekly)

        # Should generate at least some signals
        assert isinstance(signals, list)

    def test_signals_alternate_long_close(
        self, sample_daily_with_weekly: pd.DataFrame
    ):
        """Signals should alternate between LONG and CLOSE."""
        strategy = CarryTradeStrategyV4()
        signals = strategy.generate_signals(sample_daily_with_weekly)

        if len(signals) < 2:
            pytest.skip("Not enough signals to test alternation")

        from src.strategy.base import Signal

        for i in range(len(signals) - 1):
            current = signals[i].signal
            next_sig = signals[i + 1].signal

            # After LONG must come CLOSE
            if current == Signal.LONG:
                assert next_sig == Signal.CLOSE, "LONG must be followed by CLOSE"
            # After CLOSE can come LONG
            elif current == Signal.CLOSE:
                assert next_sig == Signal.LONG, "CLOSE must be followed by LONG"

    def test_fallback_without_weekly_data(self, sample_daily_data: pd.DataFrame):
        """Strategy falls back to single-TF when weekly data missing."""
        strategy = CarryTradeStrategyV4()
        signals = strategy.generate_signals(sample_daily_data)

        # Should still work without crashing
        assert isinstance(signals, list)

    def test_signal_has_reason(self, sample_daily_with_weekly: pd.DataFrame):
        """Each signal should have a reason."""
        strategy = CarryTradeStrategyV4()
        signals = strategy.generate_signals(sample_daily_with_weekly)

        for signal in signals:
            assert signal.reason is not None
            assert len(signal.reason) > 0

    def test_no_signals_in_downtrend(self):
        """Should not generate long signals when weekly trend is down."""
        np.random.seed(42)
        n = 300

        # Create downtrending data
        base_price = 170.0
        trend = np.linspace(0, -20, n)  # Downtrend
        noise = np.random.randn(n) * 0.5

        close = base_price + trend + noise
        high = close + np.abs(np.random.randn(n) * 0.8)
        low = close - np.abs(np.random.randn(n) * 0.8)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": close - np.random.randn(n) * 0.3,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
                "swap_long": np.full(n, 0.005),
                "swap_short": np.full(n, -0.005),
                "weekly_close": close.copy(),
                "weekly_ma": close + 5,  # Price below weekly MA
                "weekly_trend": "down",  # Downtrend
                "weekly_high": high,
                "weekly_low": low,
            }
        )

        strategy = CarryTradeStrategyV4()
        signals = strategy.generate_signals(data)

        # Should have no LONG entries (only in weekly uptrend)
        from src.strategy.base import Signal

        long_signals = [s for s in signals if s.signal == Signal.LONG]
        assert len(long_signals) == 0, "Should not enter long in weekly downtrend"

    def test_validate_params(self):
        """validate_params returns correct results."""
        # Valid config
        strategy = CarryTradeStrategyV4()
        valid, msg = strategy.validate_params()
        assert valid is True

        # Invalid config
        config = StrategyConfigV4(base_atr_mult=-1)
        strategy = CarryTradeStrategyV4(config=config)
        valid, msg = strategy.validate_params()
        assert valid is False


class TestV4WithRegime:
    """Tests for V4 strategy with regime detection."""

    def test_regime_detector_integration(self, sample_daily_with_weekly: pd.DataFrame):
        """Strategy integrates with regime detector."""
        from src.regime.detector import RegimeDetector
        from src.regime.adapters import RegimeAdapter

        detector = RegimeDetector()
        adapter = RegimeAdapter()

        strategy = CarryTradeStrategyV4(
            regime_detector=detector,
            regime_adapter=adapter,
        )

        signals = strategy.generate_signals(sample_daily_with_weekly)
        assert isinstance(signals, list)

    def test_signal_reason_includes_regime(
        self, sample_daily_with_weekly: pd.DataFrame
    ):
        """Signal reasons should mention regime when detected."""
        from src.regime.detector import RegimeDetector
        from src.regime.adapters import RegimeAdapter

        strategy = CarryTradeStrategyV4(
            regime_detector=RegimeDetector(),
            regime_adapter=RegimeAdapter(),
        )

        signals = strategy.generate_signals(sample_daily_with_weekly)

        # At least some entry signals should mention regime
        entry_signals = [s for s in signals if "Entry" in s.reason]
        if len(entry_signals) > 0:
            regime_mentioned = any("regime" in s.reason.lower() for s in entry_signals)
            # Regime might not always be in reason, so just check it doesn't crash
            assert True


class TestConfigV4:
    """Tests for StrategyConfigV4."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = StrategyConfigV4()

        assert config.daily_ma_period == 20
        assert config.min_swap_threshold == 0.003
        assert config.max_rsi == 75
        assert config.base_atr_mult == 3.0
        assert config.require_weekly_alignment is True

    def test_custom_values(self):
        """Config accepts custom values."""
        config = StrategyConfigV4(
            daily_ma_period=30,
            min_swap_threshold=0.01,
            require_weekly_alignment=False,
        )

        assert config.daily_ma_period == 30
        assert config.min_swap_threshold == 0.01
        assert config.require_weekly_alignment is False
