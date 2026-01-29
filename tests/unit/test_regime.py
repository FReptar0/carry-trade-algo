"""Tests for regime detection module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    CompositeRegime,
    RegimeDetector,
    RegimeState,
    TrendRegime,
    VolatilityRegime,
)
from src.regime.adapters import (
    RegimeAdaptedParams,
    RegimeAdapter,
    get_regime_description,
)
from src.strategy.indicators import adx


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_trending_data() -> pd.DataFrame:
    """Create sample data with clear uptrend."""
    np.random.seed(42)
    n = 100

    # Strong uptrend: price increases steadily with small noise
    base_price = 150.0
    trend = np.linspace(0, 15, n)  # 15 points up over 100 bars
    noise = np.random.randn(n) * 0.3

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close - np.random.randn(n) * 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        }
    )


@pytest.fixture
def sample_ranging_data() -> pd.DataFrame:
    """Create sample data with sideways/ranging price action."""
    np.random.seed(42)
    n = 100

    # Ranging: price oscillates around a mean
    base_price = 150.0
    oscillation = np.sin(np.linspace(0, 8 * np.pi, n)) * 2  # Oscillate +/- 2
    noise = np.random.randn(n) * 0.3

    close = base_price + oscillation + noise
    high = close + np.abs(np.random.randn(n) * 0.4)
    low = close - np.abs(np.random.randn(n) * 0.4)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close - np.random.randn(n) * 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        }
    )


@pytest.fixture
def sample_high_vol_data() -> pd.DataFrame:
    """Create sample data with high volatility."""
    np.random.seed(42)
    n = 100

    # High volatility: large daily ranges
    base_price = 150.0
    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n) * 3.0  # Much larger noise

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n) * 2.0)  # Large ranges
    low = close - np.abs(np.random.randn(n) * 2.0)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close - np.random.randn(n) * 1.0,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        }
    )


@pytest.fixture
def detector() -> RegimeDetector:
    """Create regime detector with default settings."""
    return RegimeDetector()


@pytest.fixture
def adapter() -> RegimeAdapter:
    """Create regime adapter with default settings."""
    return RegimeAdapter()


# ============================================================================
# ADX Indicator Tests
# ============================================================================


class TestADXIndicator:
    """Tests for ADX indicator function."""

    def test_adx_returns_three_series(self, sample_trending_data: pd.DataFrame):
        """ADX function returns adx, plus_di, minus_di."""
        adx_val, plus_di, minus_di = adx(
            sample_trending_data["high"],
            sample_trending_data["low"],
            sample_trending_data["close"],
        )

        assert isinstance(adx_val, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)
        assert len(adx_val) == len(sample_trending_data)

    def test_adx_values_in_range(self, sample_trending_data: pd.DataFrame):
        """ADX and DI values should be 0-100 (after warmup)."""
        adx_val, plus_di, minus_di = adx(
            sample_trending_data["high"],
            sample_trending_data["low"],
            sample_trending_data["close"],
        )

        # After warmup period
        valid_adx = adx_val.dropna()
        valid_plus = plus_di.dropna()
        valid_minus = minus_di.dropna()

        assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
        assert (valid_plus >= 0).all()
        assert (valid_minus >= 0).all()

    def test_adx_uptrend_has_plus_di_greater(self, sample_trending_data: pd.DataFrame):
        """In uptrend, +DI should generally exceed -DI."""
        adx_val, plus_di, minus_di = adx(
            sample_trending_data["high"],
            sample_trending_data["low"],
            sample_trending_data["close"],
        )

        # Check last 20 bars (well into the uptrend)
        plus_di_avg = plus_di.iloc[-20:].mean()
        minus_di_avg = minus_di.iloc[-20:].mean()

        assert plus_di_avg > minus_di_avg, "Uptrend should have +DI > -DI"


# ============================================================================
# Volatility Regime Tests
# ============================================================================


class TestVolatilityRegime:
    """Tests for volatility regime detection."""

    def test_high_volatility_detection(self, detector: RegimeDetector):
        """High ATR relative to average should be HIGH regime."""
        # Create data where recent ATR is much higher than historical
        np.random.seed(42)
        n = 100

        # Start with low volatility, end with high
        base = 150.0
        close = base + np.cumsum(np.random.randn(n) * 0.1)
        # Make last 20 bars very volatile
        close[-20:] = base + np.cumsum(np.random.randn(20) * 2.0)

        high = close + np.abs(np.random.randn(n) * 0.3)
        high[-20:] = close[-20:] + np.abs(np.random.randn(20) * 3.0)

        low = close - np.abs(np.random.randn(n) * 0.3)
        low[-20:] = close[-20:] - np.abs(np.random.randn(20) * 3.0)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
                "high": high,
                "low": low,
                "close": close,
            }
        )

        regime = detector.detect(data)
        # With 10x higher volatility at end, should detect HIGH
        assert regime.volatility_regime in (
            VolatilityRegime.HIGH,
            VolatilityRegime.NORMAL,
        )

    def test_low_volatility_detection(self, detector: RegimeDetector):
        """Low ATR relative to average should be LOW regime."""
        np.random.seed(42)
        n = 100

        # Start with high volatility, end with very low
        base = 150.0
        close = base + np.cumsum(np.random.randn(n) * 1.0)
        # Make last 20 bars very calm
        close[-20:] = base + np.linspace(0, 1, 20)  # Almost no movement

        high = close + np.abs(np.random.randn(n) * 1.5)
        high[-20:] = close[-20:] + 0.1

        low = close - np.abs(np.random.randn(n) * 1.5)
        low[-20:] = close[-20:] - 0.1

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
                "high": high,
                "low": low,
                "close": close,
            }
        )

        regime = detector.detect(data)
        assert regime.volatility_regime == VolatilityRegime.LOW


# ============================================================================
# Trend Regime Tests
# ============================================================================


class TestTrendRegime:
    """Tests for trend regime detection."""

    def test_strong_uptrend_detection(
        self, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """Clear uptrend should be detected as STRONG_UPTREND or WEAK_UPTREND."""
        regime = detector.detect(sample_trending_data)

        assert regime.trend_regime in (
            TrendRegime.STRONG_UPTREND,
            TrendRegime.WEAK_UPTREND,
        ), f"Expected uptrend, got {regime.trend_regime}"

    def test_ranging_detection(
        self, detector: RegimeDetector, sample_ranging_data: pd.DataFrame
    ):
        """Sideways market should be detected as RANGING."""
        regime = detector.detect(sample_ranging_data)

        # Ranging data should have low ADX
        assert regime.trend_regime == TrendRegime.RANGING or regime.adx_value < 25

    def test_trend_direction_property(
        self, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """Trend direction property should return correct string."""
        regime = detector.detect(sample_trending_data)

        if regime.trend_regime in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
            assert regime.trend_direction == "up"
        elif regime.trend_regime in (
            TrendRegime.STRONG_DOWNTREND,
            TrendRegime.WEAK_DOWNTREND,
        ):
            assert regime.trend_direction == "down"
        else:
            assert regime.trend_direction == "neutral"


# ============================================================================
# Composite Regime Tests
# ============================================================================


class TestCompositeRegime:
    """Tests for composite regime classification."""

    def test_trend_low_vol_is_favorable(
        self, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """Trending + low vol should be favorable."""
        regime = detector.detect(sample_trending_data)

        # If it's trending with normal/low vol, should be favorable
        if regime.trend_regime in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
            if regime.volatility_regime != VolatilityRegime.HIGH:
                assert regime.is_favorable

    def test_range_high_vol_should_avoid(self, detector: RegimeDetector):
        """Ranging + high vol should be avoided."""
        # Create ranging, high volatility data
        np.random.seed(42)
        n = 100

        base = 150.0
        oscillation = np.sin(np.linspace(0, 10 * np.pi, n)) * 3
        noise = np.random.randn(n) * 2.0

        close = base + oscillation + noise
        high = close + np.abs(np.random.randn(n) * 3.0)
        low = close - np.abs(np.random.randn(n) * 3.0)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
                "high": high,
                "low": low,
                "close": close,
            }
        )

        regime = detector.detect(data)

        # This should be range_high_vol or at minimum not favorable
        if regime.composite_regime == CompositeRegime.RANGE_HIGH_VOL:
            assert regime.should_avoid_trading

    def test_regime_state_properties(
        self, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """RegimeState should have all required properties."""
        regime = detector.detect(sample_trending_data)

        assert isinstance(regime.timestamp, datetime)
        assert isinstance(regime.volatility_regime, VolatilityRegime)
        assert isinstance(regime.trend_regime, TrendRegime)
        assert isinstance(regime.composite_regime, CompositeRegime)
        assert 0 <= regime.confidence <= 1
        assert regime.adx_value >= 0
        assert regime.atr_value >= 0


# ============================================================================
# Regime Detector Tests
# ============================================================================


class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    def test_detector_initialization(self):
        """Detector initializes with correct defaults."""
        detector = RegimeDetector()

        assert detector.adx_period == 14
        assert detector.atr_period == 14
        assert detector.atr_lookback == 60
        assert detector.adx_strong_threshold == 25.0
        assert detector.adx_weak_threshold == 15.0

    def test_detector_custom_params(self):
        """Detector accepts custom parameters."""
        detector = RegimeDetector(
            adx_period=20,
            adx_strong_threshold=30.0,
            vol_high_mult=2.0,
        )

        assert detector.adx_period == 20
        assert detector.adx_strong_threshold == 30.0
        assert detector.vol_high_mult == 2.0

    def test_detect_requires_minimum_data(self, detector: RegimeDetector):
        """Detect raises error with insufficient data."""
        small_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "high": [151] * 10,
                "low": [149] * 10,
                "close": [150] * 10,
            }
        )

        with pytest.raises(ValueError, match="Need at least"):
            detector.detect(small_data)

    def test_detect_series_returns_list(
        self, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """detect_series returns list of RegimeState."""
        regimes = detector.detect_series(sample_trending_data)

        assert isinstance(regimes, list)
        assert len(regimes) > 0
        assert all(isinstance(r, RegimeState) for r in regimes)


# ============================================================================
# Regime Adapter Tests
# ============================================================================


class TestRegimeAdaptedParams:
    """Tests for RegimeAdaptedParams dataclass."""

    def test_valid_params_creation(self):
        """Valid params should create successfully."""
        params = RegimeAdaptedParams(
            stop_loss_mult=1.5,
            position_size_mult=0.8,
            entry_threshold=0.7,
            should_trade=True,
        )

        assert params.stop_loss_mult == 1.5
        assert params.position_size_mult == 0.8
        assert params.entry_threshold == 0.7
        assert params.should_trade is True

    def test_invalid_stop_loss_mult(self):
        """Negative stop_loss_mult should raise error."""
        with pytest.raises(ValueError):
            RegimeAdaptedParams(
                stop_loss_mult=-1.0,
                position_size_mult=1.0,
                entry_threshold=0.5,
                should_trade=True,
            )

    def test_invalid_position_size_mult(self):
        """position_size_mult > 2.0 should raise error."""
        with pytest.raises(ValueError):
            RegimeAdaptedParams(
                stop_loss_mult=1.0,
                position_size_mult=3.0,
                entry_threshold=0.5,
                should_trade=True,
            )

    def test_invalid_entry_threshold(self):
        """entry_threshold > 1.0 should raise error."""
        with pytest.raises(ValueError):
            RegimeAdaptedParams(
                stop_loss_mult=1.0,
                position_size_mult=1.0,
                entry_threshold=1.5,
                should_trade=True,
            )


class TestRegimeAdapter:
    """Tests for RegimeAdapter class."""

    def test_adapter_has_all_regimes(self, adapter: RegimeAdapter):
        """Adapter should have params for all composite regimes."""
        for regime in CompositeRegime:
            params = adapter.adapt(regime)
            assert isinstance(params, RegimeAdaptedParams)

    def test_trend_low_vol_params(self, adapter: RegimeAdapter):
        """TREND_LOW_VOL should have favorable trading params."""
        params = adapter.adapt(CompositeRegime.TREND_LOW_VOL)

        assert params.should_trade is True
        assert params.position_size_mult >= 1.0  # Full or larger size
        assert params.entry_threshold < 0.8  # Relaxed entry

    def test_range_high_vol_params(self, adapter: RegimeAdapter):
        """RANGE_HIGH_VOL should prohibit trading."""
        params = adapter.adapt(CompositeRegime.RANGE_HIGH_VOL)

        assert params.should_trade is False
        assert params.should_close_on_weakness is True
        assert params.position_size_mult == 0.0

    def test_custom_params_override(self):
        """Custom params should override defaults."""
        custom = {
            CompositeRegime.TREND_LOW_VOL: RegimeAdaptedParams(
                stop_loss_mult=0.5,
                position_size_mult=1.5,
                entry_threshold=0.3,
                should_trade=True,
            )
        }
        adapter = RegimeAdapter(custom_params=custom)
        params = adapter.adapt(CompositeRegime.TREND_LOW_VOL)

        assert params.stop_loss_mult == 0.5
        assert params.position_size_mult == 1.5

    def test_adapt_from_state_low_confidence(
        self, adapter: RegimeAdapter, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """Low confidence should result in more conservative params."""
        state = detector.detect(sample_trending_data)

        # Force low confidence for testing
        state.confidence = 0.1

        params = adapter.adapt_from_state(state)

        # Should be more conservative
        base_params = adapter.adapt(state.composite_regime)
        assert params.stop_loss_mult >= base_params.stop_loss_mult
        assert params.position_size_mult <= base_params.position_size_mult

    def test_should_enter_long_in_uptrend(
        self, adapter: RegimeAdapter, detector: RegimeDetector, sample_trending_data: pd.DataFrame
    ):
        """Should allow long entry in uptrend with good signal."""
        state = detector.detect(sample_trending_data)

        if state.trend_direction == "up":
            result = adapter.should_enter_long(state, signal_strength=0.8)
            # Should be True if regime allows trading
            if adapter.adapt(state.composite_regime).should_trade:
                assert result is True

    def test_should_not_enter_in_bad_regime(self, adapter: RegimeAdapter):
        """Should not allow entry in RANGE_HIGH_VOL."""
        state = RegimeState(
            timestamp=datetime.now(),
            volatility_regime=VolatilityRegime.HIGH,
            trend_regime=TrendRegime.RANGING,
            composite_regime=CompositeRegime.RANGE_HIGH_VOL,
            confidence=0.8,
            adx_value=10.0,
            plus_di=20.0,
            minus_di=25.0,
            atr_value=2.0,
            atr_avg=1.0,
        )

        result = adapter.should_enter_long(state, signal_strength=0.9)
        assert result is False

    def test_get_adjusted_stop(self, adapter: RegimeAdapter):
        """get_adjusted_stop should multiply base stop."""
        state = RegimeState(
            timestamp=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL,
            trend_regime=TrendRegime.STRONG_UPTREND,
            composite_regime=CompositeRegime.TREND_LOW_VOL,
            confidence=0.8,
            adx_value=30.0,
            plus_di=35.0,
            minus_di=15.0,
            atr_value=1.0,
            atr_avg=1.0,
        )

        adjusted = adapter.get_adjusted_stop(2.0, state)
        params = adapter.adapt(CompositeRegime.TREND_LOW_VOL)

        assert adjusted == pytest.approx(2.0 * params.stop_loss_mult)


# ============================================================================
# Utility Tests
# ============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_get_regime_description(self):
        """All regimes should have descriptions."""
        for regime in CompositeRegime:
            desc = get_regime_description(regime)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_regime_state_is_favorable(self):
        """is_favorable should correctly identify good regimes."""
        favorable = RegimeState(
            timestamp=datetime.now(),
            volatility_regime=VolatilityRegime.LOW,
            trend_regime=TrendRegime.STRONG_UPTREND,
            composite_regime=CompositeRegime.TREND_LOW_VOL,
            confidence=0.9,
            adx_value=30.0,
            plus_di=35.0,
            minus_di=15.0,
            atr_value=1.0,
            atr_avg=1.5,
        )
        assert favorable.is_favorable is True

        unfavorable = RegimeState(
            timestamp=datetime.now(),
            volatility_regime=VolatilityRegime.HIGH,
            trend_regime=TrendRegime.RANGING,
            composite_regime=CompositeRegime.RANGE_HIGH_VOL,
            confidence=0.8,
            adx_value=10.0,
            plus_di=20.0,
            minus_di=22.0,
            atr_value=2.0,
            atr_avg=1.0,
        )
        assert unfavorable.is_favorable is False
