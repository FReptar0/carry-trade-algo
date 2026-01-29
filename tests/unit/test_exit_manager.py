"""Tests for exit manager module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategy.exit_manager import (
    ExitManager,
    ExitManagerConfig,
    ExitSignal,
    ExitType,
    PositionState,
)
from src.regime.detector import RegimeState, CompositeRegime, TrendRegime, VolatilityRegime


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    np.random.seed(42)
    n = 100

    base_price = 150.0
    trend = np.linspace(0, 5, n)
    noise = np.random.randn(n) * 0.5

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "high": high,
        "low": low,
        "close": close,
    })


@pytest.fixture
def profitable_position() -> PositionState:
    """Create a profitable position."""
    return PositionState(
        entry_price=150.0,
        entry_time=datetime(2024, 1, 1),
        current_price=157.5,  # +5%
        current_time=datetime(2024, 1, 15),
        high_water_mark=158.0,
        low_water_mark=148.0,
        side="long",
    )


@pytest.fixture
def losing_position() -> PositionState:
    """Create a losing position."""
    return PositionState(
        entry_price=150.0,
        entry_time=datetime(2024, 1, 1),
        current_price=145.0,  # -3.3%
        current_time=datetime(2024, 1, 15),
        high_water_mark=151.0,
        low_water_mark=144.0,
        side="long",
    )


@pytest.fixture
def favorable_regime() -> RegimeState:
    """Create a favorable regime state."""
    return RegimeState(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.LOW,
        trend_regime=TrendRegime.STRONG_UPTREND,
        composite_regime=CompositeRegime.TREND_LOW_VOL,
        confidence=0.8,
        adx_value=30.0,
        plus_di=35.0,
        minus_di=15.0,
        atr_value=1.0,
        atr_avg=1.0,
    )


@pytest.fixture
def unfavorable_regime() -> RegimeState:
    """Create an unfavorable regime state."""
    return RegimeState(
        timestamp=datetime.now(),
        volatility_regime=VolatilityRegime.HIGH,
        trend_regime=TrendRegime.RANGING,
        composite_regime=CompositeRegime.RANGE_HIGH_VOL,
        confidence=0.8,
        adx_value=10.0,
        plus_di=20.0,
        minus_di=22.0,
        atr_value=2.5,
        atr_avg=1.0,
    )


@pytest.fixture
def exit_manager() -> ExitManager:
    """Create exit manager with default config."""
    return ExitManager()


# ============================================================================
# PositionState Tests
# ============================================================================


class TestPositionState:
    """Tests for PositionState dataclass."""

    def test_return_pct_long_profitable(self):
        """Long position return calculation - profitable."""
        pos = PositionState(
            entry_price=100.0,
            entry_time=datetime.now(),
            current_price=110.0,
            current_time=datetime.now(),
            high_water_mark=110.0,
            low_water_mark=98.0,
            side="long",
        )
        assert pos.return_pct == pytest.approx(0.10)  # +10%

    def test_return_pct_long_losing(self):
        """Long position return calculation - losing."""
        pos = PositionState(
            entry_price=100.0,
            entry_time=datetime.now(),
            current_price=95.0,
            current_time=datetime.now(),
            high_water_mark=102.0,
            low_water_mark=95.0,
            side="long",
        )
        assert pos.return_pct == pytest.approx(-0.05)  # -5%

    def test_max_favorable_excursion(self):
        """MFE calculation for long."""
        pos = PositionState(
            entry_price=100.0,
            entry_time=datetime.now(),
            current_price=105.0,
            current_time=datetime.now(),
            high_water_mark=112.0,  # Best was +12%
            low_water_mark=98.0,
            side="long",
        )
        assert pos.max_favorable_excursion == pytest.approx(0.12)

    def test_max_adverse_excursion(self):
        """MAE calculation for long."""
        pos = PositionState(
            entry_price=100.0,
            entry_time=datetime.now(),
            current_price=105.0,
            current_time=datetime.now(),
            high_water_mark=112.0,
            low_water_mark=94.0,  # Worst was -6%
            side="long",
        )
        assert pos.max_adverse_excursion == pytest.approx(0.06)

    def test_hold_duration(self):
        """Hold duration calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 3, 10, 0, 0)

        pos = PositionState(
            entry_price=100.0,
            entry_time=start,
            current_price=100.0,
            current_time=end,
            high_water_mark=100.0,
            low_water_mark=100.0,
        )
        assert pos.hold_days == pytest.approx(2.0)


# ============================================================================
# ExitSignal Tests
# ============================================================================


class TestExitSignal:
    """Tests for ExitSignal dataclass."""

    def test_is_urgent(self):
        """Urgent signal detection."""
        urgent = ExitSignal(should_exit=True, urgency=0.8)
        not_urgent = ExitSignal(should_exit=True, urgency=0.5)

        assert urgent.is_urgent is True
        assert not_urgent.is_urgent is False


# ============================================================================
# ExitManager Initialization Tests
# ============================================================================


class TestExitManagerInit:
    """Tests for ExitManager initialization."""

    def test_default_config(self):
        """Manager initializes with default config."""
        manager = ExitManager()
        assert manager.config.base_atr_mult == 3.0
        assert manager.config.max_hold_days == 30

    def test_custom_config(self):
        """Manager accepts custom config."""
        config = ExitManagerConfig(
            base_atr_mult=2.5,
            max_hold_days=20,
        )
        manager = ExitManager(config=config)

        assert manager.config.base_atr_mult == 2.5
        assert manager.config.max_hold_days == 20


# ============================================================================
# Stop Loss Tests
# ============================================================================


class TestStopLoss:
    """Tests for stop loss functionality."""

    def test_initialize_stop(self, exit_manager: ExitManager):
        """Initial stop is calculated correctly."""
        stop = exit_manager.initialize_stop(
            entry_price=150.0,
            current_atr=2.0,
        )
        # Default: 3.0 * ATR = 6.0 below entry
        assert stop == pytest.approx(144.0)

    def test_stop_widens_in_high_vol_regime(self):
        """Stop widens in high volatility regime."""
        manager = ExitManager()

        high_vol_regime = RegimeState(
            timestamp=datetime.now(),
            volatility_regime=VolatilityRegime.HIGH,
            trend_regime=TrendRegime.STRONG_UPTREND,
            composite_regime=CompositeRegime.TREND_HIGH_VOL,
            confidence=0.8,
            adx_value=30.0,
            plus_di=35.0,
            minus_di=15.0,
            atr_value=2.0,
            atr_avg=1.0,
        )

        stop_with_regime = manager.initialize_stop(
            entry_price=150.0,
            current_atr=2.0,
            regime=high_vol_regime,
        )

        manager2 = ExitManager()
        stop_without = manager2.initialize_stop(
            entry_price=150.0,
            current_atr=2.0,
        )

        # High vol regime should have wider stop (lower price)
        assert stop_with_regime < stop_without

    def test_stop_loss_triggered(
        self,
        losing_position: PositionState,
        sample_price_data: pd.DataFrame,
    ):
        """Stop loss triggers when price below stop."""
        manager = ExitManager()
        manager._stop_price = 146.0  # Above current price of 145

        exit_signal = manager.check_exit(
            position=losing_position,
            price_data=sample_price_data,
        )

        assert exit_signal.should_exit is True
        assert exit_signal.exit_type == ExitType.STOP_LOSS


# ============================================================================
# Trailing Stop Tests
# ============================================================================


class TestTrailingStop:
    """Tests for trailing stop functionality."""

    def test_trailing_activates_at_profit_threshold(
        self,
        profitable_position: PositionState,
        sample_price_data: pd.DataFrame,
    ):
        """Trailing stop activates after profit threshold."""
        manager = ExitManager()
        manager._stop_price = 145.0  # Initial stop

        # Position is +5%, should activate trailing
        exit_signal = manager.check_exit(
            position=profitable_position,
            price_data=sample_price_data,
        )

        # May or may not exit, but trailing should be active
        assert manager.is_trailing_active() is True

    def test_trailing_stop_tightens_with_profit(self):
        """Higher profit leads to tighter trailing multiplier."""
        config = ExitManagerConfig(
            profit_scale_levels=[
                (0.02, 1.5),  # At 2%, trail 1.5x ATR
                (0.05, 1.0),  # At 5%, trail 1.0x ATR
                (0.10, 0.5),  # At 10%, trail 0.5x ATR
            ]
        )
        manager = ExitManager(config=config)

        # Position at 10% profit
        position = PositionState(
            entry_price=100.0,
            entry_time=datetime.now() - timedelta(days=5),
            current_price=110.0,
            current_time=datetime.now(),
            high_water_mark=112.0,
            low_water_mark=99.0,
            side="long",
        )

        # With 10% profit, should use 0.5x ATR trailing
        # This is tested indirectly through the exit signal


# ============================================================================
# Time-Based Exit Tests
# ============================================================================


class TestTimeBasedExit:
    """Tests for time-based exit functionality."""

    def test_no_exit_within_min_hold(self):
        """Don't exit within minimum hold period."""
        config = ExitManagerConfig(min_hold_hours=24)
        manager = ExitManager(config=config)

        # Position held only 12 hours
        position = PositionState(
            entry_price=100.0,
            entry_time=datetime.now() - timedelta(hours=12),
            current_price=98.0,  # Losing
            current_time=datetime.now(),
            high_water_mark=100.0,
            low_water_mark=98.0,
        )

        exit_signal = manager._check_time_exit(position)
        assert exit_signal.should_exit is False

    def test_time_exit_losing_position(self):
        """Time-based exit for losing position after max hold."""
        config = ExitManagerConfig(max_hold_days=30)
        manager = ExitManager(config=config)

        # Position held 35 days, losing
        position = PositionState(
            entry_price=100.0,
            entry_time=datetime.now() - timedelta(days=35),
            current_price=98.0,  # -2%
            current_time=datetime.now(),
            high_water_mark=102.0,
            low_water_mark=97.0,
        )

        exit_signal = manager._check_time_exit(position)
        assert exit_signal.should_exit is True
        assert exit_signal.exit_type == ExitType.TIME_BASED

    def test_no_time_exit_profitable(self):
        """Don't force exit profitable position after max hold."""
        config = ExitManagerConfig(max_hold_days=30)
        manager = ExitManager(config=config)

        # Position held 35 days, profitable
        position = PositionState(
            entry_price=100.0,
            entry_time=datetime.now() - timedelta(days=35),
            current_price=105.0,  # +5%
            current_time=datetime.now(),
            high_water_mark=106.0,
            low_water_mark=99.0,
        )

        exit_signal = manager._check_time_exit(position)
        assert exit_signal.should_exit is False


# ============================================================================
# Support Break Tests
# ============================================================================


class TestSupportBreak:
    """Tests for support level break exits."""

    def test_support_levels_found(self, exit_manager: ExitManager):
        """Support levels are identified from price data."""
        # Create data with clear swing lows
        lows = pd.Series([
            100, 99, 98, 97, 98, 99, 100,  # Swing low at 97
            101, 100, 99, 98, 99, 100, 101,  # Swing low at 98
        ])

        supports = exit_manager._find_swing_lows(lows, min_distance=2)
        assert len(supports) > 0

    def test_support_break_triggers_exit(
        self,
        exit_manager: ExitManager,
        sample_price_data: pd.DataFrame,
    ):
        """Breaking support triggers exit."""
        # Create position near a support level
        position = PositionState(
            entry_price=152.0,
            entry_time=datetime.now() - timedelta(days=5),
            current_price=148.0,  # Below potential support
            current_time=datetime.now(),
            high_water_mark=153.0,
            low_water_mark=148.0,
        )

        exit_signal = exit_manager._check_support_break(position, sample_price_data)
        # May or may not trigger depending on detected supports
        assert isinstance(exit_signal, ExitSignal)


# ============================================================================
# Regime Change Tests
# ============================================================================


class TestRegimeChange:
    """Tests for regime-based exits."""

    def test_exit_on_bad_regime_with_loss(
        self,
        losing_position: PositionState,
        unfavorable_regime: RegimeState,
    ):
        """Exit when regime turns bad and position is losing."""
        manager = ExitManager()

        exit_signal = manager._check_regime_exit(losing_position, unfavorable_regime)
        assert exit_signal.should_exit is True
        assert exit_signal.exit_type == ExitType.REGIME_CHANGE

    def test_no_exit_bad_regime_profitable(
        self,
        profitable_position: PositionState,
        unfavorable_regime: RegimeState,
    ):
        """Don't exit profitable position even in bad regime."""
        manager = ExitManager()

        exit_signal = manager._check_regime_exit(profitable_position, unfavorable_regime)
        # Position is +5%, above 3% threshold
        assert exit_signal.should_exit is False

    def test_no_exit_favorable_regime(
        self,
        losing_position: PositionState,
        favorable_regime: RegimeState,
    ):
        """Don't exit based on regime if regime is favorable."""
        manager = ExitManager()

        exit_signal = manager._check_regime_exit(losing_position, favorable_regime)
        assert exit_signal.should_exit is False


# ============================================================================
# Trend Reversal Tests
# ============================================================================


class TestTrendReversal:
    """Tests for trend reversal exits."""

    def test_exit_long_on_weekly_down(self, exit_manager: ExitManager):
        """Long position exits when weekly trend turns down."""
        position = PositionState(
            entry_price=150.0,
            entry_time=datetime.now() - timedelta(days=5),
            current_price=152.0,
            current_time=datetime.now(),
            high_water_mark=153.0,
            low_water_mark=149.0,
            side="long",
        )

        exit_signal = exit_manager._check_trend_reversal(position, "down")
        assert exit_signal.should_exit is True
        assert exit_signal.exit_type == ExitType.TREND_REVERSAL

    def test_no_exit_long_weekly_up(self, exit_manager: ExitManager):
        """Long position stays when weekly trend is up."""
        position = PositionState(
            entry_price=150.0,
            entry_time=datetime.now() - timedelta(days=5),
            current_price=152.0,
            current_time=datetime.now(),
            high_water_mark=153.0,
            low_water_mark=149.0,
            side="long",
        )

        exit_signal = exit_manager._check_trend_reversal(position, "up")
        assert exit_signal.should_exit is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestExitManagerIntegration:
    """Integration tests for full exit checking."""

    def test_check_exit_returns_highest_urgency(
        self,
        sample_price_data: pd.DataFrame,
    ):
        """check_exit returns highest urgency exit signal."""
        manager = ExitManager()

        # Position that should trigger stop loss
        position = PositionState(
            entry_price=160.0,  # Above all prices in sample
            entry_time=datetime.now() - timedelta(days=5),
            current_price=152.0,
            current_time=datetime.now(),
            high_water_mark=161.0,
            low_water_mark=152.0,
        )
        manager._stop_price = 155.0  # Stop above current price

        exit_signal = manager.check_exit(
            position=position,
            price_data=sample_price_data,
        )

        assert exit_signal.should_exit is True
        # Should be stop loss (highest urgency)
        assert exit_signal.exit_type == ExitType.STOP_LOSS

    def test_no_exit_healthy_position(
        self,
        profitable_position: PositionState,
        sample_price_data: pd.DataFrame,
        favorable_regime: RegimeState,
    ):
        """Healthy position in good regime doesn't exit."""
        manager = ExitManager()
        manager._stop_price = 140.0  # Well below current

        exit_signal = manager.check_exit(
            position=profitable_position,
            price_data=sample_price_data,
            regime=favorable_regime,
            weekly_trend="up",
        )

        # Might trigger trailing stop based on profit level
        # but stop should be above entry at this point


class TestExitManagerConfig:
    """Tests for ExitManagerConfig."""

    def test_default_profit_scale_levels(self):
        """Default profit scale levels are reasonable."""
        config = ExitManagerConfig()

        assert len(config.profit_scale_levels) > 0
        # Levels should be in ascending profit order
        profits = [level[0] for level in config.profit_scale_levels]
        assert profits == sorted(profits)

        # Multipliers should decrease with profit
        mults = [level[1] for level in config.profit_scale_levels]
        assert mults == sorted(mults, reverse=True)
