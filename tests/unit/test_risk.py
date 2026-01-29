"""Tests for risk management module (Phase 5).

Tests cover:
- Position sizing (Kelly, Fixed Fractional, Volatility-based)
- Adaptive stop loss (ATR-based, trailing)
- Circuit breakers (limit violations, trading halt)
- Risk metrics (VaR, CVaR, MAE/MFE)
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.position_sizing import PositionSizer, SizingMethod
from src.risk.stop_loss import AdaptiveStopLoss, StopType
from src.risk.circuit_breakers import (
    CircuitBreaker,
    RiskLimits,
    PortfolioState,
    LimitType,
)
from src.risk.risk_metrics import (
    RiskAnalyzer,
    TradeExcursion,
    ExcursionAnalysis,
)


# ============== Position Sizing Tests ==============

class TestKellyCriterion:
    def test_positive_edge(self):
        """Kelly returns positive size when there's an edge."""
        sizer = PositionSizer()
        result = sizer.kelly_criterion(
            win_rate=0.60,
            avg_win=0.03,
            avg_loss=0.02,
        )
        assert result.size > 0
        assert result.method == SizingMethod.KELLY

    def test_negative_edge_returns_zero(self):
        """Kelly returns 0 when no edge (losing strategy)."""
        sizer = PositionSizer()
        result = sizer.kelly_criterion(
            win_rate=0.30,  # Low win rate
            avg_win=0.02,
            avg_loss=0.03,  # Bigger losses than wins
        )
        assert result.size == 0
        assert "no trade" in result.details.get("message", "").lower()

    def test_fractional_kelly_applied(self):
        """Full Kelly is reduced by kelly_fraction."""
        sizer = PositionSizer(kelly_fraction=0.25)
        result = sizer.kelly_criterion(
            win_rate=0.60,
            avg_win=0.04,
            avg_loss=0.02,
        )
        full_kelly = result.details["full_kelly"]
        fractional = result.details["fractional_kelly"]
        assert fractional == pytest.approx(full_kelly * 0.25, rel=1e-5)

    def test_position_cap_applied(self):
        """Position size is capped at max_position_pct."""
        sizer = PositionSizer(max_position_pct=0.10, kelly_fraction=1.0)
        result = sizer.kelly_criterion(
            win_rate=0.80,  # Very high win rate
            avg_win=0.10,
            avg_loss=0.02,
        )
        assert result.size <= 0.10
        assert result.capped is True


class TestFixedFractional:
    def test_basic_calculation(self):
        """Fixed fractional calculates correct size."""
        sizer = PositionSizer(default_risk_pct=0.02)
        result = sizer.fixed_fractional(
            capital=100000,
            stop_loss_distance=2.0,  # $2 stop
            entry_price=100.0,
        )
        # Risk $2000 (2% of $100k), stop is 2% of price
        # Position = $2000 / 0.02 = $100,000 = 100% of capital
        assert result.size > 0
        assert result.method == SizingMethod.FIXED_FRACTIONAL

    def test_tighter_stop_larger_size(self):
        """Tighter stop means larger position (same risk amount)."""
        sizer = PositionSizer(default_risk_pct=0.02, max_position_pct=1.0)
        result_tight = sizer.fixed_fractional(capital=100000, stop_loss_distance=1.0, entry_price=100.0)
        result_wide = sizer.fixed_fractional(capital=100000, stop_loss_distance=3.0, entry_price=100.0)
        assert result_tight.raw_size > result_wide.raw_size

    def test_zero_stop_returns_zero(self):
        """Zero stop distance returns zero size."""
        sizer = PositionSizer()
        result = sizer.fixed_fractional(capital=100000, stop_loss_distance=0, entry_price=100.0)
        assert result.size == 0


class TestVolatilityBased:
    def test_higher_atr_smaller_size(self):
        """Higher ATR (more volatile) means smaller position."""
        sizer = PositionSizer(max_position_pct=1.0)
        result_calm = sizer.volatility_based(capital=100000, atr=0.5, entry_price=100.0)
        result_volatile = sizer.volatility_based(capital=100000, atr=2.0, entry_price=100.0)
        assert result_calm.raw_size > result_volatile.raw_size

    def test_zero_atr_returns_zero(self):
        """Zero ATR returns zero size."""
        sizer = PositionSizer()
        result = sizer.volatility_based(capital=100000, atr=0, entry_price=100.0)
        assert result.size == 0


# ============== Stop Loss Tests ==============

class TestAdaptiveStopLoss:
    def test_atr_based_stop_long(self):
        """ATR stop is placed below entry for longs."""
        stop_mgr = AdaptiveStopLoss(atr_multiplier=2.0)
        stop = stop_mgr.initial_stop(entry_price=100.0, is_long=True, atr=1.0)
        assert stop.stop_price < 100.0
        assert stop.stop_price == pytest.approx(98.0, rel=1e-5)
        assert stop.stop_type == StopType.ATR_BASED

    def test_atr_based_stop_short(self):
        """ATR stop is placed above entry for shorts."""
        stop_mgr = AdaptiveStopLoss(atr_multiplier=2.0)
        stop = stop_mgr.initial_stop(entry_price=100.0, is_long=False, atr=1.0)
        assert stop.stop_price > 100.0
        assert stop.stop_price == pytest.approx(102.0, rel=1e-5)

    def test_trailing_moves_up_for_longs(self):
        """Trailing stop moves up when price rises (for longs)."""
        stop_mgr = AdaptiveStopLoss(trailing_activation_pct=0.01)
        stop = stop_mgr.initial_stop(entry_price=100.0, is_long=True, atr=1.0)
        initial_stop = stop.stop_price

        # Price rises to 103 (3% gain, above activation threshold)
        updated = stop_mgr.update_trailing(stop, current_price=103.0, current_atr=1.0)
        assert updated.high_water_mark == 103.0
        assert updated.stop_price > initial_stop  # Stop moved up

    def test_trailing_never_moves_down_for_longs(self):
        """Trailing stop never moves down for longs."""
        stop_mgr = AdaptiveStopLoss(trailing_activation_pct=0.01)
        stop = stop_mgr.initial_stop(entry_price=100.0, is_long=True, atr=1.0)

        # Price rises then falls
        updated1 = stop_mgr.update_trailing(stop, current_price=105.0, current_atr=1.0)
        updated2 = stop_mgr.update_trailing(updated1, current_price=102.0, current_atr=1.0)
        assert updated2.stop_price >= updated1.stop_price

    def test_stop_triggered_detection(self):
        """Stop trigger is detected correctly."""
        stop_mgr = AdaptiveStopLoss()
        stop = stop_mgr.initial_stop(entry_price=100.0, is_long=True, atr=1.0)

        # Price above stop
        assert stop_mgr.check_triggered(stop, current_low=99.0, current_high=101.0) is False
        # Price hits stop
        assert stop_mgr.check_triggered(stop, current_low=97.0, current_high=99.0) is True


class TestATRCalculation:
    def test_atr_calculation(self):
        """ATR is calculated correctly."""
        stop_mgr = AdaptiveStopLoss(atr_period=3)
        df = pd.DataFrame({
            "high": [102, 103, 104, 105, 106],
            "low": [98, 99, 100, 101, 102],
            "close": [100, 101, 102, 103, 104],
        })
        atr = stop_mgr.calculate_atr(df)
        assert len(atr) == 5
        assert not np.isnan(atr.iloc[-1])  # Last value should be valid


# ============== Circuit Breaker Tests ==============

class TestCircuitBreaker:
    def test_daily_loss_limit(self):
        """Trading halts when daily loss exceeds limit."""
        limits = RiskLimits(max_daily_loss_pct=0.03)
        breaker = CircuitBreaker(limits)

        state = PortfolioState(
            equity=97000,
            high_water_mark=100000,
            daily_pnl=-3500,  # 3.5% loss
            weekly_pnl=-3500,
            open_positions=1,
        )
        violations = breaker.check_limits(state)
        assert len(violations) > 0
        assert violations[0].limit_type == LimitType.DAILY_LOSS
        assert breaker.is_trading_halted

    def test_max_positions_limit(self):
        """Can't open more positions than limit."""
        limits = RiskLimits(max_positions=3)
        breaker = CircuitBreaker(limits)

        state = PortfolioState(
            equity=100000,
            high_water_mark=100000,
            daily_pnl=0,
            weekly_pnl=0,
            open_positions=4,  # Exceeds limit
        )
        violations = breaker.check_limits(state)
        assert any(v.limit_type == LimitType.MAX_POSITIONS for v in violations)

    def test_can_open_position_checks(self):
        """can_open_position validates multiple limits."""
        limits = RiskLimits(max_positions=2, max_exposure_per_pair_pct=0.20)
        breaker = CircuitBreaker(limits)

        state = PortfolioState(
            equity=100000,
            high_water_mark=100000,
            daily_pnl=0,
            weekly_pnl=0,
            open_positions=1,
            exposure_by_pair={"USD/JPY": 0.15},
        )

        # Can open new position in different pair
        allowed, reason = breaker.can_open_position(state, "AUD/JPY", 0.10)
        assert allowed is True

        # Can't open if would exceed pair exposure
        allowed, reason = breaker.can_open_position(state, "USD/JPY", 0.10)
        assert allowed is False
        assert "exposure" in reason.lower()

    def test_max_drawdown_limit(self):
        """Trading halts when drawdown exceeds limit."""
        limits = RiskLimits(max_drawdown_pct=0.20)
        breaker = CircuitBreaker(limits)

        state = PortfolioState(
            equity=78000,  # 22% drawdown
            high_water_mark=100000,
            daily_pnl=-1000,
            weekly_pnl=-5000,
            open_positions=1,
        )
        violations = breaker.check_limits(state)
        assert any(v.limit_type == LimitType.MAX_DRAWDOWN for v in violations)

    def test_status_report(self):
        """Status report shows utilization percentages."""
        limits = RiskLimits()
        breaker = CircuitBreaker(limits)

        state = PortfolioState(
            equity=95000,
            high_water_mark=100000,
            daily_pnl=-1000,
            weekly_pnl=-2000,
            open_positions=2,
        )
        report = breaker.get_status_report(state)
        assert "daily_loss_utilization" in report
        assert "drawdown_utilization" in report
        assert 0 <= report["position_utilization"] <= 1


# ============== Risk Metrics Tests ==============

class TestVaR:
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.015, 252))  # 1 year daily

    def test_historical_var(self, sample_returns):
        """Historical VaR returns reasonable values."""
        analyzer = RiskAnalyzer()
        result = analyzer.historical_var(sample_returns, confidence=0.95, portfolio_value=100000)
        assert result.var > 0
        assert result.cvar >= result.var  # CVaR >= VaR by definition
        assert result.method == "historical"

    def test_parametric_var(self, sample_returns):
        """Parametric VaR returns reasonable values."""
        analyzer = RiskAnalyzer()
        result = analyzer.parametric_var(sample_returns, confidence=0.95, portfolio_value=100000)
        assert result.var > 0
        assert result.method == "parametric"

    def test_monte_carlo_var(self, sample_returns):
        """Monte Carlo VaR returns reasonable values."""
        analyzer = RiskAnalyzer()
        result = analyzer.monte_carlo_var(
            sample_returns, confidence=0.95, portfolio_value=100000, n_simulations=1000
        )
        assert result.var > 0
        assert result.method == "monte_carlo"

    def test_var_99_higher_than_95(self, sample_returns):
        """99% VaR should be higher than 95% VaR."""
        analyzer = RiskAnalyzer()
        var_95 = analyzer.historical_var(sample_returns, confidence=0.95, portfolio_value=100000)
        var_99 = analyzer.historical_var(sample_returns, confidence=0.99, portfolio_value=100000)
        assert var_99.var > var_95.var


class TestExcursionAnalysis:
    def test_mae_mfe_calculation(self):
        """MAE and MFE are calculated correctly."""
        excursion = TradeExcursion(
            entry_price=100.0,
            exit_price=103.0,
            max_favorable=108.0,  # Reached +8%
            max_adverse=97.0,  # Dipped -3%
            is_long=True,
        )
        assert excursion.mae == pytest.approx(3.0, rel=1e-5)  # 100 - 97
        assert excursion.mfe == pytest.approx(8.0, rel=1e-5)  # 108 - 100
        assert excursion.mae_pct == pytest.approx(0.03, rel=1e-5)
        assert excursion.mfe_pct == pytest.approx(0.08, rel=1e-5)

    def test_capture_ratio(self):
        """Capture ratio shows how much of MFE was realized."""
        # Trade went to +8% but only captured +3%
        trades = [
            TradeExcursion(
                entry_price=100.0,
                exit_price=103.0,
                max_favorable=108.0,
                max_adverse=99.0,
                is_long=True,
            )
        ]
        analysis = ExcursionAnalysis(trades=trades)
        # Realized 3% of 8% MFE = 37.5% capture
        assert analysis.capture_ratio == pytest.approx(0.375, rel=1e-2)


class TestRiskSummary:
    def test_risk_summary_keys(self):
        """Risk summary contains all expected keys."""
        analyzer = RiskAnalyzer()
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, 100)

        summary = analyzer.risk_summary(returns)
        expected_keys = [
            "var_95_pct", "var_95_dollar", "cvar_95_pct", "cvar_95_dollar",
            "var_99_pct", "var_99_dollar", "daily_volatility", "annualized_volatility",
            "worst_day", "best_day", "skewness", "kurtosis", "negative_days_pct",
        ]
        for key in expected_keys:
            assert key in summary
