"""Tests for the DynamicSizer."""

import pytest

from src.risk.dynamic_sizer import DynamicSizer


class TestDynamicSizer:
    def test_basic_sizing(self):
        sizer = DynamicSizer(base_units=10000)
        units = sizer.calculate_size(
            pair="USD/JPY",
            equity=100000,
            atr=0.5,
            price=150.0,
            win_rate=0.55,
            avg_win=200,
            avg_loss=100,
            regime_mult=1.0,
            correlation_factor=1.0,
        )
        assert 1000 <= units <= 100000

    def test_higher_regime_mult_increases_size(self):
        sizer = DynamicSizer(base_units=10000)
        kwargs = dict(
            pair="USD/JPY", equity=100000, atr=0.5, price=150.0,
            win_rate=0.55, avg_win=200, avg_loss=100,
            correlation_factor=1.0,
        )
        units_low = sizer.calculate_size(regime_mult=0.5, **kwargs)
        units_high = sizer.calculate_size(regime_mult=1.5, **kwargs)
        assert units_high >= units_low

    def test_correlation_reduces_size(self):
        sizer = DynamicSizer(base_units=10000)
        kwargs = dict(
            pair="USD/JPY", equity=100000, atr=0.5, price=150.0,
            win_rate=0.55, avg_win=200, avg_loss=100, regime_mult=1.0,
        )
        units_uncorr = sizer.calculate_size(correlation_factor=1.0, **kwargs)
        units_corr = sizer.calculate_size(correlation_factor=0.5, **kwargs)
        assert units_corr <= units_uncorr

    def test_respects_min_units(self):
        sizer = DynamicSizer(base_units=10000, min_units=5000)
        units = sizer.calculate_size(
            pair="USD/JPY", equity=1000, atr=5.0, price=150.0,
            win_rate=0.3, avg_win=50, avg_loss=100,
            regime_mult=0.1, correlation_factor=0.5,
        )
        assert units >= 5000

    def test_respects_max_units(self):
        sizer = DynamicSizer(base_units=10000, max_units=20000)
        units = sizer.calculate_size(
            pair="USD/JPY", equity=10000000, atr=0.01, price=150.0,
            win_rate=0.8, avg_win=1000, avg_loss=100,
            regime_mult=2.0, correlation_factor=1.0,
        )
        assert units <= 20000

    def test_kelly_fraction(self):
        f = DynamicSizer._kelly_fraction(0.55, 200, 100)
        assert f > 0
        assert f <= 0.25

    def test_kelly_zero_loss(self):
        f = DynamicSizer._kelly_fraction(0.5, 100, 0)
        assert f == 0.0

    def test_kelly_zero_win_rate(self):
        f = DynamicSizer._kelly_fraction(0.0, 100, 100)
        assert f == 0.0

    def test_kelly_losing_strategy(self):
        f = DynamicSizer._kelly_fraction(0.3, 100, 200)
        assert f == 0.0  # Negative Kelly clamped to 0

    def test_zero_atr_uses_base(self):
        sizer = DynamicSizer(base_units=10000)
        units = sizer.calculate_size(
            pair="USD/JPY", equity=100000, atr=0, price=150.0,
            win_rate=0.55, avg_win=200, avg_loss=100,
        )
        assert units >= sizer.min_units

    # --- Drawdown scalar tests ---

    def test_drawdown_scalar_zero_dd(self):
        sizer = DynamicSizer()
        assert sizer.drawdown_scalar(0.0) == 1.0

    def test_drawdown_scalar_max_dd(self):
        sizer = DynamicSizer(max_drawdown_pct=0.20, drawdown_floor=0.25)
        assert sizer.drawdown_scalar(0.20) == pytest.approx(0.25)

    def test_drawdown_scalar_mid_dd(self):
        sizer = DynamicSizer(max_drawdown_pct=0.20, drawdown_floor=0.25)
        # At 10% DD: 1 - (0.10/0.20)^2 = 1 - 0.25 = 0.75
        assert sizer.drawdown_scalar(0.10) == pytest.approx(0.75)

    def test_drawdown_scalar_over_max_clamps(self):
        sizer = DynamicSizer(max_drawdown_pct=0.20, drawdown_floor=0.25)
        # DD beyond max should clamp and return floor
        assert sizer.drawdown_scalar(0.30) == pytest.approx(0.25)

    def test_drawdown_reduces_position_size(self):
        sizer = DynamicSizer(base_units=10000)
        kwargs = dict(
            pair="USD/JPY", equity=100000, atr=0.5, price=150.0,
            win_rate=0.55, avg_win=200, avg_loss=100,
            regime_mult=1.0, correlation_factor=1.0,
        )
        units_no_dd = sizer.calculate_size(drawdown_pct=0.0, **kwargs)
        units_mid_dd = sizer.calculate_size(drawdown_pct=0.10, **kwargs)
        units_high_dd = sizer.calculate_size(drawdown_pct=0.18, **kwargs)
        assert units_no_dd >= units_mid_dd
        assert units_mid_dd >= units_high_dd

    def test_drawdown_zero_is_backward_compatible(self):
        sizer = DynamicSizer(base_units=10000)
        kwargs = dict(
            pair="USD/JPY", equity=100000, atr=0.5, price=150.0,
            win_rate=0.55, avg_win=200, avg_loss=100,
            regime_mult=1.0, correlation_factor=1.0,
        )
        # Default drawdown_pct=0.0 should give same result as explicit 0.0
        units_default = sizer.calculate_size(**kwargs)
        units_explicit = sizer.calculate_size(drawdown_pct=0.0, **kwargs)
        assert units_default == units_explicit
