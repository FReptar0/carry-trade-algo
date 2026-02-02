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
