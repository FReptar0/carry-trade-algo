"""Tests for the ScaleManager."""

import pytest

from src.risk.scaling import ScaleManager, ScaleLevel


class TestScaleManager:
    def test_should_add_on_favorable_move(self):
        mgr = ScaleManager(max_tranches=3, scale_in_atr=1.0)
        position = {"entry_price": 150.0, "units": 10000, "tranche_count": 1}
        assert mgr.should_add(position, current_price=151.0, atr=0.5) is True

    def test_should_not_add_without_move(self):
        mgr = ScaleManager(max_tranches=3, scale_in_atr=1.0)
        position = {"entry_price": 150.0, "units": 10000, "tranche_count": 1}
        assert mgr.should_add(position, current_price=150.1, atr=0.5) is False

    def test_should_not_add_at_max_tranches(self):
        mgr = ScaleManager(max_tranches=3, scale_in_atr=1.0)
        position = {"entry_price": 150.0, "units": 10000, "tranche_count": 3}
        assert mgr.should_add(position, current_price=155.0, atr=0.5) is False

    def test_should_reduce_at_profit_level(self):
        mgr = ScaleManager()
        position = {"entry_price": 150.0, "units": 10000, "levels_taken": 0}
        # Default first level is 2.0 * ATR
        assert mgr.should_reduce(position, current_price=151.2, atr=0.5) is True

    def test_should_not_reduce_before_level(self):
        mgr = ScaleManager()
        position = {"entry_price": 150.0, "units": 10000, "levels_taken": 0}
        assert mgr.should_reduce(position, current_price=150.5, atr=0.5) is False

    def test_should_not_reduce_all_levels_taken(self):
        mgr = ScaleManager()
        position = {"entry_price": 150.0, "units": 10000, "levels_taken": 2}
        assert mgr.should_reduce(position, current_price=155.0, atr=0.5) is False

    def test_tranche_units(self):
        mgr = ScaleManager(tranche_pct=0.33)
        units = mgr.tranche_units(10000)
        assert units == 3300

    def test_tranche_units_minimum(self):
        mgr = ScaleManager(tranche_pct=0.01)
        units = mgr.tranche_units(10)
        assert units >= 1

    def test_reduction_units(self):
        mgr = ScaleManager()
        units = mgr.reduction_units(10000, level_index=0)
        assert units == 3300  # 33% of 10000

    def test_reduction_units_invalid_level(self):
        mgr = ScaleManager()
        units = mgr.reduction_units(10000, level_index=99)
        assert units == 0

    def test_custom_profit_levels(self):
        levels = [
            ScaleLevel(atr_multiple=1.5, exit_fraction=0.25),
            ScaleLevel(atr_multiple=3.0, exit_fraction=0.50),
        ]
        mgr = ScaleManager(profit_levels=levels)
        position = {"entry_price": 150.0, "units": 10000, "levels_taken": 0}
        # 1.5 ATR at atr=0.5 = 0.75 profit needed
        assert mgr.should_reduce(position, current_price=150.8, atr=0.5) is True

    def test_zero_atr(self):
        mgr = ScaleManager()
        position = {"entry_price": 150.0, "units": 10000, "tranche_count": 1}
        assert mgr.should_add(position, current_price=155.0, atr=0) is False
        assert mgr.should_reduce(position, current_price=155.0, atr=0) is False
