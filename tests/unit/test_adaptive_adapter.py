"""Tests for the AdaptiveRegimeAdapter."""

import pytest

from src.adaptive.adaptive_adapter import AdaptiveRegimeAdapter
from src.adaptive.param_store import ParamStore
from src.regime.adapters import RegimeAdaptedParams
from src.regime.detector import CompositeRegime


@pytest.fixture
def store(tmp_path):
    return ParamStore(str(tmp_path / "test_adaptive.db"))


class TestAdaptiveRegimeAdapter:
    def test_falls_back_to_defaults(self, store):
        adapter = AdaptiveRegimeAdapter(store)
        params = adapter.adapt(CompositeRegime.TREND_LOW_VOL)
        assert params.stop_loss_mult == 0.8  # Default value
        assert params.should_trade is True

    def test_uses_stored_params(self, store):
        custom = RegimeAdaptedParams(
            stop_loss_mult=1.5,
            position_size_mult=0.9,
            entry_threshold=0.3,
            should_trade=True,
        )
        regime_val = CompositeRegime.TREND_LOW_VOL.value
        pid = store.save_params(regime_val, custom)
        store.activate(pid, regime_val)

        adapter = AdaptiveRegimeAdapter(store)
        params = adapter.adapt(CompositeRegime.TREND_LOW_VOL)
        assert params.stop_loss_mult == 1.5
        assert params.position_size_mult == 0.9

    def test_different_regimes(self, store):
        custom = RegimeAdaptedParams(
            stop_loss_mult=2.5,
            position_size_mult=0.5,
            entry_threshold=0.8,
            should_trade=True,
        )
        regime_val = CompositeRegime.TREND_HIGH_VOL.value
        pid = store.save_params(regime_val, custom)
        store.activate(pid, regime_val)

        adapter = AdaptiveRegimeAdapter(store)

        # Stored regime
        high_vol = adapter.adapt(CompositeRegime.TREND_HIGH_VOL)
        assert high_vol.stop_loss_mult == 2.5

        # Fallback regime
        low_vol = adapter.adapt(CompositeRegime.TREND_LOW_VOL)
        assert low_vol.stop_loss_mult == 0.8  # Default

    def test_inherits_from_regime_adapter(self, store):
        adapter = AdaptiveRegimeAdapter(store)
        assert hasattr(adapter, "should_enter_long")
        assert hasattr(adapter, "should_exit_early")
        assert hasattr(adapter, "get_adjusted_stop")

    def test_stored_params_override_fully(self, store):
        custom = RegimeAdaptedParams(
            stop_loss_mult=0.5,
            position_size_mult=2.0,
            entry_threshold=0.0,
            should_trade=True,
            should_close_on_weakness=True,
        )
        regime_val = CompositeRegime.RANGE_HIGH_VOL.value
        pid = store.save_params(regime_val, custom)
        store.activate(pid, regime_val)

        adapter = AdaptiveRegimeAdapter(store)
        params = adapter.adapt(CompositeRegime.RANGE_HIGH_VOL)
        assert params.should_close_on_weakness is True
        assert params.position_size_mult == 2.0
