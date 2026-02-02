"""Tests for the ParamStore."""

import os
import tempfile

import pytest

from src.adaptive.param_store import ParamStore, ParamSet
from src.regime.adapters import RegimeAdaptedParams


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_params.db")
    return ParamStore(db_path)


@pytest.fixture
def sample_params():
    return RegimeAdaptedParams(
        stop_loss_mult=0.8,
        position_size_mult=1.2,
        entry_threshold=0.5,
        should_trade=True,
        should_close_on_weakness=False,
    )


class TestParamStore:
    def test_save_and_retrieve(self, store, sample_params):
        pid = store.save_params("TREND_LOW_VOL", sample_params)
        assert pid > 0
        store.activate(pid, "TREND_LOW_VOL")
        active = store.get_active_params("TREND_LOW_VOL")
        assert active is not None
        assert active.params.stop_loss_mult == 0.8
        assert active.params.position_size_mult == 1.2

    def test_no_active_returns_none(self, store):
        result = store.get_active_params("TREND_LOW_VOL")
        assert result is None

    def test_record_performance(self, store, sample_params):
        pid = store.save_params("TREND_LOW_VOL", sample_params)
        store.record_performance(pid, sharpe=1.5, win_rate=0.6, n=50)
        store.activate(pid, "TREND_LOW_VOL")
        active = store.get_active_params("TREND_LOW_VOL")
        assert active.sharpe == 1.5
        assert active.win_rate == 0.6
        assert active.sample_size == 50

    def test_activate_deactivates_previous(self, store, sample_params):
        pid1 = store.save_params("TREND_LOW_VOL", sample_params)
        store.activate(pid1, "TREND_LOW_VOL")

        params2 = RegimeAdaptedParams(
            stop_loss_mult=1.0,
            position_size_mult=1.0,
            entry_threshold=0.6,
            should_trade=True,
        )
        pid2 = store.save_params("TREND_LOW_VOL", params2)
        store.activate(pid2, "TREND_LOW_VOL")

        active = store.get_active_params("TREND_LOW_VOL")
        assert active.id == pid2
        assert active.params.entry_threshold == 0.6

    def test_get_history(self, store, sample_params):
        for i in range(5):
            store.save_params("TREND_LOW_VOL", sample_params)
        history = store.get_history("TREND_LOW_VOL", limit=3)
        assert len(history) == 3
        assert history[0].id > history[1].id  # Newest first

    def test_different_regimes_independent(self, store, sample_params):
        pid1 = store.save_params("TREND_LOW_VOL", sample_params)
        store.activate(pid1, "TREND_LOW_VOL")

        pid2 = store.save_params("TREND_HIGH_VOL", sample_params)
        store.activate(pid2, "TREND_HIGH_VOL")

        active1 = store.get_active_params("TREND_LOW_VOL")
        active2 = store.get_active_params("TREND_HIGH_VOL")
        assert active1.id == pid1
        assert active2.id == pid2

    def test_paramset_fields(self, store, sample_params):
        pid = store.save_params("TREND_LOW_VOL", sample_params)
        store.activate(pid, "TREND_LOW_VOL")
        ps = store.get_active_params("TREND_LOW_VOL")
        assert isinstance(ps, ParamSet)
        assert ps.regime == "TREND_LOW_VOL"
        assert ps.is_active is True
        assert ps.created_at is not None

    def test_empty_history(self, store):
        history = store.get_history("NONEXISTENT")
        assert history == []
