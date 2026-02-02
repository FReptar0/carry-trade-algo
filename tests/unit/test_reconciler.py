"""Tests for the position Reconciler."""

from unittest.mock import MagicMock

import pytest

from src.ops.reconciler import Reconciler, ReconcileResult


@pytest.fixture
def reconciler():
    return Reconciler()


class TestReconciler:
    def test_clean_reconciliation(self, reconciler):
        internal = {
            "USD/JPY": {"entry_price": 150.0, "units": 10000},
        }
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 10000, "avg_price": 150.0},
        ]
        result = reconciler.reconcile(internal, broker)
        assert result.is_clean
        assert result.matched == ["USD/JPY"]

    def test_orphaned_broker_position(self, reconciler):
        internal = {}
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 10000, "avg_price": 150.0},
        ]
        result = reconciler.reconcile(internal, broker)
        assert not result.is_clean
        assert "USD/JPY" in result.orphaned_broker

    def test_orphaned_internal_position(self, reconciler):
        internal = {
            "USD/JPY": {"entry_price": 150.0, "units": 10000},
        }
        broker = []
        result = reconciler.reconcile(internal, broker)
        assert not result.is_clean
        assert "USD/JPY" in result.orphaned_internal

    def test_unit_mismatch(self, reconciler):
        internal = {
            "USD/JPY": {"entry_price": 150.0, "units": 10000},
        }
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 5000, "avg_price": 150.0},
        ]
        result = reconciler.reconcile(internal, broker)
        assert not result.is_clean
        assert len(result.mismatched_units) == 1
        assert result.mismatched_units[0]["internal_units"] == 10000
        assert result.mismatched_units[0]["broker_units"] == 5000

    def test_managed_pairs_filter(self, reconciler):
        internal = {}
        broker = [
            {"pair": "EUR/USD", "side": "BUY", "units": 5000, "avg_price": 1.10},
            {"pair": "USD/JPY", "side": "BUY", "units": 10000, "avg_price": 150.0},
        ]
        result = reconciler.reconcile(
            internal, broker, managed_pairs=["USD/JPY", "AUD/JPY"]
        )
        assert "USD/JPY" in result.orphaned_broker
        assert "EUR/USD" not in result.orphaned_broker

    def test_apply_fixes_removes_orphaned_internal(self, reconciler):
        internal = {
            "USD/JPY": {"entry_price": 150.0, "units": 10000},
        }
        result = ReconcileResult(orphaned_internal=["USD/JPY"])
        reconciler.apply_fixes(internal, result, [])
        assert "USD/JPY" not in internal

    def test_apply_fixes_adds_orphaned_broker(self, reconciler):
        internal = {}
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 10000, "avg_price": 150.0},
        ]
        result = ReconcileResult(orphaned_broker=["USD/JPY"])
        reconciler.apply_fixes(internal, result, broker)
        assert "USD/JPY" in internal
        assert internal["USD/JPY"]["units"] == 10000

    def test_apply_fixes_corrects_units(self, reconciler):
        internal = {
            "USD/JPY": {"entry_price": 150.0, "units": 10000},
        }
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 5000, "avg_price": 150.0},
        ]
        result = ReconcileResult(
            mismatched_units=[
                {"pair": "USD/JPY", "internal_units": 10000, "broker_units": 5000}
            ]
        )
        reconciler.apply_fixes(internal, result, broker)
        assert internal["USD/JPY"]["units"] == 5000

    def test_alert_on_mismatch(self):
        alert_mgr = MagicMock()
        rec = Reconciler(alert_manager=alert_mgr)
        internal = {}
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 10000, "avg_price": 150.0},
        ]
        rec.reconcile(internal, broker)
        alert_mgr.send.assert_called_once()
        call_args = alert_mgr.send.call_args
        assert call_args[0][0] == "HIGH"

    def test_empty_reconciliation(self, reconciler):
        result = reconciler.reconcile({}, [])
        assert result.is_clean
        assert result.matched == []

    def test_multiple_pairs(self, reconciler):
        internal = {
            "USD/JPY": {"entry_price": 150.0, "units": 10000},
            "AUD/JPY": {"entry_price": 95.0, "units": 5000},
        }
        broker = [
            {"pair": "USD/JPY", "side": "BUY", "units": 10000, "avg_price": 150.0},
            {"pair": "AUD/JPY", "side": "BUY", "units": 5000, "avg_price": 95.0},
        ]
        result = reconciler.reconcile(internal, broker)
        assert result.is_clean
        assert len(result.matched) == 2
