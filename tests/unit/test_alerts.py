"""Tests for the AlertManager."""

import time
from unittest.mock import patch, MagicMock

import pytest

from src.ops.alerts import AlertConfig, AlertManager


@pytest.fixture
def config():
    return AlertConfig(
        telegram_bot_token="test-token",
        telegram_chat_id="12345",
        min_severity="WARNING",
        throttle_minutes=1,
    )


@pytest.fixture
def manager(config):
    return AlertManager(config)


class TestAlertManager:
    def test_filters_below_min_severity(self, manager):
        result = manager.send("INFO", "Test", "Low severity")
        assert result is False

    def test_sends_at_min_severity(self, manager):
        with patch.object(manager, "_send_telegram", return_value=True):
            result = manager.send("WARNING", "Test", "At threshold")
            assert result is True

    def test_sends_above_min_severity(self, manager):
        with patch.object(manager, "_send_telegram", return_value=True):
            result = manager.send("CRITICAL", "Test", "High severity")
            assert result is True

    def test_throttle_same_alert(self, manager):
        with patch.object(manager, "_send_telegram", return_value=True):
            assert manager.send("HIGH", "Same", "First") is True
            assert manager.send("HIGH", "Same", "Second") is False

    def test_different_alerts_not_throttled(self, manager):
        with patch.object(manager, "_send_telegram", return_value=True):
            assert manager.send("HIGH", "Alert A", "First") is True
            assert manager.send("HIGH", "Alert B", "Different") is True

    def test_throttle_expires(self, manager):
        manager.config.throttle_minutes = 0  # No throttle
        with patch.object(manager, "_send_telegram", return_value=True):
            assert manager.send("HIGH", "Test", "First") is True
            # Manually expire throttle
            manager._last_sent.clear()
            assert manager.send("HIGH", "Test", "Second") is True

    def test_send_daily_summary(self, manager):
        snapshot = {
            "equity": 100500.0,
            "daily_pnl": 500.0,
            "drawdown": 0.02,
            "positions_count": 1,
        }
        with patch.object(manager, "_send_telegram", return_value=True) as mock:
            result = manager.send_daily_summary(snapshot)
            assert result is True
            call_text = mock.call_args[0][0]
            assert "100,500.00" in call_text
            assert "500.00" in call_text

    def test_send_telegram_failure(self, manager):
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = manager._send_telegram("test")
            assert result is False

    def test_info_severity_with_info_min(self, config):
        config.min_severity = "INFO"
        mgr = AlertManager(config)
        with patch.object(mgr, "_send_telegram", return_value=True):
            assert mgr.send("INFO", "Test", "Info msg") is True
