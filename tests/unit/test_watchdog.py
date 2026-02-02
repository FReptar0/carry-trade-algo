"""Tests for the Watchdog dead-man's-switch."""

import time
from unittest.mock import MagicMock

import pytest

from src.ops.watchdog import Watchdog


class TestWatchdog:
    def test_heartbeat_resets_timer(self):
        wd = Watchdog(expected_interval_sec=60)
        wd.heartbeat()
        assert wd.check() is True
        assert wd.miss_count == 0

    def test_check_passes_within_interval(self):
        wd = Watchdog(expected_interval_sec=3600)
        wd.heartbeat()
        assert wd.check() is True

    def test_check_fails_after_deadline(self):
        wd = Watchdog(expected_interval_sec=1)
        wd._last_heartbeat = time.time() - 10  # 10 seconds ago
        assert wd.check() is False
        assert wd.miss_count == 1

    def test_miss_count_increments(self):
        wd = Watchdog(expected_interval_sec=1)
        wd._last_heartbeat = time.time() - 10
        wd.check()
        wd.check()
        assert wd.miss_count == 2

    def test_heartbeat_resets_miss_count(self):
        wd = Watchdog(expected_interval_sec=1)
        wd._last_heartbeat = time.time() - 10
        wd.check()
        assert wd.miss_count == 1
        wd.heartbeat()
        assert wd.miss_count == 0

    def test_seconds_since_heartbeat(self):
        wd = Watchdog(expected_interval_sec=60)
        wd._last_heartbeat = time.time() - 5
        assert 4.5 < wd.seconds_since_heartbeat < 6.0

    def test_recovery_sends_alert(self):
        alert_mgr = MagicMock()
        wd = Watchdog(expected_interval_sec=1, alert_manager=alert_mgr)
        wd._last_heartbeat = time.time() - 10
        wd.check()
        alert_mgr.send.assert_called_once()
        call_args = alert_mgr.send.call_args
        assert call_args[0][0] == "CRITICAL"

    def test_recovery_without_alert_manager(self):
        wd = Watchdog(expected_interval_sec=1)
        wd._last_heartbeat = time.time() - 10
        # Should not raise
        wd.check()
        assert wd.miss_count == 1

    def test_initial_state_is_alive(self):
        wd = Watchdog(expected_interval_sec=3600)
        assert wd.check() is True
        assert wd.miss_count == 0
