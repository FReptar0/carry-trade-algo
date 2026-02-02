"""Tests for forex market hours utility."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.engine.market_hours import ForexMarketHours

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


class TestIsMarketOpen:
    """Tests for ForexMarketHours.is_market_open."""

    def test_monday_midday_open(self):
        """Market should be open on Monday midday."""
        # Monday 2026-02-02 12:00 UTC
        dt = datetime(2026, 2, 2, 12, 0, tzinfo=UTC)
        assert ForexMarketHours.is_market_open(dt) is True

    def test_wednesday_open(self):
        """Market should be open on Wednesday."""
        dt = datetime(2026, 2, 4, 15, 0, tzinfo=UTC)
        assert ForexMarketHours.is_market_open(dt) is True

    def test_saturday_closed(self):
        """Market should be closed on Saturday."""
        # Saturday 2026-02-07
        dt = datetime(2026, 2, 7, 12, 0, tzinfo=UTC)
        assert ForexMarketHours.is_market_open(dt) is False

    def test_sunday_early_closed(self):
        """Market should be closed Sunday morning ET."""
        # Sunday 2026-02-01 10:00 ET = 15:00 UTC
        dt = datetime(2026, 2, 1, 15, 0, tzinfo=UTC)
        assert ForexMarketHours.is_market_open(dt) is False

    def test_sunday_evening_open(self):
        """Market should open Sunday at 5pm ET."""
        # Sunday 2026-02-01 17:00 ET = 22:00 UTC (EST, no DST in Feb)
        dt = datetime(2026, 2, 1, 22, 0, tzinfo=UTC)
        assert ForexMarketHours.is_market_open(dt) is True

    def test_sunday_late_evening_open(self):
        """Market should be open Sunday 11pm ET."""
        # Sunday 2026-02-01 23:00 ET
        et_dt = datetime(2026, 2, 1, 23, 0, tzinfo=ET)
        utc_dt = et_dt.astimezone(UTC)
        assert ForexMarketHours.is_market_open(utc_dt) is True

    def test_friday_morning_open(self):
        """Market should be open Friday morning."""
        # Friday 2026-02-06 10:00 ET
        et_dt = datetime(2026, 2, 6, 10, 0, tzinfo=ET)
        utc_dt = et_dt.astimezone(UTC)
        assert ForexMarketHours.is_market_open(utc_dt) is True

    def test_friday_after_close(self):
        """Market should be closed Friday after 5pm ET."""
        # Friday 2026-02-06 18:00 ET
        et_dt = datetime(2026, 2, 6, 18, 0, tzinfo=ET)
        utc_dt = et_dt.astimezone(UTC)
        assert ForexMarketHours.is_market_open(utc_dt) is False

    def test_friday_at_close(self):
        """Market should be closed at exactly Friday 5pm ET."""
        et_dt = datetime(2026, 2, 6, 17, 0, tzinfo=ET)
        utc_dt = et_dt.astimezone(UTC)
        assert ForexMarketHours.is_market_open(utc_dt) is False

    def test_thursday_open(self):
        """Market should be open on Thursday."""
        dt = datetime(2026, 2, 5, 3, 0, tzinfo=UTC)
        assert ForexMarketHours.is_market_open(dt) is True


class TestNextOpen:
    """Tests for ForexMarketHours.next_open."""

    def test_next_open_from_saturday(self):
        """Next open from Saturday should be Sunday 5pm ET."""
        dt = datetime(2026, 2, 7, 12, 0, tzinfo=UTC)  # Saturday
        next_open = ForexMarketHours.next_open(dt)
        et_open = next_open.astimezone(ET)
        assert et_open.weekday() == 6  # Sunday
        assert et_open.hour == 17

    def test_next_open_from_sunday_morning(self):
        """Next open from Sunday morning should be same Sunday 5pm ET."""
        et_dt = datetime(2026, 2, 1, 10, 0, tzinfo=ET)  # Sunday 10am
        utc_dt = et_dt.astimezone(UTC)
        next_open = ForexMarketHours.next_open(utc_dt)
        et_open = next_open.astimezone(ET)
        assert et_open.weekday() == 6  # Sunday
        assert et_open.hour == 17
        assert et_open.day == 1  # Same day

    def test_next_open_from_weekday(self):
        """Next open from a weekday should be the following Sunday."""
        dt = datetime(2026, 2, 3, 12, 0, tzinfo=UTC)  # Tuesday
        next_open = ForexMarketHours.next_open(dt)
        et_open = next_open.astimezone(ET)
        assert et_open.weekday() == 6  # Sunday
        assert et_open.hour == 17


class TestNextClose:
    """Tests for ForexMarketHours.next_close."""

    def test_next_close_from_monday(self):
        """Next close from Monday should be Friday 5pm ET."""
        dt = datetime(2026, 2, 2, 12, 0, tzinfo=UTC)  # Monday
        next_close = ForexMarketHours.next_close(dt)
        et_close = next_close.astimezone(ET)
        assert et_close.weekday() == 4  # Friday
        assert et_close.hour == 17

    def test_next_close_from_thursday(self):
        """Next close from Thursday should be the next day (Friday) 5pm ET."""
        dt = datetime(2026, 2, 5, 12, 0, tzinfo=UTC)  # Thursday
        next_close = ForexMarketHours.next_close(dt)
        et_close = next_close.astimezone(ET)
        assert et_close.weekday() == 4  # Friday
        assert et_close.hour == 17

    def test_next_close_from_saturday(self):
        """Next close from Saturday should be the following Friday."""
        dt = datetime(2026, 2, 7, 12, 0, tzinfo=UTC)  # Saturday
        next_close = ForexMarketHours.next_close(dt)
        et_close = next_close.astimezone(ET)
        assert et_close.weekday() == 4  # Friday
        assert et_close.hour == 17
