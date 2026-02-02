"""Tests for economic calendar and blackout period detection."""

import json
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.news.calendar import EconomicCalendar, EconomicEvent

UTC = ZoneInfo("UTC")


@pytest.fixture
def sample_events_file(tmp_path):
    """Create a temporary events JSON file."""
    events = [
        {
            "date": "2026-03-18",
            "time_utc": "18:00",
            "name": "FOMC Rate Decision",
            "currency": "USD",
            "impact": "high",
        },
        {
            "date": "2026-03-06",
            "time_utc": "13:30",
            "name": "US Non-Farm Payrolls",
            "currency": "USD",
            "impact": "high",
        },
        {
            "date": "2026-03-19",
            "time_utc": "03:00",
            "name": "BOJ Rate Decision",
            "currency": "JPY",
            "impact": "high",
        },
    ]
    path = tmp_path / "test_events.json"
    path.write_text(json.dumps(events))
    return str(path)


class TestEconomicEvent:
    """Tests for the EconomicEvent dataclass."""

    def test_datetime_utc(self):
        event = EconomicEvent(
            date=date(2026, 3, 18),
            time_utc=time(18, 0),
            name="FOMC",
            currency="USD",
            impact="high",
        )
        dt = event.datetime_utc
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 18
        assert dt.hour == 18
        assert dt.tzinfo == UTC


class TestEconomicCalendar:
    """Tests for the EconomicCalendar class."""

    def test_load_events(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file)
        assert len(cal.events) == 3

    def test_events_sorted_chronologically(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file)
        dates = [e.datetime_utc for e in cal.events]
        assert dates == sorted(dates)

    def test_missing_file_returns_empty(self, tmp_path):
        cal = EconomicCalendar(tmp_path / "nonexistent.json")
        assert len(cal.events) == 0

    def test_blackout_during_event(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file, blackout_hours=2)
        # 1 hour before FOMC
        now = datetime(2026, 3, 18, 17, 0, tzinfo=UTC)
        is_blackout, event = cal.is_blackout_period(now)
        assert is_blackout is True
        assert event is not None
        assert event.name == "FOMC Rate Decision"

    def test_blackout_after_event(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file, blackout_hours=2)
        # 1 hour after FOMC
        now = datetime(2026, 3, 18, 19, 0, tzinfo=UTC)
        is_blackout, event = cal.is_blackout_period(now)
        assert is_blackout is True
        assert event.name == "FOMC Rate Decision"

    def test_no_blackout_outside_window(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file, blackout_hours=2)
        # 5 hours before FOMC
        now = datetime(2026, 3, 18, 13, 0, tzinfo=UTC)
        is_blackout, event = cal.is_blackout_period(now)
        assert is_blackout is False
        assert event is None

    def test_blackout_at_exact_event_time(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file, blackout_hours=2)
        now = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
        is_blackout, event = cal.is_blackout_period(now)
        assert is_blackout is True

    def test_events_today(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file)
        events = cal.events_today(date(2026, 3, 18))
        assert len(events) == 1
        assert events[0].name == "FOMC Rate Decision"

    def test_events_today_none(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file)
        events = cal.events_today(date(2026, 1, 1))
        assert len(events) == 0

    def test_next_event(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file)
        now = datetime(2026, 3, 7, 0, 0, tzinfo=UTC)
        event = cal.next_event(now)
        assert event is not None
        assert event.name == "FOMC Rate Decision"

    def test_next_event_none_past_all(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file)
        now = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
        event = cal.next_event(now)
        assert event is None

    def test_custom_blackout_hours(self, sample_events_file):
        cal = EconomicCalendar(sample_events_file, blackout_hours=4)
        # 3 hours before NFP (within 4-hour window)
        now = datetime(2026, 3, 6, 10, 30, tzinfo=UTC)
        is_blackout, event = cal.is_blackout_period(now)
        assert is_blackout is True
        assert event.name == "US Non-Farm Payrolls"

    def test_events_2026_file_loads(self):
        """Verify the actual events_2026.json file loads correctly."""
        events_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "news"
            / "events_2026.json"
        )
        if events_path.exists():
            cal = EconomicCalendar(events_path)
            assert len(cal.events) > 40
            # Check all events are in 2026
            for event in cal.events:
                assert event.date.year == 2026
