"""Tests for the LiveCalendar."""

import json
from datetime import date, datetime, time, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.news.calendar import EconomicEvent
from src.news.live_calendar import LiveCalendar

UTC = ZoneInfo("UTC")


class TestLiveCalendar:
    def test_inherits_from_economic_calendar(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
        )
        assert hasattr(cal, "is_blackout_period")
        assert hasattr(cal, "events")

    def test_fallback_on_network_error(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
        )
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = cal.refresh()
            assert result == -1

    def test_refresh_with_mock_feed(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
        )
        today = date.today()
        dt_str = f"{today.isoformat()}T18:00:00-05:00"
        feed_data = [
            {
                "country": "USD",
                "impact": "High",
                "title": "FOMC Rate Decision",
                "date": dt_str,
            },
            {
                "country": "JPY",
                "impact": "Low",
                "title": "Minor Report",
                "date": f"{today.isoformat()}T01:00:00-05:00",
            },
        ]
        response = MagicMock()
        response.read.return_value = json.dumps(feed_data).encode()
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=response):
            result = cal.refresh()
            assert result == 1  # Only high-impact event

    def test_last_refresh_initially_none(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
        )
        assert cal.last_refresh is None

    def test_blackout_with_manual_events(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
            blackout_hours=2,
        )
        now = datetime.now(UTC)
        cal.events = [
            EconomicEvent(
                date=now.date(),
                time_utc=now.time(),
                name="FOMC",
                currency="USD",
                impact="high",
            )
        ]
        in_blackout, event = cal.is_blackout_period(now)
        assert in_blackout is True
        assert event.name == "FOMC"

    def test_refresh_handles_api_error(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=Exception("API error"),
        ):
            result = cal.refresh()
            assert result == -1

    def test_filters_irrelevant_currencies(self):
        cal = LiveCalendar(
            fallback_file="nonexistent.json",
        )
        today = date.today()
        dt_str = f"{today.isoformat()}T12:00:00-05:00"
        feed_data = [
            {
                "country": "BRL",
                "impact": "High",
                "title": "Brazil GDP",
                "date": dt_str,
            },
        ]
        response = MagicMock()
        response.read.return_value = json.dumps(feed_data).encode()
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=response):
            result = cal.refresh()
            assert result == 0  # Filtered out
