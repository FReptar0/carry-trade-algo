"""Tests for the LiveCalendar."""

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
            finnhub_api_key="test",
            fallback_file="nonexistent.json",
        )
        assert hasattr(cal, "is_blackout_period")
        assert hasattr(cal, "events")

    def test_fallback_on_missing_finnhub(self):
        cal = LiveCalendar(
            finnhub_api_key="test",
            fallback_file="nonexistent.json",
        )
        with patch.dict("sys.modules", {"finnhub": None}):
            result = cal.refresh()
            assert result == -1

    def test_refresh_with_mock_finnhub(self):
        cal = LiveCalendar(
            finnhub_api_key="test",
            fallback_file="nonexistent.json",
        )
        today = date.today()
        mock_client = MagicMock()
        mock_client.calendar_economic.return_value = {
            "economicCalendar": [
                {
                    "country": "US",
                    "impact": "high",
                    "event": "FOMC Rate Decision",
                    "date": today.isoformat(),
                    "time": "18:00",
                },
                {
                    "country": "JP",
                    "impact": "low",
                    "event": "Minor Report",
                    "date": today.isoformat(),
                    "time": "01:00",
                },
            ]
        }

        mock_finnhub = MagicMock()
        mock_finnhub.Client.return_value = mock_client

        with patch.dict("sys.modules", {"finnhub": mock_finnhub}):
            result = cal.refresh()
            assert result == 1  # Only high-impact event

    def test_country_to_currency_mapping(self):
        assert LiveCalendar._country_to_currency("US") == "USD"
        assert LiveCalendar._country_to_currency("JP") == "JPY"
        assert LiveCalendar._country_to_currency("AU") == "AUD"
        assert LiveCalendar._country_to_currency("EU") == "EUR"
        assert LiveCalendar._country_to_currency("XX") == "XX"

    def test_last_refresh_initially_none(self):
        cal = LiveCalendar(
            finnhub_api_key="test",
            fallback_file="nonexistent.json",
        )
        assert cal.last_refresh is None

    def test_blackout_with_manual_events(self):
        cal = LiveCalendar(
            finnhub_api_key="test",
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
            finnhub_api_key="test",
            fallback_file="nonexistent.json",
        )
        mock_finnhub = MagicMock()
        mock_finnhub.Client.return_value.calendar_economic.side_effect = (
            Exception("API error")
        )
        with patch.dict("sys.modules", {"finnhub": mock_finnhub}):
            result = cal.refresh()
            assert result == -1

    def test_filters_irrelevant_currencies(self):
        cal = LiveCalendar(
            finnhub_api_key="test",
            fallback_file="nonexistent.json",
        )
        today = date.today()
        mock_client = MagicMock()
        mock_client.calendar_economic.return_value = {
            "economicCalendar": [
                {
                    "country": "BR",
                    "impact": "high",
                    "event": "Brazil GDP",
                    "date": today.isoformat(),
                    "time": "12:00",
                },
            ]
        }
        mock_finnhub = MagicMock()
        mock_finnhub.Client.return_value = mock_client

        with patch.dict("sys.modules", {"finnhub": mock_finnhub}):
            result = cal.refresh()
            assert result == 0  # Filtered out
