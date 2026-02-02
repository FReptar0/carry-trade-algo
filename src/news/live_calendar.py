"""Live economic calendar using Finnhub API.

Replaces static JSON with real-time economic event data from Finnhub.
Falls back to the static JSON file if the API is unavailable.
Only tracks high-impact events for currencies relevant to the
carry trade pairs (USD, JPY, AUD, EUR).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.news.calendar import EconomicCalendar, EconomicEvent

UTC = ZoneInfo("UTC")
logger = logging.getLogger(__name__)

RELEVANT_CURRENCIES = {"USD", "JPY", "AUD", "EUR"}


class LiveCalendar(EconomicCalendar):
    """Economic calendar backed by Finnhub API with static fallback.

    Args:
        finnhub_api_key: API key for Finnhub.
        fallback_file: Path to static JSON events file.
        blackout_hours: Hours around events to consider blackout.

    Example:
        >>> cal = LiveCalendar("your-api-key", "src/news/events_2026.json")
        >>> cal.refresh()  # Fetch upcoming events
        >>> in_blackout, event = cal.is_blackout_period(datetime.now(UTC))
    """

    def __init__(
        self,
        finnhub_api_key: str,
        fallback_file: str = "src/news/events_2026.json",
        blackout_hours: int = 2,
    ) -> None:
        super().__init__(fallback_file, blackout_hours=blackout_hours)
        self.finnhub_api_key = finnhub_api_key
        self._fallback_events = list(self.events)
        self._last_refresh: Optional[datetime] = None

    def refresh(self) -> int:
        """Fetch economic events for the next 7 days from Finnhub.

        Returns:
            Number of high-impact events found, or -1 on failure.
        """
        try:
            import finnhub

            client = finnhub.Client(api_key=self.finnhub_api_key)

            from_date = date.today().isoformat()
            to_date = (date.today() + timedelta(days=7)).isoformat()

            raw = client.calendar_economic(
                _from=from_date, to=to_date
            )
            entries = raw.get("economicCalendar", [])

            new_events: list[EconomicEvent] = []
            for entry in entries:
                impact = entry.get("impact", "").lower()
                country = entry.get("country", "").upper()
                currency = self._country_to_currency(country)

                if impact != "high":
                    continue
                if currency not in RELEVANT_CURRENCIES:
                    continue

                event_date = entry.get("date", "")
                event_time = entry.get("time", "00:00")

                try:
                    d = date.fromisoformat(event_date)
                    parts = event_time.split(":")
                    t = time(int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    continue

                new_events.append(
                    EconomicEvent(
                        date=d,
                        time_utc=t,
                        name=entry.get("event", "Unknown"),
                        currency=currency,
                        impact="high",
                    )
                )

            if new_events:
                # Merge with existing events (keep static events
                # that are still in the future)
                now = datetime.now(UTC)
                existing_future = [
                    e
                    for e in self._fallback_events
                    if e.datetime_utc > now
                ]
                # De-duplicate by date+name
                seen = {
                    (e.date, e.name) for e in new_events
                }
                for e in existing_future:
                    if (e.date, e.name) not in seen:
                        new_events.append(e)

                self.events = sorted(
                    new_events, key=lambda e: e.datetime_utc
                )
                self._last_refresh = now
                logger.info(
                    "LiveCalendar refreshed: %d events",
                    len(new_events),
                )
                return len(new_events)

            logger.warning(
                "No high-impact events from Finnhub, "
                "keeping fallback"
            )
            return 0

        except ImportError:
            logger.warning(
                "finnhub-python not installed, using fallback calendar"
            )
            return -1
        except Exception as e:
            logger.error(
                "Failed to fetch Finnhub calendar: %s. "
                "Using fallback.",
                e,
            )
            return -1

    @property
    def last_refresh(self) -> Optional[datetime]:
        """When the calendar was last refreshed from API."""
        return self._last_refresh

    @staticmethod
    def _country_to_currency(country: str) -> str:
        """Map Finnhub country codes to currency codes."""
        mapping = {
            "US": "USD",
            "JP": "JPY",
            "AU": "AUD",
            "EU": "EUR",
            "DE": "EUR",
            "FR": "EUR",
            "IT": "EUR",
            "GB": "GBP",
            "CA": "CAD",
            "NZ": "NZD",
            "CH": "CHF",
        }
        return mapping.get(country, country)
