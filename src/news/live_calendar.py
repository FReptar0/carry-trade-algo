"""Live economic calendar using Forex Factory JSON feed.

Fetches real-time economic event data from a free, no-auth JSON feed
(nfs.faireconomy.media). Falls back to the static JSON file if the
feed is unavailable. Only tracks high-impact events for currencies
relevant to the carry trade pairs (USD, JPY, AUD, EUR).
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.news.calendar import EconomicCalendar, EconomicEvent

UTC = ZoneInfo("UTC")
EST = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

RELEVANT_CURRENCIES = {"USD", "JPY", "AUD", "EUR", "GBP", "NZD", "CAD"}

CALENDAR_URL = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
)


class LiveCalendar(EconomicCalendar):
    """Economic calendar backed by Forex Factory feed with static fallback.

    Args:
        fallback_file: Path to static JSON events file.
        blackout_hours: Hours around events to consider blackout.

    Example:
        >>> cal = LiveCalendar("src/news/events_2026.json")
        >>> cal.refresh()  # Fetch upcoming events
        >>> in_blackout, event = cal.is_blackout_period(datetime.now(UTC))
    """

    def __init__(
        self,
        fallback_file: str = "src/news/events_2026.json",
        blackout_hours: int = 2,
    ) -> None:
        super().__init__(fallback_file, blackout_hours=blackout_hours)
        self._fallback_events = list(self.events)
        self._last_refresh: Optional[datetime] = None

    def refresh(self) -> int:
        """Fetch economic events for this week from Forex Factory feed.

        Returns:
            Number of high-impact events found, or -1 on failure.
        """
        try:
            req = urllib.request.Request(
                CALENDAR_URL,
                headers={"User-Agent": "carry-trade-bot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            new_events: list[EconomicEvent] = []
            for entry in raw:
                impact = entry.get("impact", "").lower()
                currency = entry.get("country", "").upper()

                if impact != "high":
                    continue
                if currency not in RELEVANT_CURRENCIES:
                    continue

                date_str = entry.get("date", "")
                if not date_str:
                    continue

                try:
                    # Dates are in EST (e.g. "2026-02-02T10:00:00-05:00")
                    dt_est = datetime.fromisoformat(date_str)
                    dt_utc = dt_est.astimezone(UTC)
                except (ValueError, TypeError):
                    continue

                new_events.append(
                    EconomicEvent(
                        date=dt_utc.date(),
                        time_utc=dt_utc.time().replace(tzinfo=None),
                        name=entry.get("title", "Unknown"),
                        currency=currency,
                        impact="high",
                    )
                )

            if new_events:
                # Merge with static events still in the future
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
                    "LiveCalendar refreshed: %d high-impact events",
                    len(new_events),
                )
                return len(new_events)

            logger.warning(
                "No high-impact events from feed, keeping fallback"
            )
            return 0

        except Exception as e:
            logger.error(
                "Failed to fetch calendar feed: %s. Using fallback.",
                e,
            )
            return -1

    @property
    def last_refresh(self) -> Optional[datetime]:
        """When the calendar was last refreshed from the feed."""
        return self._last_refresh
