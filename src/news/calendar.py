"""Economic event calendar with blackout period detection.

Loads a static JSON file of known high-impact economic events
(FOMC, NFP, CPI, BOJ, ECB, GDP releases) and checks whether
the current time falls within a configurable blackout window
around any event.

During blackout periods, the trading runner should skip opening
new positions to avoid volatility spikes around major releases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


@dataclass
class EconomicEvent:
    """A single high-impact economic event."""

    date: date
    time_utc: time
    name: str
    currency: str
    impact: str  # "high" or "medium"

    @property
    def datetime_utc(self) -> datetime:
        """Full datetime in UTC."""
        return datetime.combine(self.date, self.time_utc, tzinfo=UTC)


class EconomicCalendar:
    """Checks for economic event blackout periods.

    Args:
        events_file: Path to JSON file with event data.
        blackout_hours: Hours before and after an event to
            consider a blackout period.

    Example:
        >>> cal = EconomicCalendar("src/news/events_2026.json")
        >>> in_blackout, event = cal.is_blackout_period(datetime.now(UTC))
        >>> if in_blackout:
        ...     print(f"Blackout: {event.name}")
    """

    def __init__(
        self,
        events_file: str | Path,
        blackout_hours: int = 2,
    ) -> None:
        self.blackout_hours = blackout_hours
        self.events: list[EconomicEvent] = []
        self._load_events(events_file)

    def _load_events(self, events_file: str | Path) -> None:
        """Load events from JSON file."""
        path = Path(events_file)
        if not path.exists():
            return

        with open(path) as f:
            raw = json.load(f)

        for entry in raw:
            parts = entry["time_utc"].split(":")
            t = time(int(parts[0]), int(parts[1]))
            self.events.append(
                EconomicEvent(
                    date=date.fromisoformat(entry["date"]),
                    time_utc=t,
                    name=entry["name"],
                    currency=entry["currency"],
                    impact=entry.get("impact", "high"),
                )
            )

        # Sort chronologically
        self.events.sort(key=lambda e: e.datetime_utc)

    def is_blackout_period(
        self,
        now: datetime,
    ) -> tuple[bool, Optional[EconomicEvent]]:
        """Check if the given time is within a blackout period.

        Args:
            now: Current UTC datetime.

        Returns:
            Tuple of (is_blackout, nearest_event_or_None).
        """
        window = timedelta(hours=self.blackout_hours)

        for event in self.events:
            event_dt = event.datetime_utc
            if event_dt - window <= now <= event_dt + window:
                return True, event

        return False, None

    def events_today(self, today: date) -> list[EconomicEvent]:
        """Get all events scheduled for a given date.

        Args:
            today: The date to query.

        Returns:
            List of events on that date.
        """
        return [e for e in self.events if e.date == today]

    def next_event(self, now: datetime) -> Optional[EconomicEvent]:
        """Get the next upcoming event after the given time.

        Args:
            now: Current UTC datetime.

        Returns:
            The next event, or None if no future events.
        """
        for event in self.events:
            if event.datetime_utc > now:
                return event
        return None
