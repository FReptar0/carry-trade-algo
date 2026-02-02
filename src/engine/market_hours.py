"""Forex market hours utility.

The forex market operates 24/5:
- Opens: Sunday 5:00 PM Eastern Time (22:00 UTC, or 21:00 UTC during DST)
- Closes: Friday 5:00 PM Eastern Time (22:00 UTC, or 21:00 UTC during DST)

Major sessions (all times UTC):
- Sydney:  21:00 - 06:00
- Tokyo:   00:00 - 09:00
- London:  07:00 - 16:00
- New York: 13:00 - 22:00

This module uses US Eastern Time rules for open/close boundaries,
properly handling DST transitions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Market open: Sunday 17:00 ET
OPEN_DOW = 6  # Sunday
OPEN_HOUR = 17

# Market close: Friday 17:00 ET
CLOSE_DOW = 4  # Friday
CLOSE_HOUR = 17


class ForexMarketHours:
    """Determines whether the forex market is currently open."""

    @staticmethod
    def is_market_open(utc_time: datetime) -> bool:
        """Check if forex market is open at the given UTC time.

        Args:
            utc_time: A datetime in UTC.

        Returns:
            True if market is open.
        """
        et_time = utc_time.astimezone(ET)
        dow = et_time.weekday()  # Mon=0 .. Sun=6
        hour = et_time.hour

        # Saturday: always closed
        if dow == 5:
            return False

        # Sunday: open only at or after 17:00 ET
        if dow == 6:
            return hour >= OPEN_HOUR

        # Friday: open only before 17:00 ET
        if dow == 4:
            return hour < CLOSE_HOUR

        # Mon-Thu: always open
        return True

    @staticmethod
    def next_open(utc_time: datetime) -> datetime:
        """Get the next market open time after the given UTC time.

        Args:
            utc_time: A datetime in UTC.

        Returns:
            Next market open as a UTC datetime.
        """
        if ForexMarketHours.is_market_open(utc_time):
            # Already open; find next close then next open after that
            close = ForexMarketHours.next_close(utc_time)
            return ForexMarketHours.next_open(close + timedelta(minutes=1))

        et_time = utc_time.astimezone(ET)
        dow = et_time.weekday()

        # Find days until next Sunday
        if dow == 6:
            # It's Sunday but before 17:00 — opens today
            days_ahead = 0
        else:
            # Advance to next Sunday
            days_ahead = (6 - dow) % 7
            if days_ahead == 0:
                days_ahead = 7

        open_et = et_time.replace(
            hour=OPEN_HOUR, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        return open_et.astimezone(UTC)

    @staticmethod
    def next_close(utc_time: datetime) -> datetime:
        """Get the next market close time after the given UTC time.

        Args:
            utc_time: A datetime in UTC.

        Returns:
            Next market close as a UTC datetime.
        """
        et_time = utc_time.astimezone(ET)
        dow = et_time.weekday()

        if not ForexMarketHours.is_market_open(utc_time):
            # Market is closed; find next open, then next close after that
            open_time = ForexMarketHours.next_open(utc_time)
            return ForexMarketHours.next_close(
                open_time + timedelta(minutes=1)
            )

        # Market is open — find the coming Friday 17:00 ET
        days_to_friday = (4 - dow) % 7
        if days_to_friday == 0 and et_time.hour >= CLOSE_HOUR:
            # Past Friday close (shouldn't happen if market is open)
            days_to_friday = 7

        close_et = et_time.replace(
            hour=CLOSE_HOUR, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_to_friday)
        return close_et.astimezone(UTC)
