"""BOJ meeting calendar with 48-hour entry blackout.

Bank of Japan (BOJ) meetings are the single highest-risk event for
all 6 JPY pairs simultaneously.  Surprise rate decisions (Jan 2016,
Jul 2024) caused 3–5% moves within minutes.  The generic ±2h Forex
Factory blackout is insufficient — BOJ meetings require a full ±24h
window around the decision day.

Primary source: hardcoded annual schedule (BOJ announces all dates for
the year in January).  Safety net: the Forex Factory live-calendar feed
is scanned for JPY events matching BOJ keywords within 7 days, catching
rescheduled or emergency meetings.

Update BOJ_MEETINGS each January when the new schedule is published.
The system sends a Telegram WARNING on startup if the current year is
not in the table.

Example:
    >>> cal = BOJCalendar()
    >>> blocked, reason = cal.is_boj_blackout(datetime.now(UTC))
    >>> if blocked:
    ...     print(f"BOJ blackout: {reason}")
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# Hardcoded BOJ decision dates — second day of each 2-day policy meeting,
# when the rate decision is announced (typically 03:00–04:00 UTC).
#
# Source: https://www.boj.or.jp/en/mopo/mpmsche_mprrev/mpmsche/index.htm
# Update each January when the BOJ publishes the new year's schedule.
# ---------------------------------------------------------------------------
BOJ_MEETINGS: dict[int, list[date]] = {
    2025: [
        date(2025, 1, 24),
        date(2025, 3, 19),
        date(2025, 5, 1),
        date(2025, 6, 17),
        date(2025, 7, 31),
        date(2025, 9, 19),
        date(2025, 10, 30),
        date(2025, 12, 19),
    ],
    2026: [
        date(2026, 1, 24),
        date(2026, 3, 19),
        date(2026, 5, 1),
        date(2026, 6, 17),
        date(2026, 7, 31),
        date(2026, 9, 18),
        date(2026, 10, 29),
        date(2026, 12, 18),
    ],
}

# Hours before AND after the decision day centre to block new entries.
# 24h before: market increasingly nervous, vol spikes ahead of announcement.
# 24h after:  market still digesting; gaps and stop-hunts common.
_BLACKOUT_HOURS: int = 24

# Decision announcements cluster around 03:00–04:00 UTC.  We use noon UTC
# as the "centre" of the decision day so ±24h covers the eve and the day
# after symmetrically regardless of the exact announcement time.
_DECISION_HOUR_UTC: int = 12

# Forex Factory event title keywords that indicate a BOJ rate decision.
# All lowercase — we compare against event.name.lower().
_BOJ_KEYWORDS: frozenset[str] = frozenset({
    "boj",
    "bank of japan",
    "policy rate",
    "monetary policy statement",
    "monetary policy decision",
    "interest rate decision",
    "uncollateralized overnight",
})


class BOJCalendar:
    """BOJ meeting blackout: hardcoded schedule + live-feed safety net.

    Args:
        live_calendar: Optional LiveCalendar instance used as a safety
            net.  When provided, its events list is scanned for BOJ
            keywords within 7 days so rescheduled or emergency meetings
            are caught even if not in BOJ_MEETINGS.
        blackout_hours: Hours before/after decision centre to block
            entries (default 24 → 48h total window).

    Example:
        >>> boj = BOJCalendar(live_calendar=runner.calendar)
        >>> boj.check_year_coverage(2026, alert_manager)
        >>> blocked, reason = boj.is_boj_blackout(datetime.now(UTC))
    """

    def __init__(
        self,
        live_calendar=None,
        blackout_hours: int = _BLACKOUT_HOURS,
    ) -> None:
        self._live_calendar = live_calendar
        self._blackout_hours = blackout_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_boj_blackout(self, now: datetime) -> tuple[bool, str]:
        """Check whether *now* falls inside a BOJ meeting blackout window.

        Checks the hardcoded schedule first (O(n) over 8 meetings/year).
        Falls back to the Forex Factory live feed as a safety net.
        Fail-open: if the live-feed check throws, entry is NOT blocked.

        Args:
            now: Timezone-aware UTC datetime.

        Returns:
            ``(True, reason)`` if inside a blackout window,
            ``(False, "")`` otherwise.
        """
        window = timedelta(hours=self._blackout_hours)
        year = now.year

        # Check this year plus adjacent years to handle year-boundary ticks.
        candidate_dates: list[date] = []
        for y in (year - 1, year, year + 1):
            candidate_dates.extend(BOJ_MEETINGS.get(y, []))

        for decision_date in candidate_dates:
            centre = datetime(
                decision_date.year,
                decision_date.month,
                decision_date.day,
                _DECISION_HOUR_UTC, 0, 0,
                tzinfo=UTC,
            )
            if centre - window <= now <= centre + window:
                hours_delta = (centre - now).total_seconds() / 3600
                direction = "before" if hours_delta > 0 else "after"
                return (
                    True,
                    f"BOJ meeting {decision_date} "
                    f"({abs(hours_delta):.0f}h {direction} decision)",
                )

        # Safety net: live Forex Factory feed
        return self._check_live_feed(now, window)

    def check_year_coverage(
        self,
        year: int,
        alert_manager=None,
    ) -> bool:
        """Warn if hardcoded BOJ schedule does not include *year*.

        Call once at startup.  Sends a Telegram WARNING if missing so
        the operator knows to update BOJ_MEETINGS before year-end.

        Args:
            year: Year to verify (typically ``datetime.now().year``).
            alert_manager: Optional AlertManager for Telegram alert.

        Returns:
            ``True`` if the year is covered, ``False`` if missing.
        """
        if year in BOJ_MEETINGS:
            logger.info(
                "BOJCalendar: %d covered — %d meetings hardcoded",
                year,
                len(BOJ_MEETINGS[year]),
            )
            return True

        msg = (
            f"BOJ calendar missing for {year}. "
            "Entries near BOJ meetings will NOT be blocked by the hardcoded "
            "schedule (live-feed safety net still active). "
            "Update BOJ_MEETINGS in src/news/boj_calendar.py."
        )
        logger.warning(msg)
        if alert_manager is not None:
            try:
                alert_manager.send(
                    "WARNING",
                    f"BOJ Calendar Gap: {year} not hardcoded",
                    msg,
                )
            except Exception as exc:
                logger.debug("Could not send BOJ coverage alert: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_live_feed(
        self, now: datetime, window: timedelta
    ) -> tuple[bool, str]:
        """Scan the live calendar for BOJ events within 7 days.

        Args:
            now: Current UTC datetime.
            window: Blackout window (timedelta).

        Returns:
            ``(True, reason)`` if a BOJ event is within the window,
            ``(False, "")`` otherwise.  Fail-open on any exception.
        """
        if self._live_calendar is None:
            return False, ""

        try:
            horizon = timedelta(days=7)
            for event in self._live_calendar.events:
                if event.currency != "JPY":
                    continue
                name_lower = event.name.lower()
                if not any(kw in name_lower for kw in _BOJ_KEYWORDS):
                    continue
                event_dt = event.datetime_utc
                # Only consider events within the 7-day scanning horizon
                if abs((event_dt - now).total_seconds()) > horizon.total_seconds():
                    continue
                if event_dt - window <= now <= event_dt + window:
                    logger.info(
                        "BOJ live-feed safety net triggered: %s (%s)",
                        event.name,
                        event.date,
                    )
                    return (
                        True,
                        f"BOJ event (live feed): {event.name} on {event.date}",
                    )
        except Exception as exc:
            logger.debug(
                "BOJ live-feed safety net failed (fail-open): %s", exc
            )

        return False, ""
