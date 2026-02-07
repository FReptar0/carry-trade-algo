"""Dead-man's-switch watchdog for the trading system.

If no tick completes within 2x the expected interval, the watchdog
sends a CRITICAL alert and attempts recovery. Runs on its own
APScheduler job (every 5 minutes).

Market-aware: Does not alert when forex market is closed.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")


class Watchdog:
    """Monitors tick liveness and alerts on missed heartbeats.

    Args:
        expected_interval_sec: Expected seconds between heartbeats.
        alert_manager: AlertManager for sending CRITICAL alerts.
        market_open_check: Optional callable that returns True if market is open.

    Example:
        >>> wd = Watchdog(3600, alert_manager)  # expect hourly ticks
        >>> wd.heartbeat()  # called after each successful tick
        >>> wd.check()      # returns False if overdue
    """

    def __init__(
        self,
        expected_interval_sec: int,
        alert_manager: Optional[object] = None,
        market_open_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.expected_interval_sec = expected_interval_sec
        self.alert_manager = alert_manager
        self.market_open_check = market_open_check
        self._last_heartbeat: float = time.time()
        self._miss_count: int = 0
        self._market_was_open: bool = True

    def heartbeat(self) -> None:
        """Record a successful tick completion."""
        self._last_heartbeat = time.time()
        self._miss_count = 0
        logger.debug("Watchdog heartbeat recorded")

    def check(self) -> bool:
        """Check if the system is alive.

        Returns:
            True if within expected interval or market is closed, False if overdue.
        """
        # Check if market is open
        market_open = True
        if self.market_open_check is not None:
            try:
                market_open = self.market_open_check()
            except Exception as e:
                logger.debug("Market open check failed: %s", e)

        # If market is closed, don't alert - just log debug
        if not market_open:
            if self._market_was_open:
                logger.info("Watchdog: market closed, suspending alerts")
            self._market_was_open = False
            self._miss_count = 0  # Reset miss count when market closed
            return True

        # Market just reopened
        if not self._market_was_open and market_open:
            logger.info("Watchdog: market reopened, resuming monitoring")
            self._last_heartbeat = time.time()  # Reset heartbeat timer
            self._market_was_open = True
            return True

        self._market_was_open = True

        elapsed = time.time() - self._last_heartbeat
        deadline = self.expected_interval_sec * 2

        if elapsed > deadline:
            self._miss_count += 1
            logger.warning(
                "Watchdog: no heartbeat for %.0fs (limit %.0fs), "
                "miss_count=%d",
                elapsed,
                deadline,
                self._miss_count,
            )
            self._recovery_action()
            return False

        return True

    @property
    def seconds_since_heartbeat(self) -> float:
        """Seconds elapsed since last heartbeat."""
        return time.time() - self._last_heartbeat

    @property
    def miss_count(self) -> int:
        """Number of consecutive missed heartbeats."""
        return self._miss_count

    def _recovery_action(self) -> None:
        """Attempt recovery when heartbeat is missed."""
        if self.alert_manager is not None:
            try:
                self.alert_manager.send(
                    "CRITICAL",
                    "Watchdog Alert",
                    f"No tick heartbeat for "
                    f"{self.seconds_since_heartbeat:.0f}s. "
                    f"Consecutive misses: {self._miss_count}. "
                    f"System may be stuck.",
                )
            except Exception as e:
                logger.error("Watchdog alert failed: %s", e)

        logger.critical(
            "Watchdog recovery: system appears stuck "
            "(miss_count=%d)",
            self._miss_count,
        )
