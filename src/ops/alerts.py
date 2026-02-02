"""Telegram-based alert manager for trading notifications.

Sends alerts via Telegram Bot API for key trading events:
- Trade opened/closed (INFO)
- Circuit breaker triggered (HIGH)
- Protocol checkpoint reached (WARNING)
- Position reconciliation mismatch (HIGH)
- Watchdog missed heartbeat (CRITICAL)
- System startup/shutdown (WARNING)

Includes throttling to prevent alert spam.
"""

from __future__ import annotations

import logging
import time
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_EMOJI = {"INFO": "ℹ️", "WARNING": "⚠️", "HIGH": "🔴", "CRITICAL": "🚨"}


@dataclass
class AlertConfig:
    """Configuration for alert delivery.

    Attributes:
        telegram_bot_token: Telegram Bot API token.
        telegram_chat_id: Telegram chat ID to send messages to.
        min_severity: Minimum severity to send (WARNING, HIGH, CRITICAL).
        throttle_minutes: Don't repeat the same alert within this window.
    """

    telegram_bot_token: str
    telegram_chat_id: str
    min_severity: str = "WARNING"
    throttle_minutes: int = 15


class AlertManager:
    """Sends trading alerts via Telegram Bot API.

    Args:
        config: AlertConfig with Telegram credentials and settings.

    Example:
        >>> mgr = AlertManager(AlertConfig(token="...", chat_id="..."))
        >>> mgr.send("HIGH", "Circuit Breaker", "Daily loss limit hit")
    """

    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self._last_sent: dict[str, float] = {}

    def send(self, severity: str, title: str, message: str) -> bool:
        """Send an alert if it meets severity and throttle requirements.

        Args:
            severity: One of INFO, WARNING, HIGH, CRITICAL.
            title: Short alert title.
            message: Alert body text.

        Returns:
            True if alert was sent, False if filtered/throttled.
        """
        if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get(
            self.config.min_severity, 1
        ):
            return False

        throttle_key = f"{severity}:{title}"
        now = time.time()
        last = self._last_sent.get(throttle_key, 0)
        if now - last < self.config.throttle_minutes * 60:
            logger.debug("Alert throttled: %s", throttle_key)
            return False

        emoji = SEVERITY_EMOJI.get(severity, "")
        text = f"{emoji} *{severity}: {title}*\n\n{message}"

        success = self._send_telegram(text)
        if success:
            self._last_sent[throttle_key] = now
        return success

    def send_daily_summary(self, snapshot: dict) -> bool:
        """Send a daily performance summary.

        Args:
            snapshot: Dict with equity, daily_pnl, drawdown,
                positions_count, etc.

        Returns:
            True if sent successfully.
        """
        lines = [
            "📊 *Daily Summary*",
            "",
            f"Equity: {snapshot.get('equity', 0):,.2f}",
            f"Daily P&L: {snapshot.get('daily_pnl', 0):,.2f}",
            f"Drawdown: {snapshot.get('drawdown', 0):.2%}",
            f"Positions: {snapshot.get('positions_count', 0)}",
        ]
        text = "\n".join(lines)
        return self._send_telegram(text)

    def _send_telegram(self, text: str) -> bool:
        """Send a message via Telegram Bot API.

        Args:
            text: Message text (Markdown formatted).

        Returns:
            True if the API call succeeded.
        """
        url = (
            f"https://api.telegram.org/bot{self.config.telegram_bot_token}"
            f"/sendMessage"
        )
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Alert sent: %s", text[:80])
                    return True
                logger.warning("Telegram API returned %d", resp.status)
                return False
        except Exception as e:
            logger.error("Failed to send Telegram alert: %s", e)
            return False
