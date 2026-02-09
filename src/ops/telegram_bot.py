"""Telegram command bot for querying live trading system state.

Runs a long-polling loop in a daemon thread, listening for commands:
- /status  — Full system status (positions, PnL, stops, protocol)
- /health  — Quick health check (uptime, last tick, connectivity)
- /help    — List available commands

Uses only urllib (no new dependencies). Thread-safe: reads runner
state but never mutates it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")

logger = logging.getLogger(__name__)

# Maximum message length for Telegram (4096 chars)
MAX_MESSAGE_LENGTH = 4096

# Long-polling timeout (seconds). Telegram holds the connection open
# for this long, then returns an empty update list if nothing arrived.
POLL_TIMEOUT = 30

# Delay between poll cycles on error (seconds)
ERROR_DELAY = 5


class TelegramCommandBot:
    """Listens for Telegram commands and replies with live system state.

    Runs in a daemon thread using Telegram's getUpdates long-polling.
    The bot only responds to messages from the configured chat_id
    (security: ignores messages from other users/groups).

    Args:
        bot_token: Telegram Bot API token.
        chat_id: Authorized chat ID (only responds to this chat).
        state_provider: Callable that returns current system state dict.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        state_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self._token = bot_token
        self._chat_id = str(chat_id)
        self._state_provider = state_provider
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_update_id = 0
        self._start_time = datetime.now(UTC)

        # Command registry: command_name -> handler method
        self._commands: dict[str, Callable[[dict], str]] = {
            "status": self._handle_status,
            "health": self._handle_health,
            "help": self._handle_help,
            "positions": self._handle_positions,
        }

    def start(self) -> None:
        """Start the polling thread. Non-blocking."""
        if self._running:
            return

        self._running = True
        self._start_time = datetime.now(UTC)
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="telegram-bot",
            daemon=True,
        )
        self._thread.start()
        self._register_commands()
        logger.info("Telegram command bot started (chat_id=%s)", self._chat_id)

    def stop(self) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=POLL_TIMEOUT + 5)
            self._thread = None
        logger.info("Telegram command bot stopped")

    def _register_commands(self) -> None:
        """Register bot commands with Telegram for menu display."""
        commands = [
            {
                "command": "status",
                "description": "Full system status (positions, PnL, protocol)",
            },
            {"command": "health", "description": "Quick health check"},
            {"command": "positions", "description": "Detailed position breakdown"},
            {"command": "help", "description": "List available commands"},
        ]
        try:
            self._api_call("setMyCommands", {"commands": commands})
        except Exception as e:
            logger.debug("Failed to register bot commands: %s", e)

    def _poll_loop(self) -> None:
        """Main polling loop — runs in a daemon thread."""
        logger.debug("Polling loop started")

        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._process_update(update)
            except Exception as e:
                logger.debug("Poll error: %s", e)
                time.sleep(ERROR_DELAY)

    def _get_updates(self) -> list[dict]:
        """Fetch new updates from Telegram using long-polling.

        Returns:
            List of update dicts from the Telegram API.
        """
        params = {
            "offset": self._last_update_id + 1,
            "timeout": POLL_TIMEOUT,
            "allowed_updates": ["message"],
        }
        result = self._api_call("getUpdates", params, timeout=POLL_TIMEOUT + 10)
        if result and result.get("ok"):
            return result.get("result", [])
        return []

    def _process_update(self, update: dict) -> None:
        """Process a single Telegram update.

        Args:
            update: Telegram update dict.
        """
        update_id = update.get("update_id", 0)
        if update_id > self._last_update_id:
            self._last_update_id = update_id

        message = update.get("message")
        if not message:
            return

        # Security: only respond to authorized chat
        chat = message.get("chat", {})
        msg_chat_id = str(chat.get("id", ""))
        if msg_chat_id != self._chat_id:
            logger.debug(
                "Ignoring message from unauthorized chat: %s",
                msg_chat_id,
            )
            return

        text = message.get("text", "").strip()
        if not text.startswith("/"):
            return

        # Parse command (strip leading / and any @bot_name suffix)
        parts = text[1:].split()
        command = parts[0].split("@")[0].lower() if parts else ""

        handler = self._commands.get(command)
        if handler is None:
            self._send_message("Unknown command. Use /help to see available commands.")
            return

        try:
            state = self._state_provider()
            response = handler(state)
            self._send_message(response)
        except Exception as e:
            logger.error("Command /%s failed: %s", command, e)
            self._send_message(f"Error processing /{command}: {e}")

    def _handle_status(self, state: dict) -> str:
        """Build full system status response.

        Args:
            state: Current system state from the runner.

        Returns:
            Formatted status message.
        """
        lines = ["*CARRY TRADE STATUS*", ""]

        # Account
        acct = state.get("account", {})
        equity = acct.get("equity", 0)
        balance = acct.get("balance", 0)
        unrealized = acct.get("unrealized_pnl", 0)
        lines.append("*Account*")
        lines.append(f"  Equity: ${equity:,.2f}")
        lines.append(f"  Balance: ${balance:,.2f}")
        lines.append(f"  Unrealized: ${unrealized:+,.2f}")
        lines.append("")

        # Protocol
        proto = state.get("protocol", {})
        lines.append("*Protocol*")
        lines.append(
            f"  Day {proto.get('day', '?')}/{proto.get('duration', 30)}"
            f" — {proto.get('status', '?').upper()}"
        )
        dd = proto.get("drawdown", 0)
        lines.append(f"  Drawdown: {dd:.2%}")
        if proto.get("degraded_reason"):
            lines.append(f"  Degraded: {proto['degraded_reason']}")
        lines.append("")

        # Positions summary
        positions = state.get("positions", [])
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
        total_fin = sum(p.get("financing", 0) for p in positions)
        lines.append(f"*Positions ({len(positions)})*  PnL: ${total_pnl:+,.2f}")
        if total_fin != 0:
            lines.append(f"  Carry income: ${total_fin:+,.2f}")
        for pos in positions:
            pair = pos.get("pair", "???")
            units = pos.get("units", 0)
            pnl = pos.get("unrealized_pnl", 0)
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_price")
            current = pos.get("current_price", 0)

            stop_str = f"SL={stop:.3f}" if stop else "no SL"
            profit_pct = (
                ((current - entry) / entry * 100) if entry > 0 and current > 0 else 0
            )
            emoji = "+" if pnl >= 0 else ""
            lines.append(f"  {pair}: {units:,}u @ {entry:.3f}")
            lines.append(
                f"    Now {current:.3f} ({emoji}{profit_pct:.2f}%)"
                f"  ${pnl:+,.2f}  {stop_str}"
            )
        lines.append("")

        # Performance
        perf = state.get("performance", {})
        daily_pnl = perf.get("daily_pnl", 0)
        hwm = perf.get("high_water_mark", 0)
        total_financing = perf.get("total_financing", 0)
        lines.append("*Performance*")
        lines.append(f"  Daily PnL: ${daily_pnl:+,.2f}")
        lines.append(f"  High Water: ${hwm:,.2f}")
        if total_financing != 0:
            lines.append(f"  Total Carry: ${total_financing:+,.2f}")

        uptime = datetime.now(UTC) - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        lines.append(f"  Uptime: {hours}h {minutes}m")

        return _escape_markdown("\n".join(lines))

    def _handle_health(self, state: dict) -> str:
        """Build quick health check response.

        Args:
            state: Current system state.

        Returns:
            Formatted health message.
        """
        lines = ["*HEALTH CHECK*", ""]

        acct = state.get("account", {})
        connected = acct.get("equity", 0) > 0
        lines.append(f"OANDA: {'Connected' if connected else 'DISCONNECTED'}")

        proto = state.get("protocol", {})
        status = proto.get("status", "unknown").upper()
        lines.append(f"Protocol: {status}")
        lines.append(f"Day: {proto.get('day', '?')}/{proto.get('duration', 30)}")

        positions = state.get("positions", [])
        lines.append(f"Positions: {len(positions)}")

        last_tick = state.get("last_tick")
        if last_tick:
            ago = datetime.now(UTC) - last_tick
            minutes_ago = int(ago.total_seconds() // 60)
            lines.append(f"Last tick: {minutes_ago}m ago")
        else:
            lines.append("Last tick: unknown")

        market_open = state.get("market_open", False)
        lines.append(f"Market: {'OPEN' if market_open else 'CLOSED'}")

        uptime = datetime.now(UTC) - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        lines.append(f"Uptime: {hours}h {minutes}m")

        return _escape_markdown("\n".join(lines))

    def _handle_positions(self, state: dict) -> str:
        """Build detailed position breakdown.

        Args:
            state: Current system state.

        Returns:
            Formatted positions message.
        """
        positions = state.get("positions", [])
        if not positions:
            return "No open positions."

        lines = ["*POSITION DETAILS*", ""]

        for pos in positions:
            pair = pos.get("pair", "???")
            units = pos.get("units", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            pnl = pos.get("unrealized_pnl", 0)
            stop = pos.get("stop_price")
            hwm = pos.get("high_water_mark", 0)
            entry_time = pos.get("entry_time")
            tranche_count = pos.get("tranche_count", 1)
            financing = pos.get("financing", 0)

            profit_pct = (
                ((current - entry) / entry * 100) if entry > 0 and current > 0 else 0
            )

            # Decompose PnL: price movement vs carry income
            price_pnl = pnl - financing

            # Hold duration
            if entry_time:
                if isinstance(entry_time, str):
                    hold = "N/A"
                else:
                    delta = datetime.now(UTC) - entry_time
                    hold = f"{delta.days}d {delta.seconds // 3600}h"
            else:
                hold = "N/A"

            # Risk from stop
            risk_pct = ((entry - stop) / entry * 100) if stop and entry > 0 else 0

            lines.append(f"*{pair}*")
            lines.append(f"  Units: {units:,} (tranches: {tranche_count})")
            lines.append(f"  Entry: {entry:.5f}")
            lines.append(f"  Current: {current:.5f} ({profit_pct:+.2f}%)")
            lines.append(
                f"  PnL: ${pnl:+,.2f}  (price: ${price_pnl:+,.2f}, carry: ${financing:+,.2f})"
            )
            if stop:
                lines.append(f"  Stop: {stop:.3f} (risk: {risk_pct:.2f}%)")
            else:
                lines.append("  Stop: NONE")
            lines.append(f"  HWM: {hwm:.5f}")
            lines.append(f"  Held: {hold}")
            lines.append("")

        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
        total_fin = sum(p.get("financing", 0) for p in positions)
        lines.append(f"*Total PnL: ${total_pnl:+,.2f}*")
        if total_fin != 0:
            lines.append(f"*Total Carry: ${total_fin:+,.2f}*")

        return _escape_markdown("\n".join(lines))

    def _handle_help(self, state: dict) -> str:
        """Build help message listing all commands.

        Args:
            state: Unused, included for handler signature consistency.

        Returns:
            Formatted help message.
        """
        lines = [
            "*Available Commands*",
            "",
            "/status — Full system status",
            "  Account, protocol, all positions, PnL",
            "",
            "/health — Quick health check",
            "  Connectivity, protocol, last tick, uptime",
            "",
            "/positions — Detailed position breakdown",
            "  Per-pair entry, stop, risk, duration",
            "",
            "/help — This message",
        ]
        return _escape_markdown("\n".join(lines))

    def _send_message(self, text: str) -> bool:
        """Send a message to the authorized chat.

        Args:
            text: Message text (Markdown V1 formatted).

        Returns:
            True if sent successfully.
        """
        # Truncate if too long
        if len(text) > MAX_MESSAGE_LENGTH:
            text = text[: MAX_MESSAGE_LENGTH - 20] + "\n\n... (truncated)"

        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            result = self._api_call("sendMessage", payload)
            return result is not None and result.get("ok", False)
        except Exception as e:
            logger.error("Failed to send bot response: %s", e)
            return False

    def _api_call(
        self,
        method: str,
        payload: dict | None = None,
        timeout: int = 15,
    ) -> Optional[dict]:
        """Make a Telegram Bot API call.

        Args:
            method: API method name (e.g., "getUpdates").
            payload: JSON payload.
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response, or None on failure.
        """
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.debug("Telegram API %s returned %d: %s", method, e.code, body)
            return None
        except Exception as e:
            logger.debug("Telegram API %s failed: %s", method, e)
            return None


def _escape_markdown(text: str) -> str:
    """Escape characters that break Telegram Markdown V1 parsing.

    Same logic as alerts.py _sanitize_markdown — escapes _, `, [
    but leaves * alone (we use it for bold).

    Args:
        text: Raw text.

    Returns:
        Text safe for Telegram Markdown V1.
    """
    for ch in ("_", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text
