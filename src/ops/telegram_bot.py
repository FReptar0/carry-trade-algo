"""Telegram command bot for querying live trading system state.

Runs a long-polling loop in a daemon thread, listening for commands:
- /status  — Full system status (positions, PnL, stops, protocol)
- /health  — Quick health check (uptime, last tick, connectivity)
- /trends  — MA trend analysis for all pairs (entry proximity)
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

import pandas as pd

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
            "trends": self._handle_trends,
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
            {"command": "trends", "description": "MA trend analysis & entry proximity"},
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
        lines: list[str] = []

        # Header with equity front and center
        acct = state.get("account", {})
        equity = acct.get("equity", 0)
        balance = acct.get("balance", 0)
        unrealized = acct.get("unrealized_pnl", 0)
        unr_emoji = "\U0001f7e2" if unrealized >= 0 else "\U0001f534"
        lines.append(f"\U0001f4b0 *Equity: ${equity:,.2f}*")
        lines.append(f"Balance ${balance:,.2f} | Open {unr_emoji} ${unrealized:+,.2f}")

        # Protocol bar
        proto = state.get("protocol", {})
        status = proto.get("status", "?").upper()
        day = proto.get("day", "?")
        duration = proto.get("duration", 30)
        dd = proto.get("drawdown", 0)
        status_emoji = (
            "\U0001f6a8"
            if status == "DEGRADED"
            else "\u2705"
            if status == "RUNNING"
            else "\u26a0\ufe0f"
        )
        lines.append(f"{status_emoji} Day {day}/{duration} {status} | DD {dd:.1%}")
        if proto.get("degraded_reason"):
            lines.append(f"   \u26a0\ufe0f {proto['degraded_reason']}")
        lines.append("")

        # Positions — compact view
        positions = state.get("positions", [])
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
        total_fin = sum(p.get("financing", 0) for p in positions)
        pnl_emoji = "\U0001f7e2" if total_pnl >= 0 else "\U0001f534"
        header = (
            f"\U0001f4ca *{len(positions)} Positions* {pnl_emoji} ${total_pnl:+,.2f}"
        )
        if total_fin != 0:
            header += f" | Carry ${total_fin:+,.2f}"
        lines.append(header)
        for pos in positions:
            pair = pos.get("pair", "???")
            pnl = pos.get("unrealized_pnl", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            stop = pos.get("stop_price")
            units = pos.get("units", 0)

            profit_pct = (
                ((current - entry) / entry * 100) if entry > 0 and current > 0 else 0
            )
            dot = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
            stop_str = f"SL {stop:.3f}" if stop else "no SL"
            vol_ratio = pos.get("vol_ratio")
            vol_warn = (
                f" \u26a0\ufe0f{vol_ratio:.1f}x"
                if vol_ratio and vol_ratio > 1.5
                else ""
            )
            lines.append(
                f"{dot} {pair} {units:,}u ${pnl:+,.2f} ({profit_pct:+.1f}%) {stop_str}{vol_warn}"
            )

        # Pending orders (if any)
        pending = state.get("pending_orders", {})
        if pending:
            lines.append("")
            lines.append(
                f"\u23f3 *{len(pending)} Pending Order"
                f"{'s' if len(pending) != 1 else ''}*"
            )
            for pair, p in pending.items():
                tick_count = p.get("tick_count", 0)
                lines.append(
                    f"  {pair} {p.get('units', 0):,}u "
                    f"@ {p.get('limit_price', 0):.3f} "
                    f"(tick {tick_count})"
                )

        lines.append("")

        # Performance row
        perf = state.get("performance", {})
        daily_pnl = perf.get("daily_pnl", 0)
        daily_emoji = "\U0001f7e2" if daily_pnl >= 0 else "\U0001f534"
        total_financing = perf.get("total_financing", 0)
        uptime = datetime.now(UTC) - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        perf_parts = [f"Today {daily_emoji} ${daily_pnl:+,.2f}"]
        if total_financing != 0:
            perf_parts.append(f"Carry ${total_financing:+,.2f}")
        perf_parts.append(f"Up {hours}h{minutes}m")
        lines.append(" | ".join(perf_parts))

        return _escape_markdown("\n".join(lines))

    def _handle_health(self, state: dict) -> str:
        """Build quick health check response.

        Args:
            state: Current system state.

        Returns:
            Formatted health message.
        """
        lines: list[str] = []

        # Header
        lines.append("\U0001f3e5 *HEALTH*")

        # Connectivity + market
        acct = state.get("account", {})
        connected = acct.get("equity", 0) > 0
        conn_str = (
            "\U0001f517 OANDA Connected"
            if connected
            else "\U0001f534 OANDA DISCONNECTED"
        )
        market_open = state.get("market_open", False)
        market_str = "Market OPEN" if market_open else "Market CLOSED"
        lines.append(f"{conn_str} | {market_str}")

        # Protocol status
        proto = state.get("protocol", {})
        status = proto.get("status", "?").upper()
        day = proto.get("day", "?")
        duration = proto.get("duration", 30)
        positions = state.get("positions", [])
        status_emoji = (
            "\U0001f6a8"
            if status == "DEGRADED"
            else "\u2705"
            if status == "RUNNING"
            else "\u26a0\ufe0f"
        )
        lines.append(
            f"{status_emoji} Day {day}/{duration} {status} | "
            f"{len(positions)} position{'s' if len(positions) != 1 else ''}"
        )

        # Last tick + uptime
        last_tick = state.get("last_tick")
        if last_tick:
            ago = datetime.now(UTC) - last_tick
            minutes_ago = int(ago.total_seconds() // 60)
            tick_str = f"Last tick {minutes_ago}m ago"
        else:
            tick_str = "Last tick unknown"

        uptime = datetime.now(UTC) - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        lines.append(f"\U0001f552 {tick_str} | Up {hours}h {minutes}m")

        return _escape_markdown("\n".join(lines))

    def _handle_positions(self, state: dict) -> str:
        """Build detailed position breakdown.

        Args:
            state: Current system state.

        Returns:
            Formatted positions message.
        """
        positions = state.get("positions", [])
        pending = state.get("pending_orders", {})

        if not positions and not pending:
            return "No open positions."

        lines: list[str] = []
        lines.append(f"\U0001f4cb *POSITIONS ({len(positions)})*")

        for pos in positions:
            pair = pos.get("pair", "???")
            units = pos.get("units", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            pnl = pos.get("unrealized_pnl", 0)
            stop = pos.get("stop_price")
            entry_time = pos.get("entry_time")
            tranche_count = pos.get("tranche_count", 1)
            financing = pos.get("financing", 0)

            profit_pct = (
                ((current - entry) / entry * 100) if entry > 0 and current > 0 else 0
            )
            price_pnl = pnl - financing
            dot = "\U0001f7e2" if pnl >= 0 else "\U0001f534"

            # Hold duration
            if entry_time and not isinstance(entry_time, str):
                delta = datetime.now(UTC) - entry_time
                hold = f"{delta.days}d {delta.seconds // 3600}h"
            else:
                hold = "N/A"

            # Risk from stop
            risk_pct = ((entry - stop) / entry * 100) if stop and entry > 0 else 0

            lines.append("")
            lines.append(f"{dot} *{pair}*")
            lines.append(
                f"{units:,}u \u00d7 {tranche_count} "
                f"tranche{'s' if tranche_count != 1 else ''} | Held {hold}"
            )
            lines.append(f"Entry {entry:.3f} \u2192 {current:.3f} ({profit_pct:+.1f}%)")
            pnl_line = f"PnL ${pnl:+,.2f}"
            if financing != 0:
                pnl_line += f" (price ${price_pnl:+,.2f}, carry ${financing:+,.2f})"
            lines.append(pnl_line)
            if stop:
                lines.append(f"SL {stop:.3f} (risk {risk_pct:.1f}%)")
            else:
                lines.append("SL none")

            # Volatility ratio indicator
            vol_ratio = pos.get("vol_ratio")
            original_units = pos.get("original_units")
            if vol_ratio is not None:
                if vol_ratio > 1.5:
                    vol_str = f"\u26a0\ufe0f Vol {vol_ratio:.1f}x"
                else:
                    vol_str = f"Vol {vol_ratio:.1f}x"
                if original_units and original_units != units:
                    vol_str += f" | Orig {original_units:,}u"
                lines.append(vol_str)

        # Pending orders
        if pending:
            lines.append("")
            for pair, p in pending.items():
                tick_count = p.get("tick_count", 0)
                lines.append(f"\u23f3 *Pending: {pair}*")
                lines.append(
                    f"{p.get('units', 0):,}u "
                    f"@ {p.get('limit_price', 0):.3f} "
                    f"(tick {tick_count})"
                )

        # Footer totals
        if positions:
            lines.append("")
            lines.append("\u2500" * 14)
            total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            total_fin = sum(p.get("financing", 0) for p in positions)
            pnl_emoji = "\U0001f7e2" if total_pnl >= 0 else "\U0001f534"
            footer = f"Total {pnl_emoji} ${total_pnl:+,.2f}"
            if total_fin != 0:
                footer += f" | Carry ${total_fin:+,.2f}"
            lines.append(footer)

        return _escape_markdown("\n".join(lines))

    def _handle_trends(self, state: dict) -> str:
        """Build MA trend analysis for all pairs.

        Shows 50MA vs 200MA gap, price vs 50MA, and how close each
        pair is to triggering a V3 entry (golden cross).

        Args:
            state: Current system state including candle cache.

        Returns:
            Formatted trends message.
        """
        candle_cache: dict[str, pd.DataFrame] = state.get("candle_cache", {})
        pairs: list[str] = state.get("pairs", [])

        if not candle_cache:
            return "No candle data yet. Wait for next strategy tick."

        lines: list[str] = []

        # Header
        acct = state.get("account", {})
        equity = acct.get("equity", 0)
        proto = state.get("protocol", {})
        day = proto.get("day", "?")
        duration = proto.get("duration", 34)
        positions = state.get("positions", [])
        lines.append(f"\U0001f4c8 *TREND REPORT*")
        lines.append(
            f"Day {day}/{duration} | ${equity:,.0f} | "
            f"{len(positions)} pos"
        )
        lines.append("")

        # Per-pair analysis
        pair_data: list[tuple[str, str, float]] = []
        for pair in pairs:
            oanda_pair = pair.replace("/", "_")
            df = candle_cache.get(oanda_pair) or candle_cache.get(pair)
            if df is None or len(df) < 200:
                lines.append(f"{pair}: insufficient data")
                continue

            price = float(df["close"].iloc[-1])
            ma50 = float(df["close"].rolling(50).mean().iloc[-1])
            ma200 = float(df["close"].rolling(200).mean().iloc[-1])

            p_vs_50 = (price - ma50) / ma50 * 100
            ma50_vs_200 = (ma50 - ma200) / ma200 * 100

            if ma50 > ma200 and price > ma50:
                trend = "UPTREND"
                icon = "\U0001f7e2"
            elif ma50 > ma200:
                trend = "PULLBACK"
                icon = "\U0001f7e1"
            elif price > ma200:
                trend = "RECOVERING"
                icon = "\U0001f7e1"
            else:
                trend = "DOWNTREND"
                icon = "\U0001f534"

            gap_pct = (
                (ma200 - ma50) / ma200 * 100 if ma50 < ma200 else 0
            )

            lines.append(f"{icon} *{pair}* {trend}")
            lines.append(
                f"  Price {price:.3f} | 50MA {ma50:.3f} | 200MA {ma200:.3f}"
            )
            detail = f"  Pvs50 {p_vs_50:+.2f}% | 50vs200 {ma50_vs_200:+.2f}%"
            if gap_pct > 0:
                detail += f" | Gap {gap_pct:.2f}%"
            lines.append(detail)
            lines.append("")

            pair_data.append((pair, trend, gap_pct))

        # Summary
        uptrends = [p for p, t, _ in pair_data if t == "UPTREND"]
        closest = min(pair_data, key=lambda x: x[2] if x[2] > 0 else 999, default=None)

        lines.append("\U0001f3af *SUMMARY*")
        if uptrends:
            lines.append(f"Ready to trade: {', '.join(uptrends)}")
        else:
            lines.append("No pairs in uptrend yet")
        if closest and closest[2] > 0:
            lines.append(
                f"Closest to entry: {closest[0]} ({closest[2]:.2f}% gap)"
            )

        return _escape_markdown("\n".join(lines))

    def _handle_help(self, state: dict) -> str:
        """Build help message listing all commands.

        Args:
            state: Unused, included for handler signature consistency.

        Returns:
            Formatted help message.
        """
        lines = [
            "\u2753 *COMMANDS*",
            "",
            "\U0001f4ca /status \u2014 Full system overview",
            "\U0001f3e5 /health \u2014 Quick health check",
            "\U0001f4cb /positions \u2014 Detailed per-pair breakdown",
            "\U0001f4c8 /trends \u2014 MA trend analysis & entry proximity",
            "\u2753 /help \u2014 This message",
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
