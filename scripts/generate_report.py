"""Generate daily and weekly performance reports.

Can be run standalone or integrated into the trading runner.
Sends reports via Telegram if configured.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
from dataclasses import dataclass

UTC = ZoneInfo("UTC")


@dataclass
class DailyReport:
    date: str
    starting_equity: float
    ending_equity: float
    daily_pnl: float
    daily_return: float
    trades_opened: int
    trades_closed: int
    max_drawdown: float
    open_positions: int
    unrealized_pnl: float


@dataclass
class WeeklyReport:
    week_start: str
    week_end: str
    starting_equity: float
    ending_equity: float
    weekly_pnl: float
    weekly_return: float
    total_trades_opened: int
    total_trades_closed: int
    max_drawdown: float
    winning_days: int
    losing_days: int


def get_db_connection(db_path: str = "data/trading.db") -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def get_daily_report(conn: sqlite3.Connection, date: str = None) -> DailyReport | None:
    """Get report for a specific date (defaults to today)."""
    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")

    cur = conn.cursor()
    cur.execute(
        """SELECT date, starting_equity, ending_equity, daily_pnl, daily_return,
                  trades_opened, trades_closed, max_drawdown_today
           FROM daily_results WHERE date = ?""",
        (date,)
    )
    row = cur.fetchone()

    if not row:
        return None

    return DailyReport(
        date=row[0],
        starting_equity=row[1],
        ending_equity=row[2],
        daily_pnl=row[3],
        daily_return=row[4],
        trades_opened=row[5],
        trades_closed=row[6],
        max_drawdown=row[7],
        open_positions=0,
        unrealized_pnl=0,
    )


def get_weekly_report(conn: sqlite3.Connection) -> WeeklyReport | None:
    """Get report for the current week (Mon-Sun)."""
    today = datetime.now(UTC).date()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    cur = conn.cursor()
    cur.execute(
        """SELECT date, starting_equity, ending_equity, daily_pnl, daily_return,
                  trades_opened, trades_closed, max_drawdown_today
           FROM daily_results
           WHERE date >= ? AND date <= ?
           ORDER BY date""",
        (week_start.isoformat(), week_end.isoformat())
    )
    rows = cur.fetchall()

    if not rows:
        return None

    starting_equity = rows[0][1]
    ending_equity = rows[-1][2]
    weekly_pnl = sum(r[3] for r in rows)
    weekly_return = (ending_equity - starting_equity) / starting_equity if starting_equity > 0 else 0
    total_opened = sum(r[5] for r in rows)
    total_closed = sum(r[6] for r in rows)
    max_dd = max(r[7] for r in rows)
    winning_days = sum(1 for r in rows if r[3] > 0)
    losing_days = sum(1 for r in rows if r[3] < 0)

    return WeeklyReport(
        week_start=week_start.isoformat(),
        week_end=week_end.isoformat(),
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        weekly_pnl=weekly_pnl,
        weekly_return=weekly_return,
        total_trades_opened=total_opened,
        total_trades_closed=total_closed,
        max_drawdown=max_dd,
        winning_days=winning_days,
        losing_days=losing_days,
    )


def get_protocol_progress(conn: sqlite3.Connection) -> dict:
    """Get validation protocol progress."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_results")
    days_completed = cur.fetchone()[0]

    cur.execute(
        """SELECT SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END),
                  SUM(pnl), COUNT(*)
           FROM trade_log WHERE status = 'closed'"""
    )
    row = cur.fetchone()
    wins = row[0] or 0
    losses = row[1] or 0
    total_pnl = row[2] or 0
    total_trades = row[3] or 0

    cur.execute("SELECT COUNT(*) FROM trade_log WHERE status = 'open'")
    open_trades = cur.fetchone()[0]

    return {
        "days_completed": days_completed,
        "days_remaining": max(0, 30 - days_completed),
        "total_closed_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": wins / total_trades if total_trades > 0 else 0,
        "total_realized_pnl": total_pnl,
        "open_positions": open_trades,
    }


def get_current_positions(broker) -> list[dict]:
    """Get current open positions from broker."""
    return broker.get_all_positions() or []


def get_account_state(broker) -> dict:
    """Get current account state from broker."""
    return broker.get_account_state() or {}


def format_daily_report(report: DailyReport, positions: list, account: dict, progress: dict) -> str:
    """Format daily report for Telegram."""
    emoji = "🟢" if report.daily_pnl >= 0 else "🔴"

    lines = [
        f"📊 *Daily Report - {report.date}*",
        "",
        f"{emoji} *PnL:* ${report.daily_pnl:+,.2f} ({report.daily_return:+.3%})",
        f"💰 *Equity:* ${account.get('equity', report.ending_equity):,.2f}",
        f"📉 *Max Drawdown:* {report.max_drawdown:.2%}",
        "",
        f"📈 *Trades:* {report.trades_opened} opened, {report.trades_closed} closed",
        f"📍 *Open Positions:* {len(positions)}",
    ]

    if positions:
        lines.append("")
        for p in positions:
            pnl_emoji = "🟢" if p["unrealized_pnl"] >= 0 else "🔴"
            lines.append(f"  {pnl_emoji} {p['pair']}: {p['units']}u @ ${p['unrealized_pnl']:+.2f}")

    lines.extend([
        "",
        "─" * 25,
        f"📅 *Protocol:* Day {progress['days_completed']}/30",
        f"🎯 *Win Rate:* {progress['win_rate']:.0%} ({progress['winning_trades']}W/{progress['losing_trades']}L)",
        f"💵 *Total Realized:* ${progress['total_realized_pnl']:+,.2f}",
    ])

    return "\n".join(lines)


def format_weekly_report(report: WeeklyReport, progress: dict, account: dict) -> str:
    """Format weekly report for Telegram."""
    emoji = "🟢" if report.weekly_pnl >= 0 else "🔴"

    lines = [
        f"📈 *Weekly Report*",
        f"📅 {report.week_start} to {report.week_end}",
        "",
        f"{emoji} *Weekly PnL:* ${report.weekly_pnl:+,.2f} ({report.weekly_return:+.3%})",
        f"💰 *Current Equity:* ${account.get('equity', report.ending_equity):,.2f}",
        f"📉 *Max Drawdown:* {report.max_drawdown:.2%}",
        "",
        f"📊 *Days:* {report.winning_days} winning, {report.losing_days} losing",
        f"📈 *Trades:* {report.total_trades_opened} opened, {report.total_trades_closed} closed",
        "",
        "─" * 25,
        "*Protocol Progress*",
        f"📅 Day {progress['days_completed']}/30 ({progress['days_completed']/30*100:.0f}%)",
        f"🎯 Win Rate: {progress['win_rate']:.0%} ({progress['total_closed_trades']} trades)",
        f"💵 Total Realized: ${progress['total_realized_pnl']:+,.2f}",
    ]

    # Add milestone markers
    if progress['days_completed'] >= 7:
        lines.append("✅ Day 7 checkpoint passed")
    if progress['days_completed'] >= 14:
        lines.append("✅ Day 14 checkpoint passed")
    if progress['days_completed'] >= 21:
        lines.append("✅ Day 21 checkpoint passed")
    if progress['days_completed'] >= 30:
        lines.append("🏁 Protocol complete!")

    return "\n".join(lines)


def send_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """Send message via Telegram."""
    import requests

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, data=data, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def main():
    """Generate and send reports."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate trading reports")
    parser.add_argument("--type", choices=["daily", "weekly", "both"], default="daily")
    parser.add_argument("--db", default="data/trading.db", help="Database path")
    parser.add_argument("--no-telegram", action="store_true", help="Don't send to Telegram")
    args = parser.parse_args()

    # Get credentials
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # Connect to database
    conn = get_db_connection(args.db)
    progress = get_protocol_progress(conn)

    # Try to get broker data
    try:
        from src.broker.oanda import OandaBroker, OandaConfig
        config = OandaConfig(
            access_token=os.getenv("OANDA_ACCESS_TOKEN", ""),
            account_id=os.getenv("OANDA_ACCOUNT_ID", ""),
        )
        broker = OandaBroker(config)
        positions = get_current_positions(broker)
        account = get_account_state(broker)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
        positions = []
        account = {}

    messages = []

    if args.type in ("daily", "both"):
        report = get_daily_report(conn)
        if report:
            report.open_positions = len(positions)
            report.unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            msg = format_daily_report(report, positions, account, progress)
            messages.append(("Daily", msg))
            print(msg)
        else:
            print("No daily data available")

    if args.type in ("weekly", "both"):
        report = get_weekly_report(conn)
        if report:
            msg = format_weekly_report(report, progress, account)
            messages.append(("Weekly", msg))
            print(msg)
        else:
            print("No weekly data available")

    # Send to Telegram
    if not args.no_telegram and bot_token and chat_id:
        for report_type, msg in messages:
            if send_telegram(msg, bot_token, chat_id):
                print(f"\n✓ {report_type} report sent to Telegram")
            else:
                print(f"\n✗ Failed to send {report_type} report")

    conn.close()


if __name__ == "__main__":
    main()
