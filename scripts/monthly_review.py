"""Monthly review report for the carry trade algo.

Covers:
  1. Live vs backtest return comparison (past 30 days vs V3 baseline)
  2. Regime distribution (% time in each regime from daily_results)
  3. Swap income vs price PnL attribution per pair
  4. Bandit shadow stats (SKIP rate, blocked trades, counterfactual PnL)
  5. Correlation summary note (not persisted, must be observed real-time)

Run standalone:
    python scripts/monthly_review.py
    python scripts/monthly_review.py --month 2026-02
    python scripts/monthly_review.py --db /path/to/trading.db --month 2026-01
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UTC = timezone.utc

# V3 backtest baseline (JPY pairs, 2023-2024 hourly, annualized)
V3_BACKTEST_ANNUALIZED = 0.0518

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def month_date_range(month: str) -> tuple[str, str]:
    """Return (first_day, last_day) ISO strings for a YYYY-MM month string."""
    year, mon = int(month[:4]), int(month[5:7])
    _, last_day = monthrange(year, mon)
    first = f"{year:04d}-{mon:02d}-01"
    last = f"{year:04d}-{mon:02d}-{last_day:02d}"
    return first, last


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Live vs backtest return
# ─────────────────────────────────────────────────────────────────────────────


def get_monthly_return(conn: sqlite3.Connection, first: str, last: str) -> dict:
    """Compute realized return for the given date window from daily_results."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, starting_equity, ending_equity, daily_pnl, daily_return,
               trades_opened, trades_closed, max_drawdown_today,
               COALESCE(financing, 0) AS financing
        FROM daily_results
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """,
        (first, last),
    )
    rows = cur.fetchall()

    if not rows:
        return {
            "days": 0,
            "starting_equity": None,
            "ending_equity": None,
            "total_pnl": 0.0,
            "realized_return": None,
            "annualized_return": None,
            "max_drawdown": 0.0,
            "trades_opened": 0,
            "trades_closed": 0,
            "total_financing": 0.0,
            "rows": [],
        }

    starting_equity = rows[0]["starting_equity"]
    ending_equity = rows[-1]["ending_equity"]
    total_pnl = sum(r["daily_pnl"] for r in rows)
    total_financing = sum(r["financing"] for r in rows)
    max_drawdown = max(r["max_drawdown_today"] for r in rows)
    trades_opened = sum(r["trades_opened"] for r in rows)
    trades_closed = sum(r["trades_closed"] for r in rows)
    days = len(rows)

    realized_return = (
        (ending_equity - starting_equity) / starting_equity
        if starting_equity and starting_equity > 0
        else None
    )
    # Annualize: (1 + r) ^ (365 / days) - 1
    annualized_return = (
        ((1.0 + realized_return) ** (365.0 / days) - 1.0)
        if realized_return is not None and days > 0
        else None
    )

    return {
        "days": days,
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_pnl": total_pnl,
        "realized_return": realized_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "trades_opened": trades_opened,
        "trades_closed": trades_closed,
        "total_financing": total_financing,
        "rows": [dict(r) for r in rows],
    }


def section_return_comparison(data: dict, month: str) -> list[str]:
    lines = [
        "=" * 60,
        f"SECTION 1 — LIVE vs BACKTEST RETURN  [{month}]",
        "=" * 60,
    ]

    if data["days"] == 0:
        lines.append("  No daily_results rows found for this month.")
        return lines

    live_ann = data["annualized_return"]
    baseline = V3_BACKTEST_ANNUALIZED

    realized_str = (
        f"{data['realized_return']:+.3%}" if data["realized_return"] is not None else "N/A"
    )
    ann_str = f"{live_ann:+.3%}" if live_ann is not None else "N/A"

    if live_ann is not None:
        diff = live_ann - baseline
        diff_str = f"{diff:+.3%}"
        if live_ann >= baseline * 0.70:
            verdict = "PASS  (>= 70% of baseline)"
        elif live_ann >= 0:
            verdict = "WATCH (positive but below 70% of baseline)"
        else:
            verdict = "FAIL  (negative annualized return)"
    else:
        diff_str = "N/A"
        verdict = "N/A"

    lines += [
        f"  Period       : {month}  ({data['days']} trading days)",
        f"  Starting eq  : ${data['starting_equity']:,.2f}",
        f"  Ending eq    : ${data['ending_equity']:,.2f}",
        f"  Total PnL    : ${data['total_pnl']:+,.2f}",
        f"  Realized rtn : {realized_str}",
        f"  Annualized   : {ann_str}",
        f"  V3 baseline  : {baseline:+.3%} (annualized)",
        f"  Difference   : {diff_str}",
        f"  Verdict      : {verdict}",
        f"  Max drawdown : {data['max_drawdown']:.2%}",
        f"  Trades opened: {data['trades_opened']}",
        f"  Trades closed: {data['trades_closed']}",
        f"  Swap income  : ${data['total_financing']:+,.2f}",
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Regime distribution
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_REGIMES = [
    "TREND_LOW_VOL",
    "TREND_HIGH_VOL",
    "RANGE_LOW_VOL",
    "RANGE_HIGH_VOL",
]


def get_regime_distribution(
    conn: sqlite3.Connection, first: str, last: str
) -> dict[str, int]:
    """Count days per regime from daily_results in the given window."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT regime, COUNT(*) AS cnt
        FROM daily_results
        WHERE date >= ? AND date <= ?
        GROUP BY regime
        ORDER BY cnt DESC
        """,
        (first, last),
    )
    return {row["regime"]: row["cnt"] for row in cur.fetchall()}


def section_regime_distribution(dist: dict[str, int], month: str) -> list[str]:
    lines = [
        "",
        "=" * 60,
        f"SECTION 2 — REGIME DISTRIBUTION  [{month}]",
        "=" * 60,
    ]

    total = sum(dist.values())
    if total == 0:
        lines.append("  No regime data found for this month.")
        return lines

    # Build table with known regimes first, then any unknown ones
    all_regimes = KNOWN_REGIMES + [r for r in dist if r not in KNOWN_REGIMES]
    col_w = max(len(r) for r in all_regimes) + 2

    header = f"  {'Regime':<{col_w}}  {'Days':>5}  {'%':>7}  Bar"
    lines.append(header)
    lines.append("  " + "-" * (col_w + 22))

    for regime in all_regimes:
        count = dist.get(regime, 0)
        pct = count / total if total > 0 else 0
        bar_len = int(pct * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        lines.append(f"  {regime:<{col_w}}  {count:>5}  {pct:>6.1%}  [{bar}]")

    lines.append("  " + "-" * (col_w + 22))
    lines.append(f"  {'TOTAL':<{col_w}}  {total:>5}  {'100.0%':>7}")

    favorable = dist.get("TREND_LOW_VOL", 0) + dist.get("TREND_HIGH_VOL", 0)
    fav_pct = favorable / total if total > 0 else 0
    lines.append(f"\n  Favorable for trading (TREND_*): {favorable} days ({fav_pct:.1%})")
    lines.append(
        "  Note: regime stored per calendar day from daily_results.regime column."
    )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Swap income vs price PnL attribution per pair
# ─────────────────────────────────────────────────────────────────────────────


def get_trade_attribution(
    conn: sqlite3.Connection, first: str, last: str
) -> dict[str, dict]:
    """
    For each pair, split closed trades into:
      - swap_pnl  : sum of swap_earned
      - price_pnl : total pnl - swap_pnl
      - total_pnl : sum of pnl
      - count     : number of closed trades
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pair,
               COUNT(*)                          AS cnt,
               COALESCE(SUM(pnl), 0)             AS total_pnl,
               COALESCE(SUM(swap_earned), 0)     AS swap_pnl
        FROM trade_log
        WHERE status = 'closed'
          AND timestamp >= ?
          AND timestamp <  ?
        GROUP BY pair
        ORDER BY pair
        """,
        # timestamp is ISO string; use the day-after-last to capture the full last day
        (first + "T00:00:00", last + "T23:59:59"),
    )
    result: dict[str, dict] = {}
    for row in cur.fetchall():
        pair = row["pair"]
        swap_pnl = row["swap_pnl"]
        total_pnl = row["total_pnl"]
        price_pnl = total_pnl - swap_pnl
        result[pair] = {
            "count": row["cnt"],
            "total_pnl": total_pnl,
            "swap_pnl": swap_pnl,
            "price_pnl": price_pnl,
        }
    return result


def section_swap_attribution(attr: dict[str, dict], month: str) -> list[str]:
    lines = [
        "",
        "=" * 60,
        f"SECTION 3 — SWAP vs PRICE PnL ATTRIBUTION  [{month}]",
        "=" * 60,
    ]

    if not attr:
        lines.append("  No closed trades found for this month.")
        return lines

    col_pair = max(len(p) for p in attr) + 2
    col_pair = max(col_pair, 10)

    header = (
        f"  {'Pair':<{col_pair}}  {'Trades':>7}  "
        f"{'Total PnL':>11}  {'Swap PnL':>10}  {'Price PnL':>10}  {'Swap%':>7}"
    )
    sep = "  " + "-" * (col_pair + 54)
    lines += [header, sep]

    total_total = 0.0
    total_swap = 0.0
    total_price = 0.0
    total_trades = 0

    for pair, d in sorted(attr.items()):
        swap_pct = (
            d["swap_pnl"] / abs(d["total_pnl"]) if d["total_pnl"] != 0 else 0.0
        )
        lines.append(
            f"  {pair:<{col_pair}}  {d['count']:>7}  "
            f"${d['total_pnl']:>+10,.2f}  ${d['swap_pnl']:>+9,.2f}  "
            f"${d['price_pnl']:>+9,.2f}  {swap_pct:>6.1%}"
        )
        total_total += d["total_pnl"]
        total_swap += d["swap_pnl"]
        total_price += d["price_pnl"]
        total_trades += d["count"]

    total_swap_pct = (
        total_swap / abs(total_total) if total_total != 0 else 0.0
    )
    lines += [
        sep,
        f"  {'TOTAL':<{col_pair}}  {total_trades:>7}  "
        f"${total_total:>+10,.2f}  ${total_swap:>+9,.2f}  "
        f"${total_price:>+9,.2f}  {total_swap_pct:>6.1%}",
    ]
    lines.append(
        "\n  Swap% = swap_earned / |total_pnl|. "
        "Negative swap% indicates swap costs exceeded earnings."
    )
    lines.append(
        "  swap_earned is populated from trade_log.swap_earned "
        "(daily financing credited by OANDA)."
    )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Bandit shadow stats
# ─────────────────────────────────────────────────────────────────────────────


def get_bandit_stats(
    conn: sqlite3.Connection, first: str, last: str
) -> dict:
    """
    Parse trade_log.metadata JSON for bandit decision fields.

    Expected metadata keys (set by runner when bandit is active):
      bandit_decision : "TAKE" | "SKIP"
      bandit_score    : float (Thompson sample score)
      counterfactual_pnl : float (estimated PnL if SKIP was reversed)

    Returns aggregated stats dict.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT metadata, pnl, status
        FROM trade_log
        WHERE timestamp >= ?
          AND timestamp <= ?
        """,
        (first + "T00:00:00", last + "T23:59:59"),
    )
    rows = cur.fetchall()

    take_count = 0
    skip_count = 0
    take_pnl = 0.0
    counterfactual_pnl = 0.0
    has_bandit_data = False

    for row in rows:
        try:
            meta = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}

        decision = meta.get("bandit_decision")
        if decision is None:
            continue

        has_bandit_data = True

        if decision == "TAKE":
            take_count += 1
            if row["status"] == "closed" and row["pnl"] is not None:
                take_pnl += row["pnl"]
        elif decision == "SKIP":
            skip_count += 1
            cf = meta.get("counterfactual_pnl")
            if cf is not None:
                counterfactual_pnl += float(cf)

    total_signals = take_count + skip_count
    skip_rate = skip_count / total_signals if total_signals > 0 else 0.0

    return {
        "has_bandit_data": has_bandit_data,
        "total_signals": total_signals,
        "take_count": take_count,
        "skip_count": skip_count,
        "skip_rate": skip_rate,
        "take_pnl": take_pnl,
        "counterfactual_pnl": counterfactual_pnl,
        "estimated_saved": take_pnl - counterfactual_pnl,
    }


def section_bandit_stats(stats: dict, month: str) -> list[str]:
    lines = [
        "",
        "=" * 60,
        f"SECTION 4 — BANDIT SHADOW STATS  [{month}]",
        "=" * 60,
    ]

    if not stats["has_bandit_data"]:
        lines += [
            "  No bandit decision metadata found in trade_log for this month.",
            "  The bandit may still be in shadow mode (min_trades threshold not",
            "  yet reached) or metadata logging is not yet active.",
            "  Once active, trade_log.metadata will contain bandit_decision,",
            "  bandit_score, and counterfactual_pnl fields.",
        ]
        return lines

    lines += [
        f"  Total signals evaluated : {stats['total_signals']}",
        f"  TAKE decisions          : {stats['take_count']}",
        f"  SKIP decisions          : {stats['skip_count']}",
        f"  SKIP rate               : {stats['skip_rate']:.1%}",
        f"  PnL on TAKE trades      : ${stats['take_pnl']:+,.2f}",
        f"  Counterfactual PnL      : ${stats['counterfactual_pnl']:+,.2f}  "
        f"(est. PnL if SKIP signals were executed)",
        f"  Estimated value added   : ${stats['estimated_saved']:+,.2f}  "
        f"(TAKE PnL minus counterfactual)",
        "",
        "  Interpretation:",
        "  - Positive 'estimated value added' means bandit filtered",
        "    lower-quality signals successfully.",
        "  - Counterfactual PnL is an estimate stored at skip-time",
        "    by ShadowEvaluator; accuracy depends on shadow model quality.",
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Correlation summary note
# ─────────────────────────────────────────────────────────────────────────────


def section_correlation_note(month: str) -> list[str]:
    return [
        "",
        "=" * 60,
        f"SECTION 5 — CORRELATION SUMMARY  [{month}]",
        "=" * 60,
        "  NOTE: Live pair correlation data is NOT persisted to the database.",
        "  CorrelationMonitor (src/risk/correlation.py) maintains a rolling",
        "  correlation matrix in memory during the trading session, but this",
        "  data is not written to SQLite.",
        "",
        "  To observe correlation in real time:",
        "    - Check docker logs for 'correlation_factor' lines emitted by",
        "      DynamicSizer each time a new position is sized.",
        "    - Run /positions via Telegram bot for per-pair context.",
        "    - Run the correlation analysis script directly:",
        "        python scripts/run_risk_analysis.py",
        "      (uses historical data; for live correlation, inspect runner logs)",
        "",
        "  All 6 active pairs (USD/JPY, AUD/JPY, GBP/JPY, NZD/JPY, EUR/JPY,",
        "  CAD/JPY) are JPY-denominated and typically exhibit high correlation",
        "  during JPY trend events. DynamicSizer applies a portfolio_correlation",
        "  factor < 1.0 when rolling pairwise correlation exceeds threshold,",
        "  reducing position size to avoid over-concentration.",
        "",
        "  Action: If equity drawdown coincides with all-pairs simultaneous",
        "  losses, review logs for correlation_factor values and consider",
        "  tightening the correlation hard cap (see REAL_MONEY_READINESS.md).",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Daily breakdown table (bonus utility)
# ─────────────────────────────────────────────────────────────────────────────


def section_daily_breakdown(data: dict, month: str) -> list[str]:
    rows = data.get("rows", [])
    if not rows:
        return []

    lines = [
        "",
        "=" * 60,
        f"APPENDIX — DAILY BREAKDOWN  [{month}]",
        "=" * 60,
        f"  {'Date':<12}  {'PnL':>10}  {'Return':>8}  "
        f"{'Equity':>12}  {'Regime':<20}  {'CB':>3}",
        "  " + "-" * 72,
    ]

    for r in rows:
        cb = "YES" if r.get("circuit_breaker_triggered") else " no"
        regime = r.get("regime", "")[:20]
        lines.append(
            f"  {r['date']:<12}  ${r['daily_pnl']:>+9,.2f}  "
            f"{r['daily_return']:>+7.3%}  "
            f"${r['ending_equity']:>11,.2f}  "
            f"{regime:<20}  {cb:>3}"
        )

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly review report for the carry trade algo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/monthly_review.py\n"
            "  python scripts/monthly_review.py --month 2026-02\n"
            "  python scripts/monthly_review.py --db data/trading.db --month 2026-01"
        ),
    )
    parser.add_argument(
        "--db",
        default="data/trading.db",
        help="Path to the SQLite database (default: data/trading.db)",
    )
    parser.add_argument(
        "--month",
        default=datetime.now(UTC).strftime("%Y-%m"),
        help="Month to review in YYYY-MM format (default: current month)",
    )
    parser.add_argument(
        "--no-daily",
        action="store_true",
        help="Omit the per-day breakdown appendix",
    )
    args = parser.parse_args()

    # Validate month format
    try:
        datetime.strptime(args.month, "%Y-%m")
    except ValueError:
        print(f"ERROR: --month must be in YYYY-MM format, got: {args.month}", file=sys.stderr)
        sys.exit(1)

    db_path = args.db
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at: {db_path}", file=sys.stderr)
        print("  Pass --db /path/to/trading.db to specify the correct path.", file=sys.stderr)
        sys.exit(1)

    conn = get_db_connection(db_path)
    first, last = month_date_range(args.month)

    print()
    print("=" * 60)
    print(f"  MONTHLY REVIEW — {args.month}")
    print(f"  Database : {db_path}")
    print(f"  Window   : {first}  to  {last}")
    print(f"  Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Section 1
    monthly_data = get_monthly_return(conn, first, last)
    for line in section_return_comparison(monthly_data, args.month):
        print(line)

    # Section 2
    regime_dist = get_regime_distribution(conn, first, last)
    for line in section_regime_distribution(regime_dist, args.month):
        print(line)

    # Section 3
    attr = get_trade_attribution(conn, first, last)
    for line in section_swap_attribution(attr, args.month):
        print(line)

    # Section 4
    bandit = get_bandit_stats(conn, first, last)
    for line in section_bandit_stats(bandit, args.month):
        print(line)

    # Section 5
    for line in section_correlation_note(args.month):
        print(line)

    # Appendix
    if not args.no_daily:
        for line in section_daily_breakdown(monthly_data, args.month):
            print(line)

    print()
    print("=" * 60)
    print("  END OF MONTHLY REVIEW")
    print("=" * 60)
    print()

    conn.close()


if __name__ == "__main__":
    main()
