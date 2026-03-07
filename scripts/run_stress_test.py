"""Multi-scenario stress test for V3 strategy.

Runs CarryTradeStrategyV3 across 5 historical market regimes using Yahoo
Finance daily data.  Each scenario gets ~10 months of warmup for the
200-day MA, followed by a 1-year evaluation window.

Scenarios:
  1. BOJ Negative Rates + Brexit         (eval 2016)
  2. Flash Crash + Trade War              (eval 2019)
  3. COVID Crash + Recovery               (eval 2020)
  4. Fed Hikes + BOJ Intervention         (eval 2022)
  5. Baseline (existing 2024 comparison)  (eval 2024)

Generates 5 charts in results/reports/stress_test/ plus a console
scorecard with PASS / CAUTION / FAIL grades.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import BacktestMetrics, calculate_metrics
from src.config.settings import RiskConfig, StrategyConfig
from src.data.loader import HistoricalDataLoader
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAIRS = ["USD/JPY", "AUD/JPY", "EUR/JPY", "GBP/JPY", "NZD/JPY", "CAD/JPY"]
TOTAL_CAPITAL = 100_000.0
OUTPUT_DIR = Path("results/reports/stress_test")


@dataclass
class Scenario:
    """Defines a historical stress-test scenario."""
    name: str
    data_start: str   # includes ~10-month warmup for 200MA
    data_end: str
    eval_start: str   # evaluation window start
    description: str


SCENARIOS = [
    Scenario(
        name="BOJ Negative Rates + Brexit",
        data_start="2015-01-01",
        data_end="2016-12-31",
        eval_start="2016-01-01",
        description="BOJ goes negative Jan '16, GBP/JPY crashes 20% on Brexit Jun '16",
    ),
    Scenario(
        name="Flash Crash + Trade War",
        data_start="2018-01-01",
        data_end="2019-12-31",
        eval_start="2019-01-01",
        description="Jan '19 JPY flash crash (4% in minutes), US-China trade war volatility",
    ),
    Scenario(
        name="COVID Crash + Recovery",
        data_start="2019-06-01",
        data_end="2020-12-31",
        eval_start="2020-01-01",
        description="Mar '20 crash (-12% in weeks), V-shaped recovery, emergency rate cuts",
    ),
    Scenario(
        name="Fed Hikes + BOJ Intervention",
        data_start="2021-06-01",
        data_end="2022-12-31",
        eval_start="2022-01-01",
        description="USD/JPY rallies 115->152 then BOJ intervenes, drops to 127 in weeks",
    ),
    Scenario(
        name="Baseline (2024)",
        data_start="2023-01-01",
        data_end="2024-12-31",
        eval_start="2024-01-01",
        description="Existing V3 backtest period -- included for comparison",
    ),
]

# Historical swap rate overrides (annual rate differential per era).
# The loader's _estimate_swaps uses 2022-2024 averages which are wrong
# for earlier periods.  Empty dict = use loader defaults.
SWAP_OVERRIDES: dict[str, dict[str, float]] = {
    "2016": {
        "USD/JPY": 0.005, "AUD/JPY": 0.02, "EUR/JPY": -0.005,
        "GBP/JPY": 0.005, "NZD/JPY": 0.025, "CAD/JPY": 0.005,
    },
    "2019": {
        "USD/JPY": 0.025, "AUD/JPY": 0.015, "EUR/JPY": -0.005,
        "GBP/JPY": 0.007, "NZD/JPY": 0.015, "CAD/JPY": 0.018,
    },
    "2020": {
        "USD/JPY": 0.005, "AUD/JPY": 0.002, "EUR/JPY": -0.005,
        "GBP/JPY": 0.001, "NZD/JPY": 0.002, "CAD/JPY": 0.002,
    },
    "2022": {
        "USD/JPY": 0.035, "AUD/JPY": 0.03, "EUR/JPY": 0.01,
        "GBP/JPY": 0.03, "NZD/JPY": 0.03, "CAD/JPY": 0.035,
    },
    "2024": {},  # Use loader defaults
}


# ---------------------------------------------------------------------------
# Helpers (copied from run_v3_backtest.py to stay self-contained)
# ---------------------------------------------------------------------------

def fix_swap_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Divide swap columns by 100 to undo the inherited synthetic scaling."""
    df = df.copy()
    df["swap_long"] = df["swap_long"] / 100.0
    df["swap_short"] = df["swap_short"] / 100.0
    return df


def apply_swap_override(df: pd.DataFrame, pair: str, year_key: str) -> pd.DataFrame:
    """Replace swap columns with era-appropriate rate differentials."""
    overrides = SWAP_OVERRIDES.get(year_key, {})
    if not overrides or pair not in overrides:
        return df
    diff = overrides[pair]
    df = df.copy()
    # Convert annual differential to daily swap value (same formula as loader)
    df["swap_long"] = round(diff / 365, 6)
    df["swap_short"] = round(-diff / 365, 6)
    return df


def reconstruct_equity(
    trades_df: pd.DataFrame,
    price_series: pd.Series,
    timestamps: pd.Series,
    initial_capital: float,
    swap_per_day: float,
) -> pd.DataFrame:
    """Build a smooth daily equity curve from trades + price data."""
    equity = initial_capital
    peak = initial_capital
    records: list[dict] = []

    trades: list[dict] = []
    if len(trades_df) > 0:
        for _, row in trades_df.iterrows():
            trades.append({
                "entry_time": pd.Timestamp(row["entry_time"]),
                "exit_time": pd.Timestamp(row["exit_time"]),
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "size": row["size"],
                "pnl": row["pnl"],
                "swap_earned": row["swap_earned"],
            })

    current_trade_idx = -1
    in_position = False
    pos_entry_price = 0.0
    pos_size = 0.0
    cash = initial_capital
    swap_accrued = 0.0

    for i in range(len(timestamps)):
        ts = pd.Timestamp(timestamps.iloc[i])
        price = price_series.iloc[i]

        if not in_position:
            for idx, t in enumerate(trades):
                if t["entry_time"].date() == ts.date():
                    in_position = True
                    current_trade_idx = idx
                    pos_entry_price = t["entry_price"]
                    pos_size = t["size"]
                    cash -= pos_entry_price * pos_size
                    swap_accrued = 0.0
                    break

        if in_position and current_trade_idx >= 0:
            t = trades[current_trade_idx]
            if t["exit_time"].date() == ts.date():
                cash += pos_entry_price * pos_size + t["pnl"] + t["swap_earned"]
                in_position = False
                current_trade_idx = -1

        if in_position:
            swap_accrued += swap_per_day * pos_size
            unrealized = (price - pos_entry_price) * pos_size + swap_accrued
            equity = cash + pos_entry_price * pos_size + unrealized
        else:
            equity = cash

        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        records.append({"timestamp": ts, "equity": equity, "drawdown": dd})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Per-scenario backtest runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Aggregated result for one scenario."""
    scenario: Scenario
    portfolio_curve: pd.DataFrame
    portfolio_metrics: BacktestMetrics
    pair_metrics: dict[str, BacktestMetrics]
    pairs_used: list[str]
    total_trades: int
    worst_event: str  # description of deepest drawdown period


def run_scenario(scenario: Scenario) -> ScenarioResult | None:
    """Run V3 backtest for a single scenario and return aggregated results."""
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario.name}")
    print(f"  Data: {scenario.data_start} -> {scenario.data_end}")
    print(f"  Eval: {scenario.eval_start} -> {scenario.data_end}")
    print(f"  {scenario.description}")
    print(f"{'='*70}")

    eval_start_ts = pd.Timestamp(scenario.eval_start)
    year_key = scenario.eval_start[:4]

    # 1. Load data
    loader = HistoricalDataLoader()
    data_map: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        try:
            df = loader.load_pair(
                pair,
                start=scenario.data_start,
                end=scenario.data_end,
                interval="1d",
            )
            if df.empty:
                print(f"  WARNING: no data for {pair} -- skipping")
                continue
            df = fix_swap_scale(df)
            df = apply_swap_override(df, pair, year_key)
            data_map[pair] = df
            print(f"  {pair}: {len(df)} daily bars")
        except Exception as e:
            print(f"  ERROR loading {pair}: {e}")

    if not data_map:
        print("  No data loaded -- skipping scenario.")
        return None

    # Adjust capital allocation based on available pairs
    per_pair_capital = TOTAL_CAPITAL / len(data_map)

    # 2. Run V3 per pair
    print("  Running V3 backtest ...")
    strategy = CarryTradeStrategyV3(StrategyConfigV3(position_size_pct=0.10))
    results: dict[str, BacktestResult] = {}

    for pair, df in data_map.items():
        engine = BacktestEngine(
            strategy=strategy,
            strategy_config=StrategyConfig(position_size_pct=0.10),
            risk_config=RiskConfig(initial_capital=per_pair_capital),
        )
        result = engine.run(df, pair_name=pair)
        results[pair] = result

    # 3. Reconstruct daily equity curves
    pair_curves: dict[str, pd.DataFrame] = {}
    for pair, result in results.items():
        df = data_map[pair]
        avg_swap = df["swap_long"].mean()
        curve = reconstruct_equity(
            trades_df=result.trades_df,
            price_series=df["close"],
            timestamps=df["timestamp"],
            initial_capital=per_pair_capital,
            swap_per_day=avg_swap,
        )
        curve = curve[curve["timestamp"] >= eval_start_ts].reset_index(drop=True)
        pair_curves[pair] = curve

    # 4. Build portfolio equity curve
    merged = None
    for pair, curve in pair_curves.items():
        if curve.empty:
            continue
        daily = curve.set_index("timestamp")["equity"].resample("D").last().dropna()
        daily.name = pair
        if merged is None:
            merged = daily.to_frame()
        else:
            merged = merged.join(daily, how="outer")

    if merged is None or merged.empty:
        print("  No equity curves to merge -- skipping scenario.")
        return None

    merged = merged.sort_index().ffill().bfill()
    portfolio_equity = merged.sum(axis=1)
    peak = portfolio_equity.cummax()
    drawdown = (peak - portfolio_equity) / peak

    portfolio_curve = pd.DataFrame({
        "timestamp": portfolio_equity.index,
        "equity": portfolio_equity.values,
        "drawdown": drawdown.values,
    }).reset_index(drop=True)

    # 5. Calculate metrics
    pair_metrics_map: dict[str, BacktestMetrics] = {}
    all_trades_dfs: list[pd.DataFrame] = []

    for pair, curve in pair_curves.items():
        trades_df = results[pair].trades_df
        if len(trades_df) > 0:
            trades_eval = trades_df[
                trades_df["entry_time"] >= eval_start_ts
            ].reset_index(drop=True)
        else:
            trades_eval = trades_df
        m = calculate_metrics(curve, trades_eval, per_pair_capital)
        pair_metrics_map[pair] = m
        if len(trades_eval) > 0:
            all_trades_dfs.append(trades_eval)

    all_trades = pd.concat(all_trades_dfs, ignore_index=True) if all_trades_dfs else pd.DataFrame()
    portfolio_metrics = calculate_metrics(portfolio_curve, all_trades, TOTAL_CAPITAL)

    total_trades = sum(m.total_trades for m in pair_metrics_map.values())
    print(f"  Portfolio return: {portfolio_metrics.total_return:.1%}  "
          f"MaxDD: {portfolio_metrics.max_drawdown:.1%}  "
          f"Trades: {total_trades}")

    # Find worst drawdown event
    worst_event = _find_worst_event(portfolio_curve, scenario)

    return ScenarioResult(
        scenario=scenario,
        portfolio_curve=portfolio_curve,
        portfolio_metrics=portfolio_metrics,
        pair_metrics=pair_metrics_map,
        pairs_used=list(data_map.keys()),
        total_trades=total_trades,
        worst_event=worst_event,
    )


def _find_worst_event(curve: pd.DataFrame, scenario: Scenario) -> str:
    """Describe the worst drawdown period in the scenario."""
    if curve.empty or curve["drawdown"].max() == 0:
        return "No drawdown"
    idx_max = curve["drawdown"].idxmax()
    dd_val = curve["drawdown"].iloc[idx_max]
    dd_date = curve["timestamp"].iloc[idx_max]
    return f"{dd_val:.1%} on {dd_date.strftime('%Y-%m-%d')}"


# ---------------------------------------------------------------------------
# Drawdown event extraction (across all scenarios)
# ---------------------------------------------------------------------------

def extract_drawdown_events(
    all_results: list[ScenarioResult],
) -> pd.DataFrame:
    """Extract the top 10 deepest drawdown events across all scenarios."""
    events: list[dict] = []

    for sr in all_results:
        curve = sr.portfolio_curve
        if curve.empty:
            continue

        dd = curve["drawdown"].values
        ts = curve["timestamp"].values

        # Walk through drawdown to find distinct events (peaks)
        in_dd = False
        dd_start_idx = 0
        for i in range(len(dd)):
            if dd[i] > 0 and not in_dd:
                in_dd = True
                dd_start_idx = i
            elif dd[i] == 0 and in_dd:
                # Drawdown ended -- record it
                segment = dd[dd_start_idx:i]
                peak_idx = dd_start_idx + int(np.argmax(segment))
                recovery_days = i - peak_idx
                events.append({
                    "Scenario": sr.scenario.name,
                    "Start": pd.Timestamp(ts[dd_start_idx]).strftime("%Y-%m-%d"),
                    "Trough": pd.Timestamp(ts[peak_idx]).strftime("%Y-%m-%d"),
                    "MaxDD": dd[peak_idx],
                    "Recovery": pd.Timestamp(ts[i]).strftime("%Y-%m-%d"),
                    "RecoveryDays": recovery_days,
                })
                in_dd = False

        # Handle ongoing drawdown at end of curve
        if in_dd:
            segment = dd[dd_start_idx:]
            peak_idx = dd_start_idx + int(np.argmax(segment))
            events.append({
                "Scenario": sr.scenario.name,
                "Start": pd.Timestamp(ts[dd_start_idx]).strftime("%Y-%m-%d"),
                "Trough": pd.Timestamp(ts[peak_idx]).strftime("%Y-%m-%d"),
                "MaxDD": dd[peak_idx],
                "Recovery": "Ongoing",
                "RecoveryDays": len(dd) - peak_idx,
            })

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events).sort_values("MaxDD", ascending=False).head(10)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

SCENARIO_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def chart_equity_overlay(
    all_results: list[ScenarioResult], output: Path
) -> None:
    """01 - All scenario portfolio equity curves overlaid, normalized to $100k."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, sr in enumerate(all_results):
        curve = sr.portfolio_curve
        if curve.empty:
            continue
        # Normalize days from start for overlay
        days = (curve["timestamp"] - curve["timestamp"].iloc[0]).dt.days
        ax.plot(
            days,
            curve["equity"],
            label=sr.scenario.name,
            color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)],
            linewidth=1.3,
        )

    ax.axhline(TOTAL_CAPITAL, color="gray", ls="--", alpha=0.4, label="Starting Capital")
    ax.set_title("V3 Stress Test: Portfolio Equity by Scenario (Normalized Start)")
    ax.set_xlabel("Trading Days from Eval Start")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  Saved {output}")


def chart_drawdowns(
    all_results: list[ScenarioResult], output: Path
) -> None:
    """02 - Drawdown comparison across scenarios (stacked subplots)."""
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, sr in enumerate(all_results):
        ax = axes[i]
        curve = sr.portfolio_curve
        if curve.empty:
            ax.set_title(f"{sr.scenario.name} -- no data")
            continue

        ax.fill_between(
            curve["timestamp"],
            0,
            -curve["drawdown"] * 100,
            color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)],
            alpha=0.6,
        )
        ax.set_ylabel("DD (%)")
        max_dd = curve["drawdown"].max()
        ax.set_title(
            f"{sr.scenario.name}  |  Max DD: {max_dd:.1%}",
            fontsize=10,
        )
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    plt.suptitle("V3 Stress Test: Drawdowns by Scenario", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output}")


def chart_summary_table(
    all_results: list[ScenarioResult], output: Path
) -> None:
    """03 - Summary table as PNG."""
    columns = ["Scenario", "Return", "MaxDD", "Sharpe", "Trades", "WinRate", "Swap$", "Worst Event"]
    rows: list[list[str]] = []

    for sr in all_results:
        m = sr.portfolio_metrics
        rows.append([
            sr.scenario.name,
            f"{m.total_return:.1%}",
            f"{m.max_drawdown:.1%}",
            f"{m.sharpe_ratio:.2f}",
            str(sr.total_trades),
            f"{m.win_rate:.0%}",
            f"${m.total_swap_profit:.0f}",
            sr.worst_event,
        ])

    fig, ax = plt.subplots(figsize=(16, max(3, 0.6 * (len(rows) + 1))))
    ax.axis("off")

    # Color rows by grade
    row_colors = []
    for sr in all_results:
        grade = _grade(sr.portfolio_metrics)
        if grade == "PASS":
            row_colors.append(["#d4edda"] * len(columns))
        elif grade == "CAUTION":
            row_colors.append(["#fff3cd"] * len(columns))
        else:
            row_colors.append(["#f8d7da"] * len(columns))

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472c4")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("V3 Stress Test Summary", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output}")


def chart_metrics_comparison(
    all_results: list[ScenarioResult], output: Path
) -> None:
    """04 - Grouped bar charts: Return, MaxDD, Sharpe side-by-side."""
    names = [sr.scenario.name for sr in all_results]
    # Shorten names for readability
    short_names = []
    for n in names:
        if len(n) > 20:
            short_names.append(n[:18] + "...")
        else:
            short_names.append(n)

    returns = [sr.portfolio_metrics.total_return * 100 for sr in all_results]
    max_dds = [-sr.portfolio_metrics.max_drawdown * 100 for sr in all_results]
    sharpes = [sr.portfolio_metrics.sharpe_ratio for sr in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))
    width = 0.6

    # Returns
    colors_ret = ["#2ca02c" if r >= 0 else "#d62728" for r in returns]
    axes[0].bar(x, returns, width, color=colors_ret, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Total Return (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Max Drawdown (shown as negative)
    axes[1].bar(x, max_dds, width, color="#d62728", edgecolor="black", linewidth=0.5, alpha=0.7)
    axes[1].set_title("Max Drawdown (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    axes[1].axhline(-20, color="red", ls="--", alpha=0.5, label="CAUTION: 20%")
    axes[1].axhline(-30, color="darkred", ls="--", alpha=0.5, label="FAIL: 30%")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Sharpe
    colors_sh = ["#2ca02c" if s >= 0 else "#d62728" for s in sharpes]
    axes[2].bar(x, sharpes, width, color=colors_sh, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Sharpe Ratio")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("V3 Stress Test: Metrics Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output}")


def chart_worst_drawdowns(
    events_df: pd.DataFrame, output: Path
) -> None:
    """05 - Table of the 10 deepest drawdown events across all scenarios."""
    if events_df.empty:
        print("  Skipped worst drawdowns chart (no events)")
        return

    columns = ["Scenario", "Start", "Trough", "MaxDD", "Recovery", "Days to Recover"]
    rows: list[list[str]] = []
    for _, row in events_df.iterrows():
        rows.append([
            row["Scenario"],
            row["Start"],
            row["Trough"],
            f"{row['MaxDD']:.1%}",
            row["Recovery"],
            str(row["RecoveryDays"]),
        ])

    fig, ax = plt.subplots(figsize=(16, max(3, 0.5 * (len(rows) + 1))))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        cellColours=[["#f0f0f0"] * len(columns)] * len(rows),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472c4")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("V3 Stress Test: 10 Deepest Drawdown Events", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output}")


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def _grade(m: BacktestMetrics) -> str:
    """Grade a scenario: PASS / CAUTION / FAIL."""
    if m.max_drawdown > 0.30 or m.total_return < -0.15:
        return "FAIL"
    if m.max_drawdown > 0.20 or m.total_return < -0.05:
        return "CAUTION"
    return "PASS"


def print_scorecard(all_results: list[ScenarioResult]) -> None:
    """Print the stress scorecard to console."""
    print("\n" + "=" * 90)
    print("  V3 STRESS TEST SCORECARD")
    print("=" * 90)
    header = (
        f"{'Scenario':<30} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
        f"{'Trades':>7} {'WinRate':>8} {'Grade':>8}"
    )
    print(header)
    print("-" * len(header))

    grades = []
    for sr in all_results:
        m = sr.portfolio_metrics
        grade = _grade(m)
        grades.append(grade)
        print(
            f"{sr.scenario.name:<30} {m.total_return:>7.1%} {m.max_drawdown:>7.1%} "
            f"{m.sharpe_ratio:>8.2f} {sr.total_trades:>7d} "
            f"{m.win_rate:>7.0%} {grade:>8}"
        )

    print("-" * len(header))

    # Overall assessment
    fails = grades.count("FAIL")
    cautions = grades.count("CAUTION")
    passes = grades.count("PASS")

    print(f"\n  Results: {passes} PASS  |  {cautions} CAUTION  |  {fails} FAIL")
    print()

    if fails == 0 and cautions == 0:
        print("  OVERALL: STRONG -- V3 survives all tested market regimes.")
    elif fails == 0:
        print("  OVERALL: ACCEPTABLE -- V3 shows stress in some regimes but no failures.")
    elif fails <= 1:
        print("  OVERALL: MARGINAL -- V3 fails in at least one crisis scenario.")
    else:
        print("  OVERALL: WEAK -- V3 fails in multiple scenarios. Consider redesign.")

    print()
    print("  Grading criteria:")
    print("    PASS:    MaxDD < 20% AND Return > -5%")
    print("    CAUTION: MaxDD 20-30% OR Return -5% to -15%")
    print("    FAIL:    MaxDD > 30% OR Return < -15%")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[ScenarioResult] = []

    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nNo scenarios completed successfully -- aborting.")
        return

    # Generate charts
    print("\n\nGenerating charts ...")
    chart_equity_overlay(all_results, OUTPUT_DIR / "01_scenario_equity_overlay.png")
    chart_drawdowns(all_results, OUTPUT_DIR / "02_scenario_drawdowns.png")
    chart_summary_table(all_results, OUTPUT_DIR / "03_scenario_summary_table.png")
    chart_metrics_comparison(all_results, OUTPUT_DIR / "04_metrics_comparison.png")

    events_df = extract_drawdown_events(all_results)
    chart_worst_drawdowns(events_df, OUTPUT_DIR / "05_worst_drawdown_events.png")

    # Console scorecard
    print_scorecard(all_results)
    print(f"\nCharts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
