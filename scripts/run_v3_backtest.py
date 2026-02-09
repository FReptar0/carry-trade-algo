"""Multi-pair V3 backtest with charts.

Runs CarryTradeStrategyV3 on all 6 live JPY carry pairs using real Yahoo
Finance daily data, with OANDA-realistic swap rates.  Generates per-pair
and portfolio-level performance charts.

Pairs: USD/JPY, AUD/JPY, EUR/JPY, GBP/JPY, NZD/JPY, CAD/JPY
Data:  2023-01-01 to 2024-12-31  (first ~10 months warmup for 200MA)
Eval:  2024-01-01 to 2024-12-31
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import BacktestMetrics, calculate_metrics
from src.config.settings import RiskConfig, StrategyConfig
from src.data.loader import HistoricalDataLoader
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3
from src.visualization.charts import plot_trade_timeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAIRS = ["USD/JPY", "AUD/JPY", "EUR/JPY", "GBP/JPY", "NZD/JPY", "CAD/JPY"]
START = "2023-01-01"
END = "2024-12-31"
EVAL_START = pd.Timestamp("2024-01-01")
TOTAL_CAPITAL = 100_000.0
PER_PAIR_CAPITAL = TOTAL_CAPITAL / len(PAIRS)
OUTPUT_DIR = Path("results/reports/v3_backtest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fix_swap_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Divide swap columns by 100 to undo the inherited synthetic scaling.

    The loader computes ``diff / 365 * 100`` which was tuned for the
    synthetic generator.  OANDA daily swaps are ~100x smaller, so we
    divide back.
    """
    df = df.copy()
    df["swap_long"] = df["swap_long"] / 100.0
    df["swap_short"] = df["swap_short"] / 100.0
    return df


def reconstruct_equity(
    trades_df: pd.DataFrame,
    price_series: pd.Series,
    timestamps: pd.Series,
    initial_capital: float,
    swap_per_day: float,
) -> pd.DataFrame:
    """Build a smooth daily equity curve from trades + price data.

    The backtest engine records equity only every 24th bar, which is too
    sparse for daily data (~10 points over 250 bars).  This function
    replays all trades against the daily close prices to produce a
    proper daily equity curve.
    """
    equity = initial_capital
    peak = initial_capital
    records: list[dict] = []

    # Build trade lookup: list of (entry_time, exit_time, entry_price, size, swap_earned, pnl)
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

    # Walk through each day
    current_trade_idx = -1  # index into trades list when in a position
    in_position = False
    pos_entry_price = 0.0
    pos_size = 0.0
    cash = initial_capital
    swap_accrued = 0.0

    for i in range(len(timestamps)):
        ts = pd.Timestamp(timestamps.iloc[i])
        price = price_series.iloc[i]

        # Check if a trade was entered on this day
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

        # Check if a trade was exited on this day
        if in_position and current_trade_idx >= 0:
            t = trades[current_trade_idx]
            if t["exit_time"].date() == ts.date():
                # Close: return capital + pnl + swap
                cash += pos_entry_price * pos_size + t["pnl"] + t["swap_earned"]
                in_position = False
                current_trade_idx = -1

        # Mark-to-market
        if in_position:
            swap_accrued += swap_per_day * pos_size
            unrealized = (price - pos_entry_price) * pos_size + swap_accrued
            equity = cash + pos_entry_price * pos_size + unrealized
        else:
            equity = cash

        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        records.append({
            "timestamp": ts,
            "equity": equity,
            "drawdown": dd,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def chart_per_pair_equity(
    pair_curves: dict[str, pd.DataFrame],
    pair_metrics: dict[str, BacktestMetrics],
    output: Path,
) -> None:
    """3x2 grid with one equity curve per pair."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False)
    axes_flat = axes.flatten()

    for idx, pair in enumerate(PAIRS):
        ax = axes_flat[idx]
        if pair not in pair_curves or pair_curves[pair].empty:
            ax.set_title(f"{pair} — no data")
            continue
        curve = pair_curves[pair]
        m = pair_metrics[pair]
        ax.plot(curve["timestamp"], curve["equity"], linewidth=1.0)
        ax.axhline(PER_PAIR_CAPITAL, color="gray", ls="--", alpha=0.4)
        ax.set_title(
            f"{pair}  |  Ret: {m.total_return:.1%}  Sharpe: {m.sharpe_ratio:.2f}"
        )
        ax.set_ylabel("$")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("V3 Per-Pair Equity Curves (2024)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output}")


def chart_portfolio_equity_drawdown(
    portfolio_curve: pd.DataFrame, output: Path
) -> None:
    """Combined portfolio equity (top) + drawdown fill (bottom)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
    )

    ax1.plot(portfolio_curve["timestamp"], portfolio_curve["equity"], lw=1.2)
    ax1.axhline(TOTAL_CAPITAL, color="gray", ls="--", alpha=0.4)
    ax1.set_title("V3 Portfolio Equity (6 JPY Carry Pairs)")
    ax1.set_ylabel("Portfolio Value ($)")

    ax2.fill_between(
        portfolio_curve["timestamp"],
        0,
        -portfolio_curve["drawdown"] * 100,
        color="red",
        alpha=0.5,
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  Saved {output}")


def chart_trade_timelines(
    results: dict[str, BacktestResult],
    data_map: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """One trade-timeline chart per pair using the existing helper."""
    for pair, result in results.items():
        safe = pair.replace("/", "")
        out = output_dir / f"03_trades_{safe}.png"
        plot_trade_timeline(result, data_map[pair], out)
        print(f"  Saved {out}")


def chart_returns_distribution(
    portfolio_curve: pd.DataFrame, output: Path
) -> None:
    """Histogram of daily portfolio returns."""
    daily = portfolio_curve.set_index("timestamp")["equity"].resample("D").last().dropna()
    rets = daily.pct_change().dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rets, bins=50, alpha=0.7, edgecolor="black", lw=0.3, density=True)
    ax.axvline(rets.mean(), color="red", ls="--", label=f"Mean: {rets.mean():.5f}")
    ax.set_title("V3 Portfolio — Daily Returns Distribution (2024)")
    ax.set_xlabel("Daily Return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  Saved {output}")


def chart_pair_correlation(
    pair_curves: dict[str, pd.DataFrame], output: Path
) -> None:
    """Heatmap of daily return correlations between pairs."""
    daily_returns: dict[str, pd.Series] = {}
    for pair, curve in pair_curves.items():
        if curve.empty:
            continue
        daily = curve.set_index("timestamp")["equity"].resample("D").last().dropna()
        daily_returns[pair] = daily.pct_change().dropna()

    if len(daily_returns) < 2:
        print("  Skipped correlation chart (not enough pairs)")
        return

    ret_df = pd.DataFrame(daily_returns)
    corr = ret_df.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ticks = range(len(corr))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(corr.index)
    ax.set_title("Pair Return Correlation (2024)")

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  Saved {output}")


def chart_summary_table(
    pair_metrics: dict[str, BacktestMetrics],
    portfolio_metrics: BacktestMetrics,
    output: Path,
) -> None:
    """Render the summary metrics table as a PNG."""
    columns = ["Pair", "Return", "MaxDD", "Sharpe", "Trades", "WinRate", "Swap$"]
    rows: list[list[str]] = []

    for pair in PAIRS:
        if pair not in pair_metrics:
            continue
        m = pair_metrics[pair]
        rows.append([
            pair,
            f"{m.total_return:.1%}",
            f"{m.max_drawdown:.1%}",
            f"{m.sharpe_ratio:.2f}",
            str(m.total_trades),
            f"{m.win_rate:.0%}",
            f"${m.total_swap_profit:.2f}",
        ])

    # Portfolio row
    m = portfolio_metrics
    rows.append([
        "PORTFOLIO",
        f"{m.total_return:.1%}",
        f"{m.max_drawdown:.1%}",
        f"{m.sharpe_ratio:.2f}",
        str(m.total_trades),
        f"{m.win_rate:.0%}",
        f"${m.total_swap_profit:.2f}",
    ])

    fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * (len(rows) + 1))))
    ax.axis("off")

    colors = [["#f0f0f0"] * len(columns)] * (len(rows) - 1) + [["#d0e0ff"] * len(columns)]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        cellColours=colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Header styling
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472c4")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("V3 Multi-Pair Backtest Summary (2024)", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading daily data for 6 pairs ...")
    loader = HistoricalDataLoader()
    data_map: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        df = loader.load_pair(pair, start=START, end=END, interval="1d")
        if df.empty:
            print(f"  WARNING: no data for {pair}")
            continue
        df = fix_swap_scale(df)
        data_map[pair] = df
        print(f"  {pair}: {len(df)} daily bars")

    if not data_map:
        print("No data loaded — aborting.")
        return

    # ------------------------------------------------------------------
    # 2. Run V3 per pair
    # ------------------------------------------------------------------
    print("\nRunning V3 backtest per pair ...")
    strategy = CarryTradeStrategyV3(StrategyConfigV3(position_size_pct=0.10))
    results: dict[str, BacktestResult] = {}

    for pair, df in data_map.items():
        engine = BacktestEngine(
            strategy=strategy,
            strategy_config=StrategyConfig(position_size_pct=0.10),
            risk_config=RiskConfig(initial_capital=PER_PAIR_CAPITAL),
        )
        result = engine.run(df, pair_name=pair)
        results[pair] = result
        print(f"  {pair}: {result.metrics.total_trades} trades, "
              f"return={result.metrics.total_return:.2%}")

    # ------------------------------------------------------------------
    # 3. Reconstruct daily equity curves
    # ------------------------------------------------------------------
    print("\nReconstructing daily equity curves ...")
    pair_curves: dict[str, pd.DataFrame] = {}
    for pair, result in results.items():
        df = data_map[pair]
        # Average daily swap rate (already scaled)
        avg_swap = df["swap_long"].mean()
        curve = reconstruct_equity(
            trades_df=result.trades_df,
            price_series=df["close"],
            timestamps=df["timestamp"],
            initial_capital=PER_PAIR_CAPITAL,
            swap_per_day=avg_swap,
        )
        # Trim to evaluation period
        curve = curve[curve["timestamp"] >= EVAL_START].reset_index(drop=True)
        pair_curves[pair] = curve
        print(f"  {pair}: {len(curve)} daily points")

    # ------------------------------------------------------------------
    # 4. Build portfolio-level equity curve
    # ------------------------------------------------------------------
    print("\nBuilding portfolio equity curve ...")
    # Merge all pair curves on date
    merged = None
    for pair, curve in pair_curves.items():
        daily = curve.set_index("timestamp")["equity"].resample("D").last().dropna()
        daily.name = pair
        if merged is None:
            merged = daily.to_frame()
        else:
            merged = merged.join(daily, how="outer")

    if merged is not None:
        merged = merged.sort_index().ffill().bfill()
        portfolio_equity = merged.sum(axis=1)
        peak = portfolio_equity.cummax()
        drawdown = (peak - portfolio_equity) / peak

        portfolio_curve = pd.DataFrame({
            "timestamp": portfolio_equity.index,
            "equity": portfolio_equity.values,
            "drawdown": drawdown.values,
        }).reset_index(drop=True)
    else:
        print("No curves to merge — aborting.")
        return

    # ------------------------------------------------------------------
    # 5. Calculate per-pair & portfolio metrics
    # ------------------------------------------------------------------
    print("\nCalculating metrics ...")
    pair_metrics: dict[str, BacktestMetrics] = {}
    all_trades_dfs: list[pd.DataFrame] = []

    for pair, curve in pair_curves.items():
        trades_df = results[pair].trades_df
        # Filter trades to eval period
        if len(trades_df) > 0:
            trades_eval = trades_df[
                trades_df["entry_time"] >= EVAL_START
            ].reset_index(drop=True)
        else:
            trades_eval = trades_df
        m = calculate_metrics(curve, trades_eval, PER_PAIR_CAPITAL)
        pair_metrics[pair] = m
        if len(trades_eval) > 0:
            all_trades_dfs.append(trades_eval)

    # Portfolio-level metrics
    all_trades = pd.concat(all_trades_dfs, ignore_index=True) if all_trades_dfs else pd.DataFrame()
    portfolio_metrics = calculate_metrics(portfolio_curve, all_trades, TOTAL_CAPITAL)

    # ------------------------------------------------------------------
    # 6. Generate charts
    # ------------------------------------------------------------------
    print("\nGenerating charts ...")
    chart_per_pair_equity(pair_curves, pair_metrics, OUTPUT_DIR / "01_per_pair_equity.png")
    chart_portfolio_equity_drawdown(portfolio_curve, OUTPUT_DIR / "02_portfolio_equity_drawdown.png")
    chart_trade_timelines(results, data_map, OUTPUT_DIR)
    chart_returns_distribution(portfolio_curve, OUTPUT_DIR / "04_returns_distribution.png")
    chart_pair_correlation(pair_curves, OUTPUT_DIR / "05_pair_correlation.png")
    chart_summary_table(pair_metrics, portfolio_metrics, OUTPUT_DIR / "06_summary_table.png")

    # ------------------------------------------------------------------
    # 7. Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V3 MULTI-PAIR BACKTEST SUMMARY  (Eval: 2024-01-01 to 2024-12-31)")
    print("=" * 80)
    header = f"{'Pair':<12} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'Swap$':>10}"
    print(header)
    print("-" * len(header))
    for pair in PAIRS:
        if pair not in pair_metrics:
            continue
        m = pair_metrics[pair]
        print(
            f"{pair:<12} {m.total_return:>7.1%} {m.max_drawdown:>7.1%} "
            f"{m.sharpe_ratio:>8.2f} {m.total_trades:>7d} "
            f"{m.win_rate:>7.0%} {m.total_swap_profit:>10.2f}"
        )
    print("-" * len(header))
    m = portfolio_metrics
    print(
        f"{'PORTFOLIO':<12} {m.total_return:>7.1%} {m.max_drawdown:>7.1%} "
        f"{m.sharpe_ratio:>8.2f} {m.total_trades:>7d} "
        f"{m.win_rate:>7.0%} {m.total_swap_profit:>10.2f}"
    )
    print("=" * 80)
    print(f"\nCharts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
