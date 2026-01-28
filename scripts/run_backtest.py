"""
Phase 2: Run Carry Trade Backtest
==================================
Run this to see the strategy results with charts.

Usage:
    uv run python scripts/run_backtest.py

What this does (simple version):
    1. Generates 1 year of fake forex data
    2. Runs the carry trade bot on it
    3. Shows you: did the bot make money? How much risk did it take?
    4. Saves charts so you can see the equity curve and trades

The bot's rules:
    BUY when: swap is positive + price trending up + not overbought
    SELL when: lost 2% (stop loss) OR gained 6% (take profit) OR trailing stop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.config.settings import DEFAULT_PAIRS, StrategyConfig
from src.data.generator import SyntheticDataGenerator
from src.strategy.carry_trade import CarryTradeStrategy
from src.backtest.engine import BacktestEngine

OUT = Path("results/reports/phase2")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")


def main() -> None:
    print("=" * 60)
    print("PHASE 2: Carry Trade Backtest Results")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all_pairs(days=365)

    strategy = CarryTradeStrategy()
    engine = BacktestEngine(strategy)

    results = {}
    for pair_name, df in data.items():
        result = engine.run(df, pair_name)
        results[pair_name] = result
        print(f"\n{result.summary()}")

    # --- Chart 1: Equity curves ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (name, result) in zip(axes, results.items()):
        eq = result.equity_df
        if len(eq) > 0:
            ax.plot(eq["timestamp"], eq["equity"], linewidth=1.2)
            ax.axhline(10_000, color="gray", linestyle="--", alpha=0.5, label="Starting capital")
            ax.set_title(f"{name} - Equity Curve")
            ax.set_ylabel("Account Value ($)")
            ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "1_equity_curves.png", dpi=120)
    plt.close()
    print(f"\nSaved: {OUT}/1_equity_curves.png")

    # --- Chart 2: Drawdown ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (name, result) in zip(axes, results.items()):
        eq = result.equity_df
        if len(eq) > 0:
            ax.fill_between(eq["timestamp"], 0, -eq["drawdown"] * 100, alpha=0.7, color="red")
            ax.set_title(f"{name} - Drawdown (how far below peak)")
            ax.set_ylabel("Drawdown (%)")
            ax.set_ylim(bottom=min(-eq["drawdown"].max() * 100 * 1.1, -1))
    plt.tight_layout()
    plt.savefig(OUT / "2_drawdown.png", dpi=120)
    plt.close()
    print(f"Saved: {OUT}/2_drawdown.png")

    # --- Chart 3: Trade scatter (for pairs with trades) ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (name, result) in zip(axes, results.items()):
        df = data[name]
        ax.plot(df["timestamp"], df["close"], linewidth=0.4, alpha=0.6, color="gray")
        ax.set_title(f"{name} - Trades on Price Chart")
        ax.set_ylabel("Price")

        trades = result.trades_df
        if len(trades) > 0:
            # Green triangle = entry, red = exit
            ax.scatter(trades["entry_time"], trades["entry_price"],
                      marker="^", color="green", s=30, zorder=5, label="Entry")
            ax.scatter(trades["exit_time"], trades["exit_price"],
                      marker="v", color="red", s=30, zorder=5, label="Exit")
            ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "3_trades.png", dpi=120)
    plt.close()
    print(f"Saved: {OUT}/3_trades.png")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Pair':<10} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'SwapPnL':>8}")
    print("-" * 62)
    for name, r in results.items():
        m = r.metrics
        print(
            f"{name:<10} {m.total_return:>7.2%} {m.max_drawdown:>7.2%} "
            f"{m.sharpe_ratio:>8.2f} {m.total_trades:>7d} {m.win_rate:>7.1%} {m.total_swap_profit:>8.2f}"
        )

    print("\n--- What do these numbers mean? ---")
    print("Return:  Total profit/loss as % of starting capital")
    print("MaxDD:   Worst peak-to-valley drop (lower = less painful)")
    print("Sharpe:  Return per unit of risk (>1 = good, <0 = losing money)")
    print("WinRate: % of trades that made money")
    print("SwapPnL: How much money came purely from daily swap income")
    print()
    print("Key insight: AUD/JPY has many trades but terrible results.")
    print("This happens because the 2% stop loss gets triggered constantly")
    print("on a volatile pair -- the price bounces around more than 2%")
    print("regularly, so the bot keeps buying and getting stopped out.")
    print("This is called 'stop-loss whipsaw' and it's a real problem")
    print("that carry traders face. Phase 3 will optimize these parameters.")
    print()
    print(f"All charts saved to: {OUT}/")


if __name__ == "__main__":
    main()
