"""
Compare All Strategy Versions
=============================
V1 (Basic) vs V2 (Over-filtered) vs V3 (Long-term) vs Buy-and-Hold

Usage:
    uv run python scripts/compare_all_strategies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import HistoricalDataLoader
from src.data.preprocessor import clean_data
from src.strategy.carry_trade import CarryTradeStrategy
from src.strategy.carry_trade_v2 import CarryTradeStrategyV2, StrategyConfigV2
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3, BuyAndHoldStrategy
from src.backtest.engine import BacktestEngine
from src.config.settings import StrategyConfig

OUT = Path("results/reports/comparison")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 70)

    # Load real data
    print("\n--- Loading Real Data ---")
    loader = HistoricalDataLoader(cache_dir="data/cache")

    pairs = ["USD/JPY", "AUD/JPY"]
    data = {}

    for pair in pairs:
        df = loader.load_pair(pair, start="2021-01-01", end="2024-12-31", interval="1d")
        if len(df) > 0:
            data[pair] = clean_data(df)
            print(f"  {pair}: {len(data[pair])} daily bars")

    if not data:
        print("No data available. Run scripts/run_real_data.py first.")
        return

    # Define all strategies
    strategies = {}

    # V1 - Basic
    v1_config = StrategyConfig(
        stop_loss_pct=0.06,
        take_profit_pct=0.08,
        trend_filter_period=50,
        min_swap_threshold=0.005,
    )
    strategies["V1 (Basic)"] = (CarryTradeStrategy(v1_config), v1_config)

    # V2 - Over-filtered (proved worse)
    v2_config = StrategyConfigV2(
        fast_ma_period=20,
        slow_ma_period=50,
        adx_threshold=20.0,
        min_swap_threshold=0.005,
        atr_stop_mult=2.5,
        atr_tp_mult=5.0,
    )
    strategies["V2 (Filtered)"] = (CarryTradeStrategyV2(v2_config), StrategyConfig(
        stop_loss_pct=0.10,
        take_profit_pct=0.20,
        position_size_pct=0.10,
    ))

    # V3 - Long-term trend following
    v3_config = StrategyConfigV3(
        short_ma_period=50,
        long_ma_period=200,
        min_swap_threshold=0.003,
        atr_stop_mult=3.0,
        trailing_activation_pct=0.05,
    )
    strategies["V3 (Long-term)"] = (CarryTradeStrategyV3(v3_config), StrategyConfig(
        stop_loss_pct=0.15,  # Will be overridden by ATR
        take_profit_pct=0.50,  # No fixed TP
        position_size_pct=0.08,
    ))

    # Buy and Hold - Pure carry benchmark
    strategies["Buy & Hold"] = (BuyAndHoldStrategy(min_swap=0.003), StrategyConfig(
        stop_loss_pct=0.99,  # Essentially no stop
        take_profit_pct=0.99,  # No TP
        position_size_pct=0.10,
    ))

    # Run all strategies
    all_results = {name: {} for name in strategies}

    for name, (strategy, config) in strategies.items():
        print(f"\n--- {name} ---")
        engine = BacktestEngine(strategy, config)

        for pair, df in data.items():
            result = engine.run(df, pair)
            all_results[name][pair] = result
            m = result.metrics
            print(f"  {pair}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}  "
                  f"Trades={m.total_trades:>3}  MaxDD={m.max_drawdown:>6.2%}")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY - USD/JPY")
    print("=" * 70)
    print(f"\n{'Strategy':<18} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'MaxDD':>10}")
    print("-" * 60)

    for name in strategies:
        if "USD/JPY" in all_results[name]:
            m = all_results[name]["USD/JPY"].metrics
            print(f"{name:<18} {m.total_return:>9.2%} {m.sharpe_ratio:>8.2f} "
                  f"{m.total_trades:>8} {m.max_drawdown:>9.2%}")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY - AUD/JPY")
    print("=" * 70)
    print(f"\n{'Strategy':<18} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'MaxDD':>10}")
    print("-" * 60)

    for name in strategies:
        if "AUD/JPY" in all_results[name]:
            m = all_results[name]["AUD/JPY"].metrics
            print(f"{name:<18} {m.total_return:>9.2%} {m.sharpe_ratio:>8.2f} "
                  f"{m.total_trades:>8} {m.max_drawdown:>9.2%}")

    # Average performance
    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE (Both Pairs)")
    print("=" * 70)
    print(f"\n{'Strategy':<18} {'Avg Return':>12} {'Avg Sharpe':>12}")
    print("-" * 45)

    for name in strategies:
        returns = [all_results[name][p].metrics.total_return for p in data if p in all_results[name]]
        sharpes = [all_results[name][p].metrics.sharpe_ratio for p in data if p in all_results[name]]
        if returns:
            avg_ret = sum(returns) / len(returns)
            avg_sharpe = sum(sharpes) / len(sharpes)
            print(f"{name:<18} {avg_ret:>11.2%} {avg_sharpe:>12.2f}")

    # Plot comparison
    fig, axes = plt.subplots(len(data), 1, figsize=(14, 6 * len(data)))
    if len(data) == 1:
        axes = [axes]

    colors = {"V1 (Basic)": "blue", "V2 (Filtered)": "orange",
              "V3 (Long-term)": "green", "Buy & Hold": "red"}

    for ax, pair in zip(axes, data.keys()):
        for name in strategies:
            if pair in all_results[name]:
                eq = all_results[name][pair].equity_df
                if len(eq) > 0:
                    ax.plot(eq["timestamp"], eq["equity"], label=name,
                           linewidth=1.5, alpha=0.8, color=colors.get(name, "gray"))

        ax.axhline(100000, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
        ax.set_title(f"{pair} - All Strategies Comparison", fontsize=14)
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "all_strategies_comparison.png", dpi=120)
    plt.close()

    print(f"\n  Chart saved to: {OUT}/all_strategies_comparison.png")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. V1 (Basic): Simple rules, decent performance
       - Trades more frequently
       - Fixed percentage stops can whipsaw

    2. V2 (Filtered): WORSE! Over-filtering killed opportunities
       - ADX + MACD + Pullback = too restrictive
       - When all conditions align, opportunity often passed
       - Lesson: More filters ≠ better results

    3. V3 (Long-term): Follows the big trend
       - Uses 200-day MA (long-term direction)
       - Wider stops (3x ATR) = more breathing room
       - Holds longer to accumulate swap income
       - Exits on trend reversal, not arbitrary targets

    4. Buy & Hold: Pure carry benchmark
       - Shows what happens with no timing
       - If this beats active strategies, timing adds no value
       - Helps quantify how much "alpha" strategy generates
    """)

    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
    IF BUY & HOLD WINS:
    - The pair trends strongly (like USD/JPY 2021-2024)
    - Timing adds complexity without value
    - Consider simpler approach with risk management only

    IF V3 WINS:
    - Long-term trend following captures big moves
    - Wider stops avoid whipsaw
    - Patience (holding for carry) is rewarded

    IF V1 WINS:
    - Simpler is better for this market condition
    - Over-optimization (V2) hurts performance
    - Don't fix what isn't broken

    GENERAL ADVICE:
    - Start with the simplest strategy that works
    - Add complexity only if it clearly improves results
    - Backtest is optimistic - expect 30-50% worse live
    - Position sizing matters more than entry timing
    """)


if __name__ == "__main__":
    main()
