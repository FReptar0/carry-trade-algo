"""
Compare Strategy V1 vs V2 Performance
=====================================
Run this to see the improvement from enhanced strategy.

Usage:
    uv run python scripts/compare_strategies.py
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
from src.backtest.engine import BacktestEngine
from src.config.settings import StrategyConfig

OUT = Path("results/reports/comparison")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("STRATEGY COMPARISON: V1 (Basic) vs V2 (Enhanced)")
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

    # Strategy V1 (basic)
    print("\n--- Strategy V1 (Basic) ---")
    v1_config = StrategyConfig(
        stop_loss_pct=0.06,
        take_profit_pct=0.08,
        trend_filter_period=50,
        min_swap_threshold=0.005,
    )
    v1_strategy = CarryTradeStrategy(v1_config)
    v1_engine = BacktestEngine(v1_strategy, v1_config)

    v1_results = {}
    for pair, df in data.items():
        result = v1_engine.run(df, pair)
        v1_results[pair] = result
        m = result.metrics
        print(f"  {pair}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}  "
              f"Trades={m.total_trades:>3}  MaxDD={m.max_drawdown:>6.2%}")

    # Strategy V2 (enhanced)
    print("\n--- Strategy V2 (Enhanced) ---")
    v2_config = StrategyConfigV2(
        fast_ma_period=20,
        slow_ma_period=50,
        adx_threshold=20.0,  # Require some trend
        min_swap_threshold=0.005,
        atr_stop_mult=2.5,
        atr_tp_mult=5.0,
        trailing_activation=0.02,
    )
    v2_strategy = CarryTradeStrategyV2(v2_config)
    v2_engine = BacktestEngine(v2_strategy, StrategyConfig(
        stop_loss_pct=0.10,  # Will be overridden by ATR
        take_profit_pct=0.20,
        position_size_pct=0.10,
    ))

    v2_results = {}
    for pair, df in data.items():
        result = v2_engine.run(df, pair)
        v2_results[pair] = result
        m = result.metrics
        print(f"  {pair}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}  "
              f"Trades={m.total_trades:>3}  MaxDD={m.max_drawdown:>6.2%}")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Pair':<12} {'V1 Return':>12} {'V2 Return':>12} {'Improvement':>12}")
    print("-" * 50)

    total_v1 = 0
    total_v2 = 0
    for pair in data.keys():
        v1_ret = v1_results[pair].metrics.total_return
        v2_ret = v2_results[pair].metrics.total_return
        imp = v2_ret - v1_ret
        total_v1 += v1_ret
        total_v2 += v2_ret
        print(f"{pair:<12} {v1_ret:>11.2%} {v2_ret:>11.2%} {imp:>+11.2%}")

    print("-" * 50)
    print(f"{'Average':<12} {total_v1/len(data):>11.2%} {total_v2/len(data):>11.2%} {(total_v2-total_v1)/len(data):>+11.2%}")

    # Plot comparison
    fig, axes = plt.subplots(len(data), 1, figsize=(14, 5 * len(data)))
    if len(data) == 1:
        axes = [axes]

    for ax, pair in zip(axes, data.keys()):
        v1_eq = v1_results[pair].equity_df
        v2_eq = v2_results[pair].equity_df

        if len(v1_eq) > 0:
            ax.plot(v1_eq["timestamp"], v1_eq["equity"], label="V1 (Basic)", linewidth=1.2, alpha=0.8)
        if len(v2_eq) > 0:
            ax.plot(v2_eq["timestamp"], v2_eq["equity"], label="V2 (Enhanced)", linewidth=1.2, alpha=0.8)

        ax.axhline(100000, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{pair} - Strategy Comparison")
        ax.set_ylabel("Equity ($)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / "strategy_comparison.png", dpi=120)
    plt.close()

    print(f"\n  Chart saved to: {OUT}/strategy_comparison.png")

    # Analysis
    print("\n" + "=" * 70)
    print("WHAT V2 DOES DIFFERENTLY")
    print("=" * 70)
    print("""
    1. ADX FILTER (Trend Strength)
       - Only trades when ADX > 20 (meaningful trend)
       - Avoids whipsaw in ranging/choppy markets

    2. PULLBACK ENTRIES
       - Doesn't chase breakouts
       - Waits for price to retrace to moving average
       - Better entry prices = more room to stop

    3. DYNAMIC ATR STOPS
       - Stop = 2.5x ATR (adapts to volatility)
       - Wide in volatile markets, tight in calm
       - Avoids fixed % whipsaw problem

    4. MACD MOMENTUM
       - Only enters when momentum is improving
       - Confirms the trend has "fuel"

    5. TREND REVERSAL EXIT
       - Exits when fast MA crosses below slow MA
       - Doesn't wait for stop loss to be hit
    """)

    print("=" * 70)
    print("NEXT IMPROVEMENTS TO CONSIDER")
    print("=" * 70)
    print("""
    1. MULTI-TIMEFRAME: Use weekly trend + daily entry
    2. CORRELATION FILTER: Avoid trading correlated pairs together
    3. REGIME DETECTION: Identify carry-friendly vs risk-off regimes
    4. MACHINE LEARNING: Predict favorable entry conditions
    5. FUNDAMENTAL DATA: Interest rate expectations, not just current rates
    6. SENTIMENT: COT positioning, options market data
    """)


if __name__ == "__main__":
    main()
