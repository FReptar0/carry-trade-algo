"""
Multi-Timeframe Strategy Backtest
=================================
Compare V3 (single timeframe) vs V4 (multi-timeframe with regime).

Usage:
    uv run python scripts/run_multi_timeframe_backtest.py
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
from src.data.multi_timeframe import MultiTimeframeLoader
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3
from src.strategy.carry_trade_v4 import CarryTradeStrategyV4, StrategyConfigV4
from src.regime.detector import RegimeDetector
from src.regime.adapters import RegimeAdapter
from src.backtest.engine import BacktestEngine
from src.config.settings import StrategyConfig

OUT = Path("results/reports/phase8")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME STRATEGY COMPARISON")
    print("V3 (Single TF) vs V4 (Multi-TF + Regime)")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    loader = HistoricalDataLoader(cache_dir="data/cache")
    mtf_loader = MultiTimeframeLoader(cache_dir="data/cache")

    pairs = ["USD/JPY", "AUD/JPY"]

    # Store results
    v3_results = {}
    v4_results = {}
    v4_mtf_results = {}

    for pair in pairs:
        print(f"\n--- {pair} ---")

        # Load daily data (for V3 and V4 fallback)
        daily = loader.load_pair(pair, start="2021-01-01", end="2024-12-31", interval="1d")
        if len(daily) == 0:
            print(f"  No data for {pair}")
            continue

        daily = clean_data(daily)
        print(f"  Daily bars: {len(daily)}")

        # Load multi-timeframe data
        try:
            mtf_context = mtf_loader.load_aligned(pair, start="2021-01-01", end="2024-12-31")
            daily_mtf = mtf_context.daily
            print(f"  Multi-TF bars: {len(daily_mtf)} (with weekly context)")
        except Exception as e:
            print(f"  Multi-TF loading failed: {e}")
            daily_mtf = None

        # V3 Strategy (baseline)
        print("\n  Running V3 (Single TF)...")
        v3_config = StrategyConfigV3(
            short_ma_period=50,
            long_ma_period=200,
            min_swap_threshold=0.003,
            atr_stop_mult=3.0,
        )
        v3_strategy = CarryTradeStrategyV3(v3_config)
        v3_engine = BacktestEngine(v3_strategy, StrategyConfig(
            stop_loss_pct=0.15,
            take_profit_pct=0.50,
            position_size_pct=0.08,
        ))
        result = v3_engine.run(daily, pair)
        v3_results[pair] = result
        m = result.metrics
        print(f"    Return: {m.total_return:>7.2%}  Sharpe: {m.sharpe_ratio:>6.2f}  "
              f"Trades: {m.total_trades:>3}  MaxDD: {m.max_drawdown:>6.2%}")

        # V4 Strategy (single TF fallback)
        print("\n  Running V4 (Single TF fallback)...")
        v4_config = StrategyConfigV4(
            daily_ma_period=20,
            min_swap_threshold=0.003,
            base_atr_mult=3.0,
            require_weekly_alignment=True,  # Will use fallback since no weekly cols
        )
        v4_strategy = CarryTradeStrategyV4(
            config=v4_config,
            regime_detector=RegimeDetector(),
            regime_adapter=RegimeAdapter(),
        )
        v4_engine = BacktestEngine(v4_strategy, StrategyConfig(
            stop_loss_pct=0.15,
            take_profit_pct=0.50,
            position_size_pct=0.08,
        ))
        result = v4_engine.run(daily, pair)
        v4_results[pair] = result
        m = result.metrics
        print(f"    Return: {m.total_return:>7.2%}  Sharpe: {m.sharpe_ratio:>6.2f}  "
              f"Trades: {m.total_trades:>3}  MaxDD: {m.max_drawdown:>6.2%}")

        # V4 Strategy (multi-TF)
        if daily_mtf is not None:
            print("\n  Running V4 (Multi-TF + Regime)...")
            v4_mtf_strategy = CarryTradeStrategyV4(
                config=v4_config,
                regime_detector=RegimeDetector(),
                regime_adapter=RegimeAdapter(),
            )
            result = v4_engine.run(daily_mtf, pair)
            v4_mtf_results[pair] = result
            m = result.metrics
            print(f"    Return: {m.total_return:>7.2%}  Sharpe: {m.sharpe_ratio:>6.2f}  "
                  f"Trades: {m.total_trades:>3}  MaxDD: {m.max_drawdown:>6.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Pair':<12} {'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'MaxDD':>10}")
    print("-" * 70)

    for pair in pairs:
        if pair in v3_results:
            m = v3_results[pair].metrics
            print(f"{pair:<12} {'V3 (Single TF)':<20} {m.total_return:>9.2%} {m.sharpe_ratio:>8.2f} "
                  f"{m.total_trades:>8} {m.max_drawdown:>9.2%}")

        if pair in v4_results:
            m = v4_results[pair].metrics
            print(f"{'':<12} {'V4 (ST Fallback)':<20} {m.total_return:>9.2%} {m.sharpe_ratio:>8.2f} "
                  f"{m.total_trades:>8} {m.max_drawdown:>9.2%}")

        if pair in v4_mtf_results:
            m = v4_mtf_results[pair].metrics
            print(f"{'':<12} {'V4 (Multi-TF)':<20} {m.total_return:>9.2%} {m.sharpe_ratio:>8.2f} "
                  f"{m.total_trades:>8} {m.max_drawdown:>9.2%}")

        print()

    # Plot comparison
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 6 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]

    colors = {
        "V3": "blue",
        "V4 (Fallback)": "orange",
        "V4 (Multi-TF)": "green",
    }

    for ax, pair in zip(axes, pairs):
        if pair in v3_results:
            eq = v3_results[pair].equity_df
            if len(eq) > 0:
                ax.plot(eq["timestamp"], eq["equity"], label="V3 (Single TF)",
                       linewidth=1.5, alpha=0.8, color=colors["V3"])

        if pair in v4_results:
            eq = v4_results[pair].equity_df
            if len(eq) > 0:
                ax.plot(eq["timestamp"], eq["equity"], label="V4 (Fallback)",
                       linewidth=1.5, alpha=0.8, color=colors["V4 (Fallback)"])

        if pair in v4_mtf_results:
            eq = v4_mtf_results[pair].equity_df
            if len(eq) > 0:
                ax.plot(eq["timestamp"], eq["equity"], label="V4 (Multi-TF)",
                       linewidth=1.5, alpha=0.8, color=colors["V4 (Multi-TF)"])

        ax.axhline(100000, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
        ax.set_title(f"{pair} - Strategy Comparison", fontsize=14)
        ax.set_ylabel("Equity ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(OUT / "multi_timeframe_comparison.png", dpi=120)
    plt.close()

    print(f"\n  Chart saved: {OUT}/multi_timeframe_comparison.png")

    # Insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    MULTI-TIMEFRAME BENEFITS:
    1. Weekly trend filter prevents trading against big picture
    2. Regime awareness adjusts parameters dynamically
    3. Better entry timing with daily pullbacks

    EXPECTED IMPROVEMENTS:
    - Fewer trades (more selective)
    - Better win rate (aligned with weekly trend)
    - Lower drawdown (regime-adjusted stops)

    NEXT STEPS:
    - Phase 9: Optimize exit logic (profit-scaled trailing)
    - Phase 10: Validate with 30-day paper trading protocol
    """)


if __name__ == "__main__":
    main()
