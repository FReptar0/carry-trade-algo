"""
Phase 4: Real Data Integration & Strategy Recalibration
=========================================================
Run this to download real forex data, validate it, run the strategy,
and compare results with synthetic data.

Usage:
    uv run python scripts/run_real_data.py

What this does:
    1. Downloads real USD/JPY, AUD/JPY, EUR/USD data from Yahoo Finance
    2. Runs quality checks (gaps, outliers, completeness)
    3. Backtests the carry trade strategy on REAL data
    4. Compares synthetic vs real data statistics
    5. Re-optimizes parameters for real data

Why real data matters:
    Our synthetic data was random walks in a box. Real markets have:
    - Multi-month trends (USD/JPY went 115 → 150 during 2022-2023 rate hikes)
    - Volatility spikes (carry trade unwinding events)
    - Actual rate cycles that create/destroy carry opportunities
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
from src.data.loader import HistoricalDataLoader
from src.data.preprocessor import clean_data, compare_synthetic_vs_real, validate_data
from src.strategy.carry_trade import CarryTradeStrategy
from src.backtest.engine import BacktestEngine
from src.optimization.grid_search import GridSearchOptimizer
from src.visualization.charts import (
    plot_equity_with_drawdown,
    plot_trade_timeline,
    plot_buy_and_hold_comparison,
)

OUT = Path("results/reports/phase4")
OUT.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")


def main() -> None:
    print("=" * 60)
    print("PHASE 4: Real Data Integration")
    print("=" * 60)

    # ===== STEP 1: Download real data =====
    print("\n--- 1. DOWNLOADING REAL DATA ---")
    print("(First run downloads from Yahoo Finance; subsequent runs use cache)\n")

    loader = HistoricalDataLoader(cache_dir="data/cache")

    # yfinance hourly data is limited to ~2 years back
    # Use daily data for longer history
    pairs = ["USD/JPY", "AUD/JPY", "EUR/USD"]
    real_data_daily = {}
    real_data_hourly = {}

    # Daily data: 3+ years
    for pair in pairs:
        try:
            df = loader.load_pair(pair, start="2021-01-01", end="2024-12-31", interval="1d")
            if len(df) > 0:
                real_data_daily[pair] = clean_data(df)
                print(f"  {pair} daily: {len(real_data_daily[pair])} bars "
                      f"({real_data_daily[pair]['timestamp'].iloc[0].date()} to "
                      f"{real_data_daily[pair]['timestamp'].iloc[-1].date()})")
            else:
                print(f"  {pair} daily: No data returned")
        except Exception as e:
            print(f"  {pair} daily: Error - {e}")

    # Hourly data: Yahoo limits to ~730 days back from today
    for pair in pairs:
        try:
            df = loader.load_pair(pair, start="2024-06-01", end="2025-12-31", interval="1h")
            if len(df) > 0:
                real_data_hourly[pair] = clean_data(df)
                print(f"  {pair} hourly: {len(real_data_hourly[pair])} bars")
            else:
                print(f"  {pair} hourly: No data returned")
        except Exception as e:
            print(f"  {pair} hourly: Error - {e}")

    if not real_data_daily:
        print("\n  WARNING: No data was downloaded. This could mean:")
        print("  - No internet connection")
        print("  - Yahoo Finance is temporarily unavailable")
        print("  - yfinance needs updating")
        print("\n  Falling back to synthetic data for demonstration...")
        _run_fallback()
        return

    # ===== STEP 2: Data quality validation =====
    print("\n--- 2. DATA QUALITY VALIDATION ---")
    print("Checking for gaps, outliers, and completeness...\n")

    for pair, df in real_data_daily.items():
        report = validate_data(df, pair, freq_hours=24)
        status = "PASS" if report.passed else "FAIL"
        print(f"  {pair} [{status}]")
        print(f"    Rows: {report.total_rows}, Expected: ~{report.expected_bars}")
        print(f"    Completeness: {report.completeness_pct}%")
        print(f"    Duplicates: {report.duplicate_timestamps}")
        print(f"    Outliers (>5σ): {report.outlier_bars}")
        print(f"    Weekend bars removed: {report.weekend_bars_removed}")
        print()

    # ===== STEP 3: Price charts =====
    print("--- 3. REAL DATA PRICE CHARTS ---")

    fig, axes = plt.subplots(len(real_data_daily), 1, figsize=(14, 4 * len(real_data_daily)))
    if len(real_data_daily) == 1:
        axes = [axes]
    for ax, (pair, df) in zip(axes, real_data_daily.items()):
        ax.plot(df["timestamp"], df["close"], linewidth=0.8)
        ax.set_title(f"{pair} - Real Daily Prices")
        ax.set_ylabel("Price")
    plt.tight_layout()
    plt.savefig(OUT / "1_real_prices.png", dpi=120)
    plt.close()
    print(f"  Saved: {OUT}/1_real_prices.png")

    # ===== STEP 4: Synthetic vs Real comparison =====
    print("\n--- 4. SYNTHETIC vs REAL COMPARISON ---")
    print("How well did our fake data match reality?\n")

    gen = SyntheticDataGenerator(seed=42)
    syn_data = gen.generate_all_pairs(days=365)

    for pair in pairs:
        if pair not in real_data_daily:
            continue

        real_df = real_data_daily[pair]
        syn_df = syn_data.get(pair)
        if syn_df is None:
            continue

        comp = compare_synthetic_vs_real(syn_df, real_df, pair)
        print(f"  {pair}:")
        print(f"    {'Metric':<20} {'Synthetic':>12} {'Real':>12}")
        print(f"    {'-'*44}")
        for metric, syn_val, real_val in zip(comp["metric"], comp["synthetic"], comp["real"]):
            print(f"    {metric:<20} {str(syn_val):>12} {str(real_val):>12}")
        print()

    # ===== STEP 5: Backtest on real data =====
    print("--- 5. BACKTEST ON REAL DATA ---")

    # Use the data we have (daily or hourly)
    backtest_data = real_data_hourly if real_data_hourly else real_data_daily

    # First try default params
    print("\nDefault parameters:")
    strategy = CarryTradeStrategy()
    engine = BacktestEngine(strategy)

    for pair, df in backtest_data.items():
        result = engine.run(df, pair)
        m = result.metrics
        print(f"  {pair}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}  "
              f"Trades={m.total_trades:>4d}  MaxDD={m.max_drawdown:>6.2%}")

        plot_equity_with_drawdown(result, OUT / f"2_equity_{pair.replace('/', '')}.png")
        plot_trade_timeline(result, df, OUT / f"3_trades_{pair.replace('/', '')}.png")

    print(f"\n  Saved equity and trade charts to {OUT}/")

    # ===== STEP 6: Re-optimize for real data =====
    print("\n--- 6. PARAMETER OPTIMIZATION ON REAL DATA ---")

    # Pick the pair with the most data for optimization
    best_pair = max(backtest_data.keys(), key=lambda p: len(backtest_data[p]))
    opt_df = backtest_data[best_pair]
    print(f"Optimizing on {best_pair} ({len(opt_df)} bars)...")

    param_grid = {
        "stop_loss_pct": [0.02, 0.04, 0.06, 0.08],
        "trend_filter_period": [20, 50, 100],
        "position_size_pct": [0.05, 0.10, 0.15],
    }

    optimizer = GridSearchOptimizer(opt_df, best_pair)
    results = optimizer.run(param_grid)

    print(f"\nTop 5 (real data):")
    print(f"  {'#':>3} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Trades':>7} {'SL':>6} {'SMA':>5} {'Size':>6}")
    print(f"  {'-'*50}")
    for i, r in enumerate(results[:5]):
        print(
            f"  {i+1:>3} {r.sharpe:>7.2f} {r.total_return:>7.2%} {r.max_drawdown:>7.2%} "
            f"{r.total_trades:>7d} {r.params['stop_loss_pct']:>5.0%} "
            f"{r.params['trend_filter_period']:>5d} {r.params['position_size_pct']:>5.0%}"
        )

    # Run best params
    if results:
        best = results[0]
        print(f"\nBest real-data params: {best.params}")
        best_config = StrategyConfig(**best.params)
        best_strategy = CarryTradeStrategy(best_config)
        best_engine = BacktestEngine(best_strategy, best_config)

        for pair, df in backtest_data.items():
            result = best_engine.run(df, pair)
            m = result.metrics
            print(f"  {pair}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}  "
                  f"Trades={m.total_trades:>4d}")

            plot_equity_with_drawdown(result, OUT / f"4_best_{pair.replace('/', '')}.png")
            plot_buy_and_hold_comparison(result, df, OUT / f"5_vs_bh_{pair.replace('/', '')}.png")

    print(f"\n{'='*60}")
    print(f"All charts saved to: {OUT}/")
    print(f"{'='*60}")


def _run_fallback() -> None:
    """Run with synthetic data if no internet available."""
    print("\nRunning Phase 4 demo with synthetic data...")
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all_pairs(days=365)

    strategy = CarryTradeStrategy(StrategyConfig(stop_loss_pct=0.06))
    engine = BacktestEngine(strategy)

    for pair, df in data.items():
        result = engine.run(df, pair)
        m = result.metrics
        print(f"  {pair}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}")


if __name__ == "__main__":
    main()
