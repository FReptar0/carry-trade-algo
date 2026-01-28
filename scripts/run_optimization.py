"""
Phase 3: Parameter Optimization & Analysis
============================================
Run this to find the best strategy parameters and validate them.

Usage:
    uv run python scripts/run_optimization.py

What this does (simple version):
    1. GRID SEARCH: Tries many combinations of stop loss, SMA period, etc.
       Finds which settings make the most money with least risk.
    2. SENSITIVITY ANALYSIS: Shows heatmaps of how changing one parameter
       affects results. Helps you understand WHICH parameters matter most.
    3. WALK-FORWARD: Tests if the "best" parameters actually work on
       data they've never seen (guards against overfitting).
    4. BUY & HOLD COMPARISON: Compares the strategy vs just buying and holding.

Why this matters:
    In Phase 2, AUD/JPY lost 55% because the 2% stop loss was too tight.
    Grid search will find a better stop loss distance. But we need walk-forward
    to make sure we're not just fitting to noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import time

from src.config.settings import DEFAULT_PAIRS
from src.data.generator import SyntheticDataGenerator
from src.optimization.grid_search import GridSearchOptimizer
from src.optimization.walk_forward import WalkForwardAnalyzer
from src.strategy.carry_trade import CarryTradeStrategy
from src.backtest.engine import BacktestEngine
from src.visualization.charts import (
    plot_buy_and_hold_comparison,
    plot_equity_with_drawdown,
    plot_optimization_heatmap,
    plot_trade_timeline,
    plot_walk_forward,
)

OUT = Path("results/reports/phase3")
OUT.mkdir(parents=True, exist_ok=True)


# Smaller grid adapted to our synthetic data's swap scale
PARAM_GRID = {
    "min_swap_threshold": [0.001, 0.005, 0.01],
    "trend_filter_period": [20, 50, 100],
    "stop_loss_pct": [0.02, 0.04, 0.06, 0.08],
    "position_size_pct": [0.05, 0.10, 0.15],
}


def main() -> None:
    print("=" * 60)
    print("PHASE 3: Parameter Optimization")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all_pairs(days=365)

    # Focus on AUD/JPY -- the pair that had the worst results in Phase 2
    # because it has enough volatility and trades to optimize
    target_pair = "AUD/JPY"
    df = data[target_pair]

    # ===== STEP 1: Grid Search =====
    print(f"\n--- 1. GRID SEARCH on {target_pair} ---")
    total_combos = 1
    for v in PARAM_GRID.values():
        total_combos *= len(v)
    print(f"Testing {total_combos} parameter combinations...")

    t0 = time.time()
    optimizer = GridSearchOptimizer(df, target_pair)
    results = optimizer.run(PARAM_GRID)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    results_df = optimizer.results_to_df(results)

    # Show top 5
    print(f"\nTop 5 parameter sets (ranked by Sharpe ratio):")
    print(f"{'#':>3} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7} {'StopLoss':>9} {'SMA':>5} {'PosSize':>8}")
    print("-" * 72)
    for i, r in enumerate(results[:5]):
        print(
            f"{i+1:>3} {r.sharpe:>7.2f} {r.total_return:>7.2%} {r.max_drawdown:>7.2%} "
            f"{r.win_rate:>7.1%} {r.total_trades:>7d} {r.params.get('stop_loss_pct',0):>9.2%} "
            f"{r.params.get('trend_filter_period',0):>5d} {r.params.get('position_size_pct',0):>7.1%}"
        )

    best = results[0]
    print(f"\nBest parameters: {best.params}")
    print(f"  Sharpe: {best.sharpe:.2f}, Return: {best.total_return:.2%}, MaxDD: {best.max_drawdown:.2%}")

    # Show worst for comparison
    worst = results[-1]
    print(f"\nWorst parameters: {worst.params}")
    print(f"  Sharpe: {worst.sharpe:.2f}, Return: {worst.total_return:.2%}, MaxDD: {worst.max_drawdown:.2%}")

    print("\nWhat this tells you:")
    print("  The difference between good and bad parameters can be huge.")
    print("  A wider stop loss reduces whipsaw (getting stopped out too often).")
    print("  But too wide = you hold losing trades too long.")

    # ===== STEP 2: Sensitivity Heatmaps =====
    print(f"\n--- 2. PARAMETER SENSITIVITY ---")

    plot_optimization_heatmap(
        results_df, "stop_loss_pct", "trend_filter_period", "sharpe",
        OUT / "1_heatmap_stoploss_vs_sma.png"
    )
    print(f"  Saved: {OUT}/1_heatmap_stoploss_vs_sma.png")
    print("  (Shows how stop loss and SMA period interact)")

    plot_optimization_heatmap(
        results_df, "stop_loss_pct", "position_size_pct", "total_return",
        OUT / "2_heatmap_stoploss_vs_possize.png"
    )
    print(f"  Saved: {OUT}/2_heatmap_stoploss_vs_possize.png")
    print("  (Shows how stop loss and bet size affect total return)")

    plot_optimization_heatmap(
        results_df, "stop_loss_pct", "min_swap_threshold", "sharpe",
        OUT / "3_heatmap_stoploss_vs_swap.png"
    )
    print(f"  Saved: {OUT}/3_heatmap_stoploss_vs_swap.png")

    # ===== STEP 3: Run best params and generate charts =====
    print(f"\n--- 3. BACKTEST WITH BEST PARAMS ---")

    from src.config.settings import StrategyConfig

    best_config = StrategyConfig(**best.params)
    best_strategy = CarryTradeStrategy(best_config)
    best_engine = BacktestEngine(best_strategy, best_config)
    best_result = best_engine.run(df, target_pair)
    print(best_result.summary())

    plot_equity_with_drawdown(best_result, OUT / "4_best_equity.png")
    print(f"\n  Saved: {OUT}/4_best_equity.png")

    plot_trade_timeline(best_result, df, OUT / "5_best_trades.png")
    print(f"  Saved: {OUT}/5_best_trades.png")

    plot_buy_and_hold_comparison(best_result, df, OUT / "6_vs_buyhold.png")
    print(f"  Saved: {OUT}/6_vs_buyhold.png")
    print("  (Compares your strategy vs just buying on day 1 and holding)")

    # ===== STEP 4: Walk-Forward Analysis =====
    print(f"\n--- 4. WALK-FORWARD ANALYSIS ---")
    print("Testing if best parameters work on unseen data...")
    print("(Train on 6 months, test on next 1 month, slide forward)\n")

    # Use a smaller grid for walk-forward (speed)
    wf_grid = {
        "stop_loss_pct": [0.03, 0.05, 0.08],
        "trend_filter_period": [20, 50, 100],
    }

    wf = WalkForwardAnalyzer(df, target_pair, wf_grid, train_days=150, test_days=30, step_days=30)
    wf_results = wf.run()
    wf_df = wf.results_to_df(wf_results)

    print(WalkForwardAnalyzer.summary(wf_results))

    if len(wf_df) > 0:
        print(f"\nPer-window details:")
        print(f"{'Window':>7} {'Train Sharpe':>13} {'Test Sharpe':>12} {'Test Return':>12} {'Best StopLoss':>14} {'Best SMA':>9}")
        print("-" * 70)
        for _, row in wf_df.iterrows():
            print(
                f"{row['window']:>7.0f} {row['train_sharpe']:>13.2f} {row['test_sharpe']:>12.2f} "
                f"{row['test_return']:>11.2%} {row.get('param_stop_loss_pct', 'N/A'):>14} "
                f"{row.get('param_trend_filter_period', 'N/A'):>9}"
            )

        plot_walk_forward(wf_df, OUT / "7_walk_forward.png")
        print(f"\n  Saved: {OUT}/7_walk_forward.png")
        print("  (Bars should be similar height = parameters are stable)")

    # ===== STEP 5: All pairs with best params =====
    print(f"\n--- 5. BEST PARAMS APPLIED TO ALL PAIRS ---")
    for name, pair_df in data.items():
        r = best_engine.run(pair_df, name)
        m = r.metrics
        print(f"  {name}: Return={m.total_return:>7.2%}  Sharpe={m.sharpe_ratio:>6.2f}  Trades={m.total_trades:>4d}")

    print(f"\n{'='*60}")
    print(f"All charts saved to: {OUT}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
