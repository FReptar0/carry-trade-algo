"""
Exit Analysis Script
====================
Analyze MAE/MFE and exit optimization opportunities.

Usage:
    uv run python scripts/analyze_exits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.loader import HistoricalDataLoader
from src.data.preprocessor import clean_data
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3
from src.strategy.exit_manager import ExitManager, ExitManagerConfig, PositionState
from src.backtest.engine import BacktestEngine
from src.config.settings import StrategyConfig

OUT = Path("results/reports/phase9")
OUT.mkdir(parents=True, exist_ok=True)


def analyze_trade_excursions(trades_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
    """Analyze MAE/MFE for each trade."""
    results = []

    for _, trade in trades_df.iterrows():
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        pnl = trade["pnl"]

        # Get price data during trade
        mask = (price_data["timestamp"] >= entry_time) & (price_data["timestamp"] <= exit_time)
        trade_data = price_data[mask]

        if len(trade_data) == 0:
            continue

        # Calculate MFE and MAE
        high_during = trade_data["high"].max()
        low_during = trade_data["low"].min()

        mfe = (high_during - entry_price) / entry_price  # Best possible
        mae = (entry_price - low_during) / entry_price  # Worst drawdown
        actual_return = (exit_price - entry_price) / entry_price

        # Capture ratio: how much of MFE was captured
        capture_ratio = actual_return / mfe if mfe > 0 else 0

        results.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "actual_return": actual_return,
            "mfe": mfe,
            "mae": mae,
            "capture_ratio": capture_ratio,
            "duration_days": (exit_time - entry_time).total_seconds() / 86400,
        })

    return pd.DataFrame(results)


def plot_mfe_mae_analysis(excursions: pd.DataFrame, pair: str, output_path: Path):
    """Create MAE/MFE analysis chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MFE vs Actual Return
    ax1 = axes[0, 0]
    colors = ["green" if r > 0 else "red" for r in excursions["actual_return"]]
    ax1.scatter(excursions["mfe"] * 100, excursions["actual_return"] * 100, c=colors, alpha=0.6)
    ax1.plot([0, 15], [0, 15], "k--", alpha=0.3, label="Perfect capture")
    ax1.set_xlabel("MFE (%)")
    ax1.set_ylabel("Actual Return (%)")
    ax1.set_title("MFE vs Actual Return (below line = left money on table)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE Distribution
    ax2 = axes[0, 1]
    ax2.hist(excursions["mae"] * 100, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(excursions["mae"].mean() * 100, color="red", linestyle="--",
                label=f"Mean: {excursions['mae'].mean()*100:.1f}%")
    ax2.set_xlabel("MAE (%)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Maximum Adverse Excursion Distribution")
    ax2.legend()

    # 3. Capture Ratio Distribution
    ax3 = axes[1, 0]
    valid_capture = excursions[excursions["capture_ratio"] > 0]["capture_ratio"]
    ax3.hist(valid_capture * 100, bins=20, edgecolor="black", alpha=0.7)
    ax3.axvline(valid_capture.mean() * 100, color="red", linestyle="--",
                label=f"Mean: {valid_capture.mean()*100:.1f}%")
    ax3.set_xlabel("Capture Ratio (%)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("MFE Capture Ratio (how much of max profit captured)")
    ax3.legend()

    # 4. Duration vs Return
    ax4 = axes[1, 1]
    ax4.scatter(excursions["duration_days"], excursions["actual_return"] * 100, c=colors, alpha=0.6)
    ax4.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax4.set_xlabel("Duration (days)")
    ax4.set_ylabel("Return (%)")
    ax4.set_title("Trade Duration vs Return")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"{pair} - Trade Excursion Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def main():
    print("=" * 70)
    print("EXIT ANALYSIS - MAE/MFE and Optimization")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    loader = HistoricalDataLoader(cache_dir="data/cache")

    pairs = ["USD/JPY", "AUD/JPY"]
    all_excursions = {}

    for pair in pairs:
        print(f"\n--- {pair} ---")

        daily = loader.load_pair(pair, start="2021-01-01", end="2024-12-31", interval="1d")
        if len(daily) == 0:
            print(f"  No data for {pair}")
            continue

        daily = clean_data(daily)
        print(f"  Daily bars: {len(daily)}")

        # Run V3 strategy to get trades
        v3_config = StrategyConfigV3(
            short_ma_period=50,
            long_ma_period=200,
            min_swap_threshold=0.003,
            atr_stop_mult=3.0,
        )
        strategy = CarryTradeStrategyV3(v3_config)
        engine = BacktestEngine(strategy, StrategyConfig(
            stop_loss_pct=0.15,
            take_profit_pct=0.50,
            position_size_pct=0.08,
        ))

        result = engine.run(daily, pair)
        trades = result.trades_df

        if len(trades) == 0:
            print(f"  No trades for {pair}")
            continue

        print(f"  Trades: {len(trades)}")

        # Analyze excursions
        excursions = analyze_trade_excursions(trades, daily)
        all_excursions[pair] = excursions

        # Calculate statistics
        print(f"\n  Excursion Statistics:")
        print(f"    Average MFE:         {excursions['mfe'].mean()*100:>6.2f}%")
        print(f"    Average MAE:         {excursions['mae'].mean()*100:>6.2f}%")
        print(f"    Avg Capture Ratio:   {excursions['capture_ratio'].mean()*100:>6.1f}%")
        print(f"    Avg Duration:        {excursions['duration_days'].mean():>6.1f} days")

        # Winners vs Losers
        winners = excursions[excursions["actual_return"] > 0]
        losers = excursions[excursions["actual_return"] <= 0]

        if len(winners) > 0:
            print(f"\n  Winners ({len(winners)} trades):")
            print(f"    Avg Return:          {winners['actual_return'].mean()*100:>6.2f}%")
            print(f"    Avg MFE:             {winners['mfe'].mean()*100:>6.2f}%")
            print(f"    Avg Capture:         {winners['capture_ratio'].mean()*100:>6.1f}%")

        if len(losers) > 0:
            print(f"\n  Losers ({len(losers)} trades):")
            print(f"    Avg Return:          {losers['actual_return'].mean()*100:>6.2f}%")
            print(f"    Avg MAE:             {losers['mae'].mean()*100:>6.2f}%")

        # Create chart
        output_path = OUT / f"excursion_analysis_{pair.replace('/', '_')}.png"
        plot_mfe_mae_analysis(excursions, pair, output_path)
        print(f"\n  Chart saved: {output_path}")

    # Summary insights
    print("\n" + "=" * 70)
    print("EXIT OPTIMIZATION INSIGHTS")
    print("=" * 70)
    print("""
    1. MFE ANALYSIS
       - If capture ratio < 50%, exits are too early
       - Look at where price went after exit
       - Consider tighter trailing stops for winners

    2. MAE ANALYSIS
       - If MAE > stop loss %, stops are getting hit
       - If MAE << stop loss, stops could be tighter
       - Optimal stop = slightly above max MAE

    3. DURATION ANALYSIS
       - Short winners: possibly exiting too early
       - Long losers: should cut losses faster
       - Time-based exits can help marginal trades

    4. RECOMMENDATIONS
       - If avg capture < 60%: Trail stops tighter on profit
       - If avg MAE > 5%: Use regime-aware wider stops
       - If duration > 20 days with small profit: Add time exits
    """)

    # Exit Manager Demo
    print("\n" + "=" * 70)
    print("EXIT MANAGER DEMO")
    print("=" * 70)

    manager = ExitManager(ExitManagerConfig(
        base_atr_mult=3.0,
        profit_scale_levels=[
            (0.02, 1.5),
            (0.04, 1.2),
            (0.06, 1.0),
            (0.10, 0.8),
        ],
        max_hold_days=30,
    ))

    print("\n  Profit-Scaled Trailing:")
    for profit_pct, mult in manager.config.profit_scale_levels:
        print(f"    At {profit_pct*100:.0f}% profit: Trail by {mult:.1f}x ATR")

    print("\n  Time-Based Exits:")
    print(f"    Min hold: {manager.config.min_hold_hours} hours")
    print(f"    Max hold: {manager.config.max_hold_days} days")

    print(f"\n  All charts saved to: {OUT}/")


if __name__ == "__main__":
    main()
