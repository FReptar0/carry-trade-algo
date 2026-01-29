"""
Regime Analysis Script
======================
Visualize and validate regime detection on real data.

Usage:
    uv run python scripts/run_regime_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from src.data.loader import HistoricalDataLoader
from src.data.preprocessor import clean_data
from src.regime.detector import RegimeDetector, CompositeRegime, TrendRegime
from src.regime.adapters import RegimeAdapter, get_regime_description

OUT = Path("results/reports/phase7")
OUT.mkdir(parents=True, exist_ok=True)


def plot_regime_analysis(
    data: pd.DataFrame,
    regimes: list,
    pair: str,
    output_path: Path,
) -> None:
    """Create comprehensive regime analysis chart."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    timestamps = data["timestamp"].iloc[-len(regimes) :]
    prices = data["close"].iloc[-len(regimes) :]

    # Extract regime data
    adx_vals = [r.adx_value for r in regimes]
    plus_di = [r.plus_di for r in regimes]
    minus_di = [r.minus_di for r in regimes]
    atr_vals = [r.atr_value for r in regimes]
    atr_avgs = [r.atr_avg for r in regimes]
    composite = [r.composite_regime for r in regimes]

    # Colors for composite regimes
    regime_colors = {
        CompositeRegime.TREND_LOW_VOL: "#4CAF50",  # Green
        CompositeRegime.TREND_HIGH_VOL: "#FF9800",  # Orange
        CompositeRegime.RANGE_LOW_VOL: "#2196F3",  # Blue
        CompositeRegime.RANGE_HIGH_VOL: "#F44336",  # Red
    }

    # Panel 1: Price with regime background
    ax1 = axes[0]
    ax1.plot(timestamps, prices, "k-", linewidth=1, label="Close")

    # Color background by regime
    for i in range(len(regimes) - 1):
        ax1.axvspan(
            timestamps.iloc[i],
            timestamps.iloc[i + 1],
            alpha=0.3,
            color=regime_colors[composite[i]],
        )

    ax1.set_ylabel("Price")
    ax1.set_title(f"{pair} - Price with Regime Background")
    ax1.legend(loc="upper left")

    # Panel 2: ADX with +DI/-DI
    ax2 = axes[1]
    ax2.plot(timestamps, adx_vals, "b-", linewidth=1.5, label="ADX")
    ax2.plot(timestamps, plus_di, "g--", linewidth=1, alpha=0.7, label="+DI")
    ax2.plot(timestamps, minus_di, "r--", linewidth=1, alpha=0.7, label="-DI")
    ax2.axhline(25, color="gray", linestyle=":", label="Strong trend (25)")
    ax2.axhline(15, color="gray", linestyle=":", alpha=0.5, label="Weak trend (15)")
    ax2.set_ylabel("ADX / DI")
    ax2.set_title("Trend Strength (ADX) and Direction (+DI / -DI)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 60)

    # Panel 3: ATR and volatility regime
    ax3 = axes[2]
    ax3.plot(timestamps, atr_vals, "purple", linewidth=1.5, label="ATR")
    ax3.plot(timestamps, atr_avgs, "orange", linewidth=1, alpha=0.7, label="ATR 60-day avg")
    ax3.fill_between(
        timestamps,
        [a * 1.5 for a in atr_avgs],
        [a * 0.7 for a in atr_avgs],
        alpha=0.2,
        color="gray",
        label="Normal range",
    )
    ax3.set_ylabel("ATR")
    ax3.set_title("Volatility (ATR vs Historical Average)")
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Regime timeline
    ax4 = axes[3]

    # Create regime timeline
    regime_num = [list(CompositeRegime).index(r) for r in composite]
    ax4.fill_between(timestamps, 0, 1, alpha=0.3, color="white")

    for i in range(len(regimes) - 1):
        ax4.axvspan(
            timestamps.iloc[i],
            timestamps.iloc[i + 1],
            alpha=0.8,
            color=regime_colors[composite[i]],
        )

    ax4.set_ylabel("Regime")
    ax4.set_yticks([])
    ax4.set_title("Composite Regime Over Time")

    # Add legend for regimes
    patches = [
        mpatches.Patch(color=c, label=r.value.replace("_", " ").title())
        for r, c in regime_colors.items()
    ]
    ax4.legend(handles=patches, loc="upper left", ncol=2, fontsize=8)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def print_regime_statistics(regimes: list, pair: str) -> dict:
    """Print and return regime statistics."""
    total = len(regimes)

    # Count each regime
    counts = {}
    for regime in CompositeRegime:
        count = sum(1 for r in regimes if r.composite_regime == regime)
        counts[regime] = count

    print(f"\n{pair} Regime Statistics:")
    print("-" * 50)

    stats = {}
    for regime, count in counts.items():
        pct = count / total * 100
        stats[regime.value] = {"count": count, "pct": pct}
        print(f"  {regime.value:20s}: {count:4d} bars ({pct:5.1f}%)")

    # Favorable vs unfavorable
    favorable = sum(1 for r in regimes if r.is_favorable)
    avoid = sum(1 for r in regimes if r.should_avoid_trading)

    print(f"\n  Favorable for trading: {favorable:4d} bars ({favorable/total*100:5.1f}%)")
    print(f"  Should avoid trading:  {avoid:4d} bars ({avoid/total*100:5.1f}%)")

    # Average confidence
    avg_conf = np.mean([r.confidence for r in regimes])
    print(f"\n  Average confidence: {avg_conf:.2f}")

    return stats


def main():
    print("=" * 70)
    print("REGIME DETECTION ANALYSIS")
    print("=" * 70)

    # Load real data
    print("\n--- Loading Real Data ---")
    loader = HistoricalDataLoader(cache_dir="data/cache")

    pairs = ["USD/JPY", "AUD/JPY"]
    all_stats = {}

    for pair in pairs:
        df = loader.load_pair(pair, start="2021-01-01", end="2024-12-31", interval="1d")
        if len(df) == 0:
            print(f"  {pair}: No data available")
            continue

        df = clean_data(df)
        print(f"  {pair}: {len(df)} daily bars")

        # Detect regimes
        detector = RegimeDetector()
        regimes = detector.detect_series(df)

        if not regimes:
            print(f"  {pair}: Insufficient data for regime detection")
            continue

        # Print statistics
        stats = print_regime_statistics(regimes, pair)
        all_stats[pair] = stats

        # Create visualization
        output_path = OUT / f"regime_analysis_{pair.replace('/', '_')}.png"
        plot_regime_analysis(df, regimes, pair, output_path)
        print(f"\n  Chart saved: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("REGIME DETECTION SUMMARY")
    print("=" * 70)

    adapter = RegimeAdapter()

    print("\nRegime Trading Recommendations:")
    for regime in CompositeRegime:
        params = adapter.adapt(regime)
        print(f"\n  {regime.value.upper()}:")
        print(f"    Should trade: {params.should_trade}")
        print(f"    Position size mult: {params.position_size_mult:.1f}x")
        print(f"    Stop loss mult: {params.stop_loss_mult:.1f}x")
        print(f"    Entry threshold: {params.entry_threshold:.0%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print(
        """
    1. REGIME DETECTION WORKING
       - ADX identifies trend strength
       - ATR ratio identifies volatility levels
       - 4-state composite provides clear guidance

    2. TRADING IMPLICATIONS
       - TREND_LOW_VOL: Trade aggressively (best regime)
       - TREND_HIGH_VOL: Trade with wider stops, smaller size
       - RANGE_LOW_VOL: Hold existing, avoid new entries
       - RANGE_HIGH_VOL: Do not trade (worst regime)

    3. NEXT STEPS
       - Integrate regime into V4 strategy (Phase 8)
       - Test regime-aware backtest vs baseline
       - Optimize regime parameters
    """
    )

    print(f"\nAll charts saved to: {OUT}/")


if __name__ == "__main__":
    main()
