"""
Synthetic Data Exploration - Phase 1
=====================================
Run this to visualize and understand the generated data.

Usage:
    uv run python scripts/explore_data.py

What is a carry trade? (simple version)
---------------------------------------
Imagine two bank accounts:
  - Japan pays you 0.25% per year
  - USA pays you 5.25% per year

A carry trade = borrow from Japan (cheap), put it in USA (earns more).
You pocket the ~5% difference every year. That daily payment is called "swap".

The catch: if the Japanese Yen gets stronger vs the Dollar, you lose money
on the exchange rate -- potentially way more than you earned from swap.

This script shows what our simulated forex data looks like.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")  # Save to files instead of showing windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_PAIRS
from src.data.generator import SyntheticDataGenerator
from src.data.interest_rates import InterestRateSimulator

# Output folder
OUT = Path("results/reports/phase1")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")


def main() -> None:
    print("=" * 60)
    print("PHASE 1: Synthetic Data Exploration")
    print("=" * 60)

    # --- Generate data ---
    gen = SyntheticDataGenerator(seed=42)
    data = gen.generate_all_pairs(days=365, dynamic_rates=False)
    data_dyn = SyntheticDataGenerator(seed=42).generate_all_pairs(
        days=365, dynamic_rates=True
    )

    # --- 1. Basic stats ---
    print("\n--- 1. DATA SUMMARY ---")
    print(f"{'Pair':<10} {'Bars':>6} {'Price Min':>10} {'Price Max':>10} {'Swap/Day':>10} {'Ann.Carry':>10}")
    print("-" * 58)
    for name, df in data.items():
        swap = df["swap_long"].iloc[0]
        print(
            f"{name:<10} {len(df):>6} {df['close'].min():>10.2f} "
            f"{df['close'].max():>10.2f} {swap:>10.4f} {swap*365:>10.2f}"
        )

    print("\nWhat this means:")
    print("  - USD/JPY swap +0.0137/day = you EARN holding long (USD rate > JPY rate)")
    print("  - EUR/USD swap -0.0021/day = you PAY holding long (EUR rate < USD rate)")
    print("  - Annual carry = daily swap x 365 = total yearly income from the rate gap")

    # --- 2. Price charts ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (name, df) in zip(axes, data.items()):
        ax.plot(df["timestamp"], df["close"], linewidth=0.5, alpha=0.9)
        pair_cfg = [p for p in DEFAULT_PAIRS if p.name == name][0]
        ax.axhline(pair_cfg.price_min, color="red", linestyle="--", alpha=0.4)
        ax.axhline(pair_cfg.price_max, color="red", linestyle="--", alpha=0.4)
        ax.set_title(f"{name} - 1 Year Hourly Prices")
        ax.set_ylabel("Price")
    plt.tight_layout()
    plt.savefig(OUT / "1_price_charts.png", dpi=120)
    plt.close()
    print(f"\nSaved: {OUT}/1_price_charts.png")

    # --- 3. Returns ---
    print("\n--- 2. RETURNS ANALYSIS ---")
    print("(Returns = % price change per bar. We use returns, not prices,")
    print(" because they're comparable across different price levels.)\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, df) in zip(axes, data.items()):
        ret = df["close"].pct_change().dropna()
        ax.hist(ret, bins=80, alpha=0.7, density=True, edgecolor="black", linewidth=0.2)
        ax.set_title(f"{name}")
        ax.set_xlabel("Hourly Return")
        info = f"std={ret.std():.5f}\nskew={ret.skew():.2f}\nkurt={ret.kurtosis():.1f}"
        ax.text(0.72, 0.95, info, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=dict(facecolor="wheat", alpha=0.5))

        print(f"  {name}: avg={ret.mean():.7f}  std={ret.std():.5f}  skew={ret.skew():.2f}  kurtosis={ret.kurtosis():.1f}")

    plt.suptitle("Hourly Returns Distribution", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "2_returns_dist.png", dpi=120)
    plt.close()
    print(f"\n  Kurtosis > 3 means 'fat tails' = extreme moves happen more often")
    print(f"  than a normal bell curve predicts. This is typical of real markets.")
    print(f"\n  Saved: {OUT}/2_returns_dist.png")

    # --- 4. Volatility ---
    print("\n--- 3. VOLATILITY ---")
    print("Volatility = how much prices jump around. High vol = risky for carry trades")
    print("because price swings can erase weeks of swap income in one bad day.\n")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    for ax, (name, df) in zip(axes, data.items()):
        ret = df["close"].pct_change().dropna()
        # 5-day rolling vol, annualized
        roll_vol = ret.rolling(24 * 5).std() * np.sqrt(24 * 252)
        ax.plot(df["timestamp"].iloc[1:], roll_vol.values, linewidth=0.6)
        ax.set_title(f"{name} - Rolling 5-Day Annualized Volatility")
        ax.set_ylabel("Vol")
        avg_vol = roll_vol.dropna().mean()
        ax.axhline(avg_vol, color="red", linestyle="--", alpha=0.5, label=f"Avg: {avg_vol:.1%}")
        ax.legend()
        print(f"  {name}: avg annualized vol = {avg_vol:.1%}")

    plt.tight_layout()
    plt.savefig(OUT / "3_volatility.png", dpi=120)
    plt.close()
    print(f"\n  Saved: {OUT}/3_volatility.png")

    # --- 5. Interest rates ---
    print("\n--- 4. INTEREST RATES ---")
    print("Central banks (Fed, BOJ, ECB) set rates every ~6 weeks.")
    print("They move in 0.25% steps. The GAP between two rates = your carry.\n")

    rate_sim = InterestRateSimulator(seed=42)
    all_rates = rate_sim.generate_all_pair_rates(DEFAULT_PAIRS, days=365)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for ax, (name, rdf) in zip(axes, all_rates.items()):
        base, quote = name.split("/")
        ax.step(rdf["date"], rdf["rate_base"] * 100, where="post", label=f"{base} rate", linewidth=1.5)
        ax.step(rdf["date"], rdf["rate_quote"] * 100, where="post", label=f"{quote} rate", linewidth=1.5)
        ax.fill_between(rdf["date"], rdf["rate_base"] * 100, rdf["rate_quote"] * 100,
                        alpha=0.15, step="post", color="green")
        ax.set_title(f"{name} - Central Bank Rates (the shaded gap = your carry)")
        ax.set_ylabel("Rate (%)")
        ax.legend()
        diff_start = rdf["rate_differential"].iloc[0] * 100
        diff_end = rdf["rate_differential"].iloc[-1] * 100
        print(f"  {name}: rate gap start={diff_start:.2f}%  end={diff_end:.2f}%")

    plt.tight_layout()
    plt.savefig(OUT / "4_interest_rates.png", dpi=120)
    plt.close()
    print(f"\n  Saved: {OUT}/4_interest_rates.png")

    # --- 6. Swap income comparison ---
    print("\n--- 5. SWAP INCOME: STATIC vs DYNAMIC RATES ---")
    print("Static = rates never change. Dynamic = central banks adjust rates over time.")
    print("In real life, rates always change. This shows why it matters.\n")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for ax, name in zip(axes, data.keys()):
        df_s = data[name]
        df_d = data_dyn[name]

        # One swap credit per day (first bar of each day)
        mask_s = ~df_s["timestamp"].dt.date.duplicated()
        mask_d = ~df_d["timestamp"].dt.date.duplicated()

        cum_s = df_s.loc[mask_s, "swap_long"].cumsum()
        cum_d = df_d.loc[mask_d, "swap_long"].cumsum()

        ax.plot(df_s.loc[mask_s, "timestamp"].values, cum_s.values, label="Static rates", linewidth=1.5)
        ax.plot(df_d.loc[mask_d, "timestamp"].values, cum_d.values, label="Dynamic rates", linewidth=1.5, linestyle="--")
        ax.axhline(0, color="black", alpha=0.3)
        ax.set_title(f"{name} - Cumulative Swap Income (holding long 1 year)")
        ax.set_ylabel("Total Swap Earned")
        ax.legend()

        final_s = cum_s.iloc[-1]
        final_d = cum_d.iloc[-1]
        print(f"  {name}: static={final_s:.2f}  dynamic={final_d:.2f}")

    plt.tight_layout()
    plt.savefig(OUT / "5_swap_income.png", dpi=120)
    plt.close()
    print(f"\n  Saved: {OUT}/5_swap_income.png")

    # --- Done ---
    print("\n" + "=" * 60)
    print(f"All charts saved to: {OUT}/")
    print("Open the PNG files to see the visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
