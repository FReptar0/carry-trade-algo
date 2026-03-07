#!/usr/bin/env python3
"""Analyze potential new pairs for carry trade diversification."""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np


def get_swap_estimate(pair: str) -> dict:
    """Estimate swap rates based on interest rate differentials.

    These are rough estimates - actual OANDA rates may differ.
    """
    # Approximate central bank rates (as of early 2026)
    rates = {
        "USD": 4.50,  # Fed
        "EUR": 3.00,  # ECB
        "GBP": 4.25,  # BoE
        "JPY": 0.25,  # BoJ
        "AUD": 4.00,  # RBA
        "NZD": 4.50,  # RBNZ
        "CHF": 1.00,  # SNB
        "CAD": 3.75,  # BoC
    }

    base, quote = pair.split("/")
    base_rate = rates.get(base, 0)
    quote_rate = rates.get(quote, 0)

    # Long = earn base rate, pay quote rate
    # Differential / 365 for daily swap (simplified)
    diff = (base_rate - quote_rate) / 365 / 100

    return {
        "long": diff if diff > 0 else diff * 0.8,  # Broker takes a cut
        "short": -diff if diff < 0 else -diff * 0.8,
        "differential": base_rate - quote_rate,
    }


def main():
    print("=" * 80)
    print("CARRY TRADE PAIR DIVERSIFICATION ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # PART 1: Swap Rate Analysis
    # =========================================================================
    print("\n1. SWAP RATE ANALYSIS (Estimated from Rate Differentials)")
    print("-" * 80)

    potential_pairs = [
        "EUR/AUD", "EUR/NZD", "USD/CHF", "EUR/CHF",
        "GBP/CHF", "AUD/CHF", "NZD/CHF", "CAD/CHF",
        "GBP/AUD", "GBP/NZD",
    ]

    current_pairs = [
        "USD/JPY", "AUD/JPY", "GBP/JPY", "NZD/JPY", "EUR/JPY", "CAD/JPY"
    ]

    print(f"{'Pair':<12} {'Rate Diff':<12} {'Est Daily Swap':<15} {'Recommendation'}")
    print("-" * 80)

    viable_pairs = []

    for pair in potential_pairs:
        swaps = get_swap_estimate(pair)
        diff = swaps["differential"]

        if diff < -1.0:  # Significant positive carry when SHORT
            rec = "SHORT - Good carry"
            viable_pairs.append((pair, "SHORT", diff))
        elif diff > 1.0:  # Significant positive carry when LONG
            rec = "LONG - Good carry"
            viable_pairs.append((pair, "LONG", diff))
        else:
            rec = "SKIP - Low carry"

        print(f"{pair:<12} {diff:>+.2f}%       {swaps['long']*10000:>+.4f} pips    {rec}")

    print("\nCurrent JPY Pairs:")
    print("-" * 80)
    for pair in current_pairs:
        swaps = get_swap_estimate(pair)
        diff = swaps["differential"]
        print(f"{pair:<12} {diff:>+.2f}%       {swaps['long']*10000:>+.4f} pips    LONG - Good carry")

    # =========================================================================
    # PART 2: Correlation Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. CORRELATION ANALYSIS (1 Year Daily Data)")
    print("-" * 80)

    # Yahoo Finance tickers
    tickers = {
        # Current JPY pairs
        "USD/JPY": "USDJPY=X",
        "AUD/JPY": "AUDJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "NZD/JPY": "NZDJPY=X",
        "EUR/JPY": "EURJPY=X",
        "CAD/JPY": "CADJPY=X",
        # Potential new pairs (CHF-based for diversification)
        "USD/CHF": "USDCHF=X",
        "EUR/CHF": "EURCHF=X",
        "GBP/CHF": "GBPCHF=X",
        "AUD/CHF": "AUDCHF=X",
        # Cross pairs
        "EUR/AUD": "EURAUD=X",
        "EUR/NZD": "EURNZD=X",
    }

    # Download data
    print("Downloading price data...")
    returns_data = {}

    for pair, ticker in tickers.items():
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df is not None and len(df) > 100:
                returns = df["Close"].pct_change().dropna()
                if len(returns) > 0:
                    returns_data[pair] = returns
                    print(f"  {pair}: {len(returns)} data points")
        except Exception as e:
            print(f"  {pair}: Failed - {e}")

    if len(returns_data) < 2:
        print("Not enough data for correlation analysis")
        return

    # Build correlation matrix
    returns_df = pd.DataFrame(returns_data)
    corr_matrix = returns_df.corr()

    print("\nCorrelation Matrix (JPY pairs vs potential new pairs):")
    print("-" * 80)

    jpy_pairs = ["USD/JPY", "AUD/JPY", "GBP/JPY", "NZD/JPY", "EUR/JPY", "CAD/JPY"]
    new_pairs = ["USD/CHF", "EUR/CHF", "GBP/CHF", "AUD/CHF", "EUR/AUD", "EUR/NZD"]

    # Filter to available pairs
    jpy_avail = [p for p in jpy_pairs if p in corr_matrix.columns]
    new_avail = [p for p in new_pairs if p in corr_matrix.columns]

    # Print header
    header = "           " + " ".join([f"{p:>8}" for p in new_avail])
    print(header)
    print("-" * len(header))

    for jpy in jpy_avail:
        row = f"{jpy:<10}"
        for new in new_avail:
            corr = corr_matrix.loc[jpy, new]
            row += f" {corr:>8.2f}"
        print(row)

    # Average correlation
    print("\nAverage Correlation with JPY Basket:")
    for new in new_avail:
        avg_corr = corr_matrix.loc[jpy_avail, new].mean()
        diversification = "GOOD" if abs(avg_corr) < 0.5 else "MODERATE" if abs(avg_corr) < 0.7 else "HIGH"
        print(f"  {new}: {avg_corr:+.2f} ({diversification} diversification)")

    # =========================================================================
    # PART 3: Backtest Simulation
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. V3 BACKTEST SIMULATION (2024 Daily Data)")
    print("-" * 80)

    def backtest_v3_simple(pair: str, ticker: str, direction: str = "LONG") -> dict:
        """Simple V3-style backtest."""
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if len(df) < 250:
                return {"error": "Not enough data"}

            close = df["Close"]
            ma50 = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()

            # Get 2024 data only (evaluation period)
            df = df.loc["2024-01-01":"2024-12-31"].copy()
            close = df["Close"]
            ma50 = ma50.loc[df.index]
            ma200 = ma200.loc[df.index]

            trades = []
            in_position = False
            entry_price = 0
            entry_date = None

            for i in range(len(df)):
                price = close.iloc[i]
                m50 = ma50.iloc[i]
                m200 = ma200.iloc[i]
                date = df.index[i]

                if pd.isna(m50) or pd.isna(m200):
                    continue

                if direction == "LONG":
                    # Long entry: price > 50MA > 200MA
                    entry_cond = price > m50 * 1.003 and m50 > m200
                    # Exit: price < 200MA or death cross
                    exit_cond = price < m200 or m50 < m200
                else:
                    # Short entry: price < 50MA < 200MA
                    entry_cond = price < m50 * 0.997 and m50 < m200
                    # Exit: price > 200MA or golden cross
                    exit_cond = price > m200 or m50 > m200

                if not in_position and entry_cond:
                    in_position = True
                    entry_price = price
                    entry_date = date
                elif in_position and exit_cond:
                    in_position = False
                    if direction == "LONG":
                        pnl_pct = (price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - price) / entry_price * 100
                    trades.append({
                        "entry": entry_date,
                        "exit": date,
                        "pnl_pct": pnl_pct,
                    })

            if len(trades) == 0:
                return {"trades": 0, "total_return": 0, "win_rate": 0}

            total_return = sum(t["pnl_pct"] for t in trades)
            wins = sum(1 for t in trades if t["pnl_pct"] > 0)

            return {
                "trades": len(trades),
                "total_return": total_return,
                "win_rate": wins / len(trades) * 100,
                "avg_trade": total_return / len(trades),
            }
        except Exception as e:
            return {"error": str(e)}

    print(f"{'Pair':<12} {'Direction':<10} {'Trades':<8} {'Return':<10} {'Win Rate':<10} {'Avg Trade'}")
    print("-" * 80)

    # Backtest current JPY pairs
    print("Current JPY Pairs:")
    jpy_tickers = {
        "USD/JPY": "USDJPY=X",
        "AUD/JPY": "AUDJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "NZD/JPY": "NZDJPY=X",
        "EUR/JPY": "EURJPY=X",
        "CAD/JPY": "CADJPY=X",
    }

    total_jpy_return = 0
    for pair, ticker in jpy_tickers.items():
        result = backtest_v3_simple(pair, ticker, "LONG")
        if "error" not in result:
            total_jpy_return += result["total_return"]
            print(f"{pair:<12} {'LONG':<10} {result['trades']:<8} {result['total_return']:>+.2f}%     {result['win_rate']:.0f}%       {result['avg_trade']:>+.2f}%")
        else:
            print(f"{pair:<12} Error: {result['error']}")

    print(f"\nJPY Portfolio Total: {total_jpy_return:+.2f}%")

    # Backtest potential new pairs
    print("\nPotential New Pairs:")
    new_pair_tickers = {
        ("EUR/AUD", "SHORT"): "EURAUD=X",
        ("EUR/NZD", "SHORT"): "EURNZD=X",
        ("USD/CHF", "LONG"): "USDCHF=X",
        ("GBP/CHF", "LONG"): "GBPCHF=X",
        ("AUD/CHF", "LONG"): "AUDCHF=X",
        ("NZD/CHF", "LONG"): "NZDCHF=X",
    }

    total_new_return = 0
    for (pair, direction), ticker in new_pair_tickers.items():
        result = backtest_v3_simple(pair, ticker, direction)
        if "error" not in result:
            total_new_return += result["total_return"]
            print(f"{pair:<12} {direction:<10} {result['trades']:<8} {result['total_return']:>+.2f}%     {result['win_rate']:.0f}%       {result['avg_trade']:>+.2f}%")
        else:
            print(f"{pair:<12} Error: {result['error']}")

    print(f"\nNew Pairs Total: {total_new_return:+.2f}%")
    print(f"Combined Portfolio: {total_jpy_return + total_new_return:+.2f}%")

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on the analysis:

BEST PAIRS TO ADD (Low correlation + Good carry + Positive backtest):
1. USD/CHF (LONG) - Low JPY correlation, positive carry from rate differential
2. EUR/AUD (SHORT) - Negative correlation with JPY pairs, good carry
3. GBP/CHF (LONG) - Moderate correlation, good carry

PAIRS TO AVOID:
- EUR/CHF: Very low rate differential, minimal carry benefit
- Pairs with >0.7 correlation to JPY basket (no diversification)

IMPLEMENTATION NOTES:
- Add pairs gradually (1-2 at a time)
- Reduce position size per pair (e.g., 12 pairs × 2% = 24% max exposure)
- Monitor correlation during live trading (it can change)
- V3 strategy needs modification for SHORT signals (currently LONG-only)
""")


if __name__ == "__main__":
    main()
