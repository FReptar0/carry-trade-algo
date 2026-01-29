"""
Phase 5: Advanced Risk Management Demo
======================================
Run this to see all risk management components in action.

Usage:
    uv run python scripts/run_risk_analysis.py

What this does:
    1. Loads real forex data from cache
    2. Calculates VaR/CVaR using multiple methods
    3. Demonstrates position sizing (Kelly, Fixed Fractional, Volatility)
    4. Shows circuit breaker operation
    5. Generates risk heat maps and dashboards

Financial concepts covered:
    - Value at Risk (VaR): "How much could I lose on a bad day?"
    - Kelly Criterion: "How much should I bet to maximize growth?"
    - Circuit Breakers: "When should I stop trading?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import HistoricalDataLoader
from src.data.preprocessor import clean_data
from src.risk.position_sizing import PositionSizer
from src.risk.stop_loss import AdaptiveStopLoss
from src.risk.circuit_breakers import CircuitBreaker, RiskLimits, PortfolioState
from src.risk.risk_metrics import RiskAnalyzer
from src.visualization.charts import (
    plot_risk_heatmap,
    plot_var_comparison,
    plot_circuit_breaker_status,
)
from src.strategy.indicators import atr

OUT = Path("results/reports/phase5")
OUT.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")


def main() -> None:
    print("=" * 60)
    print("PHASE 5: Advanced Risk Management")
    print("=" * 60)

    # ===== STEP 1: Load real data =====
    print("\n--- 1. LOADING REAL DATA ---")
    loader = HistoricalDataLoader(cache_dir="data/cache")
    pairs = ["USD/JPY", "AUD/JPY", "EUR/USD"]
    data = {}

    for pair in pairs:
        try:
            df = loader.load_pair(pair, start="2021-01-01", end="2024-12-31", interval="1d")
            if len(df) > 0:
                data[pair] = clean_data(df)
                print(f"  {pair}: {len(data[pair])} bars")
        except Exception as e:
            print(f"  {pair}: Error - {e}")

    if not data:
        print("  No data available. Please run scripts/run_real_data.py first.")
        return

    # ===== STEP 2: Value at Risk Analysis =====
    print("\n--- 2. VALUE AT RISK (VaR) ANALYSIS ---")
    print("VaR answers: 'How much could I lose on a bad day?'\n")

    analyzer = RiskAnalyzer()
    portfolio_value = 100000

    for pair, df in data.items():
        returns = df["close"].pct_change().dropna()

        print(f"  {pair}:")
        var_95 = analyzer.historical_var(returns, confidence=0.95, portfolio_value=portfolio_value)
        var_99 = analyzer.historical_var(returns, confidence=0.99, portfolio_value=portfolio_value)

        print(f"    VaR 95%:  ${var_95.var:,.0f} ({var_95.var_pct:.2%})")
        print(f"    CVaR 95%: ${var_95.cvar:,.0f} ({var_95.cvar_pct:.2%})")
        print(f"    VaR 99%:  ${var_99.var:,.0f} ({var_99.var_pct:.2%})")
        print()

        # Generate VaR comparison chart for first pair
        if pair == list(data.keys())[0]:
            plot_var_comparison(returns, portfolio_value, OUT / "1_var_comparison.png")
            print(f"  Saved VaR comparison chart to {OUT}/1_var_comparison.png")

    # ===== STEP 3: Risk Summary =====
    print("\n--- 3. COMPREHENSIVE RISK SUMMARY ---")

    combined_returns = pd.concat([data[p]["close"].pct_change().dropna() for p in data.keys()])
    summary = analyzer.risk_summary(combined_returns, portfolio_value)

    print(f"  Daily Volatility:      {summary['daily_volatility']:.4f}")
    print(f"  Annualized Volatility: {summary['annualized_volatility']:.2%}")
    print(f"  Worst Day:             {summary['worst_day']:.2%}")
    print(f"  Best Day:              {summary['best_day']:.2%}")
    print(f"  Skewness:              {summary['skewness']:.3f}")
    print(f"  Kurtosis:              {summary['kurtosis']:.3f}")
    print(f"  Negative Days:         {summary['negative_days_pct']:.1%}")

    # ===== STEP 4: Position Sizing Demo =====
    print("\n--- 4. POSITION SIZING METHODS ---")
    print("How much should we risk on each trade?\n")

    sizer = PositionSizer(max_position_pct=0.20, kelly_fraction=0.25, default_risk_pct=0.02)

    # Simulated historical performance
    win_rate = 0.45
    avg_win = 0.035
    avg_loss = 0.025
    entry_price = 150.0
    current_atr = 0.75

    print(f"  Scenario: {portfolio_value:,} capital, {entry_price} entry, ATR={current_atr}")
    print(f"  Historical: {win_rate:.0%} win rate, {avg_win:.1%} avg win, {avg_loss:.1%} avg loss\n")

    # Kelly
    kelly_result = sizer.kelly_criterion(win_rate, avg_win, avg_loss)
    print(f"  KELLY CRITERION:")
    print(f"    Full Kelly:      {kelly_result.details['full_kelly']:.1%}")
    print(f"    Quarter Kelly:   {kelly_result.size:.1%}")
    if kelly_result.size == 0:
        print(f"    Verdict: Don't trade (no edge)")
    else:
        print(f"    Edge:            {kelly_result.details['expected_edge']:.4f} per trade")

    # Fixed Fractional
    stop_distance = entry_price * 0.02  # 2% stop
    ff_result = sizer.fixed_fractional(portfolio_value, stop_distance, entry_price, risk_pct=0.02)
    print(f"\n  FIXED FRACTIONAL (2% risk):")
    print(f"    Position size:   {ff_result.size:.1%} of capital")
    print(f"    Dollar risk:     ${ff_result.details['risk_amount']:,.0f}")

    # Volatility-based
    vol_result = sizer.volatility_based(portfolio_value, current_atr, entry_price, atr_multiplier=2.0)
    print(f"\n  VOLATILITY-BASED (2x ATR stop):")
    print(f"    Stop distance:   {vol_result.details['stop_distance_pct']:.2%}")
    print(f"    Position size:   {vol_result.size:.1%} of capital")

    # ===== STEP 5: Adaptive Stop Loss Demo =====
    print("\n--- 5. ADAPTIVE STOP LOSS (ATR-BASED) ---")
    print("Stop loss that adapts to market volatility.\n")

    stop_mgr = AdaptiveStopLoss(atr_multiplier=2.0, trailing_activation_pct=0.02)

    # Use first pair's data
    first_pair = list(data.keys())[0]
    first_df = data[first_pair]

    # Calculate ATR
    atr_series = stop_mgr.calculate_atr(first_df)
    current_atr = atr_series.iloc[-1]
    current_price = first_df["close"].iloc[-1]

    print(f"  {first_pair}:")
    print(f"    Current price:   {current_price:.2f}")
    print(f"    Current ATR:     {current_atr:.4f}")

    # Initial stop
    stop = stop_mgr.initial_stop(entry_price=current_price, is_long=True, atr=current_atr)
    print(f"    Initial stop:    {stop.stop_price:.2f} ({stop.distance_pct:.2%} below entry)")

    # Simulate trailing
    print(f"\n  Trailing stop simulation:")
    prices = [current_price, current_price * 1.02, current_price * 1.04, current_price * 1.03]
    for i, p in enumerate(prices):
        stop = stop_mgr.update_trailing(stop, p, current_atr)
        print(f"    Price {p:.2f} -> Stop {stop.stop_price:.2f} (HWM: {stop.high_water_mark:.2f})")

    # ===== STEP 6: Circuit Breakers Demo =====
    print("\n--- 6. CIRCUIT BREAKERS (KILL SWITCHES) ---")
    print("Automatic trading halt when limits are breached.\n")

    limits = RiskLimits(
        max_daily_loss_pct=0.03,
        max_weekly_loss_pct=0.07,
        max_drawdown_pct=0.20,
        max_positions=3,
        max_exposure_per_pair_pct=0.15,
    )

    breaker = CircuitBreaker(limits)

    print("  Configured limits:")
    print(f"    Daily loss:      {limits.max_daily_loss_pct:.0%}")
    print(f"    Weekly loss:     {limits.max_weekly_loss_pct:.0%}")
    print(f"    Max drawdown:    {limits.max_drawdown_pct:.0%}")
    print(f"    Max positions:   {limits.max_positions}")
    print(f"    Max per pair:    {limits.max_exposure_per_pair_pct:.0%}")

    # Scenario 1: Normal operation
    state_normal = PortfolioState(
        equity=99000,
        high_water_mark=100000,
        daily_pnl=-500,
        weekly_pnl=-1000,
        open_positions=2,
        exposure_by_pair={"USD/JPY": 0.10, "AUD/JPY": 0.08},
    )

    violations = breaker.check_limits(state_normal)
    print(f"\n  Normal scenario: {len(violations)} violations")

    status = breaker.get_status_report(state_normal)
    print(f"    Daily loss utilization:    {status['daily_loss_utilization']:.0%}")
    print(f"    Drawdown utilization:      {status['drawdown_utilization']:.0%}")
    print(f"    Trading halted:            {status['trading_halted']}")

    # Reset and test breach scenario
    breaker.reset()

    # Scenario 2: Breach
    state_breach = PortfolioState(
        equity=95000,
        high_water_mark=100000,
        daily_pnl=-3500,  # 3.5% daily loss
        weekly_pnl=-5000,
        open_positions=2,
        exposure_by_pair={"USD/JPY": 0.10},
    )

    violations = breaker.check_limits(state_breach)
    print(f"\n  Breach scenario: {len(violations)} violations")
    for v in violations:
        print(f"    [{v.severity}] {v.message}")
    print(f"    Trading halted: {breaker.is_trading_halted}")

    # Generate circuit breaker status chart
    status_breach = breaker.get_status_report(state_breach)
    plot_circuit_breaker_status(status_breach, OUT / "2_circuit_breaker_status.png")
    print(f"\n  Saved circuit breaker chart to {OUT}/2_circuit_breaker_status.png")

    # ===== STEP 7: Portfolio Risk Heat Map =====
    print("\n--- 7. PORTFOLIO RISK HEAT MAP ---")

    risk_by_pair = {}
    for pair, df in data.items():
        returns = df["close"].pct_change().dropna()
        var_result = analyzer.historical_var(returns, 0.95, portfolio_value)
        dd = (df["close"].max() - df["close"].iloc[-1]) / df["close"].max()

        risk_by_pair[pair] = {
            "exposure_pct": 0.10,  # Simulated exposure
            "var_contribution": var_result.var_pct,
            "drawdown_pct": dd,
            "volatility": returns.std() * np.sqrt(252),
        }

    plot_risk_heatmap(risk_by_pair, OUT / "3_risk_heatmap.png")
    print(f"  Saved risk heat map to {OUT}/3_risk_heatmap.png")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
    print("=" * 60)
    print("""
    Risk Management Components Built:

    1. POSITION SIZING
       - Kelly Criterion: Mathematically optimal (but use 1/4 Kelly)
       - Fixed Fractional: Simple, consistent risk per trade
       - Volatility-based: Adapts to market conditions

    2. ADAPTIVE STOP LOSS
       - ATR-based stops that widen in volatile markets
       - Trailing stops that lock in profits

    3. CIRCUIT BREAKERS
       - Daily/weekly loss limits
       - Maximum drawdown limit
       - Position count and exposure limits
       - Automatic trading halt on breach

    4. VaR ANALYSIS
       - Historical VaR (actual past losses)
       - Parametric VaR (assumes normal distribution)
       - Monte Carlo VaR (simulation-based)
       - Conditional VaR (expected shortfall)

    Key Insight:
    Risk management is MORE important than signal generation.
    A bad strategy with good risk management survives.
    A good strategy with bad risk management blows up.
    """)

    print(f"All charts saved to: {OUT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
