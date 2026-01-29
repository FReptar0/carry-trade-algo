#!/usr/bin/env python3
"""Run paper trading validation protocol.

This script demonstrates the 30-day paper trading validation protocol
that compares simulated paper trading results against backtest expectations.

Protocol checkpoints: Days 7, 14, 21, 30
Auto-abort triggers: 15% drawdown, 5 consecutive losses, 3 circuit breakers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date, timedelta
import random

import numpy as np
import pandas as pd

from src.validation import (
    PaperTradingValidator,
    ValidationCriteria,
    TradingProtocol,
    ProtocolDay,
    ProtocolStatus,
)
from src.backtest.metrics import BacktestMetrics


def generate_simulated_day(
    day_num: int,
    current_equity: float,
    regime: str,
    base_daily_return: float = 0.002,
    volatility: float = 0.01,
) -> ProtocolDay:
    """Generate simulated trading day results.

    In a real implementation, this would come from actual paper trading.
    """
    # Add some randomness to daily returns
    daily_return = base_daily_return + np.random.normal(0, volatility)

    # Worse performance in bad regimes
    if regime == "range_high_vol":
        daily_return -= 0.005
    elif regime == "trend_high_vol":
        daily_return -= 0.002

    daily_pnl = current_equity * daily_return
    ending_equity = current_equity + daily_pnl

    # Simulate occasional losing streaks
    if random.random() < 0.15:  # 15% chance of circuit breaker trigger
        circuit_breaker = True
        daily_return = -abs(daily_return) * 2  # Worse day
        daily_pnl = current_equity * daily_return
        ending_equity = current_equity + daily_pnl
    else:
        circuit_breaker = False

    return ProtocolDay(
        date=date.today() + timedelta(days=day_num),
        starting_equity=current_equity,
        ending_equity=ending_equity,
        daily_pnl=daily_pnl,
        daily_return=daily_return,
        trades_opened=random.randint(0, 2),
        trades_closed=random.randint(0, 2),
        max_drawdown_today=max(0, -daily_return),
        regime=regime,
        circuit_breaker_triggered=circuit_breaker,
    )


def run_validation_protocol():
    """Run the full 30-day validation protocol."""
    print("=" * 70)
    print("PAPER TRADING VALIDATION PROTOCOL")
    print("=" * 70)
    print()

    # Create protocol with custom abort criteria
    protocol = TradingProtocol(
        duration_days=30,
        initial_equity=100000.0,
    )

    # Simulated regime sequence (realistic market conditions)
    regimes = [
        "trend_low_vol",    # Week 1: Good conditions
        "trend_low_vol",
        "trend_low_vol",
        "trend_high_vol",   # Mid-week volatility
        "trend_low_vol",
        "trend_low_vol",
        "trend_low_vol",
        "range_low_vol",    # Week 2: Ranging market
        "range_low_vol",
        "range_low_vol",
        "trend_low_vol",
        "trend_low_vol",
        "trend_high_vol",
        "trend_high_vol",
        "trend_low_vol",    # Week 3: Back to trending
        "trend_low_vol",
        "trend_low_vol",
        "trend_low_vol",
        "range_high_vol",   # Brief bad conditions
        "trend_low_vol",
        "trend_low_vol",
        "trend_low_vol",    # Week 4: Finish strong
        "trend_low_vol",
        "trend_low_vol",
        "trend_low_vol",
        "trend_high_vol",
        "trend_low_vol",
        "trend_low_vol",
        "trend_low_vol",
        "trend_low_vol",
    ]

    # Start the protocol
    protocol.start()
    print(f"Protocol started: {protocol.start_date}")
    print(f"Duration: {protocol.duration} days")
    print()

    # Run 30 days
    current_equity = protocol.initial_equity

    for day_num in range(30):
        regime = regimes[day_num]

        # Generate simulated day
        day_result = generate_simulated_day(
            day_num=day_num,
            current_equity=current_equity,
            regime=regime,
            base_daily_return=0.002,  # 0.2% average daily return
            volatility=0.008,
        )

        # Record the day
        protocol.record_day(day_result)
        current_equity = day_result.ending_equity

        # Check abort conditions
        should_abort, abort_reason = protocol.check_abort_conditions()
        if should_abort:
            print(f"⚠️  ABORT TRIGGERED on Day {day_num + 1}: {abort_reason}")
            protocol.abort(abort_reason)
            break

        # Run checkpoints
        if day_num + 1 in TradingProtocol.CHECKPOINT_DAYS:
            print(f"\n--- Day {day_num + 1} Checkpoint ---")
            checkpoint = protocol.run_checkpoint(day_num + 1)
            print(f"  Cumulative Return: {checkpoint.cumulative_return:.2%}")
            print(f"  Max Drawdown: {checkpoint.max_drawdown:.2%}")
            print(f"  Win Rate: {checkpoint.win_rate:.1%}")
            print(f"  Rolling Sharpe: {checkpoint.sharpe_rolling:.2f}")
            print(f"  Total Trades: {checkpoint.total_trades}")

            if checkpoint.issues_identified:
                print(f"  Issues: {', '.join(checkpoint.issues_identified)}")

            print(f"  Recommendation: {checkpoint.recommendation.upper()}")

            if checkpoint.recommendation == "abort":
                print(f"\n⚠️  CHECKPOINT RECOMMENDS ABORT")
                protocol.abort(f"Day {day_num + 1} checkpoint: {checkpoint.issues_identified}")
                break

    # Complete if not aborted
    if protocol.status == ProtocolStatus.RUNNING:
        protocol.complete()

    # Generate final report
    print("\n" + "=" * 70)
    print("FINAL PROTOCOL REPORT")
    print("=" * 70)

    report = protocol.final_report()

    print(f"\nStatus: {report['status'].upper()}")
    print(f"Duration: {report['duration_days']} days")
    print(f"Start Date: {report['start_date']}")
    print(f"End Date: {report['end_date']}")
    print()
    print("--- Performance ---")
    print(f"  Initial Equity: ${report['initial_equity']:,.2f}")
    print(f"  Final Equity: ${report['final_equity']:,.2f}")
    print(f"  Total Return: {report['total_return']:.2%}")
    print(f"  Max Drawdown: {report['max_drawdown']:.2%}")
    print(f"  Win Rate: {report['win_rate']:.1%}")
    print(f"  Sharpe Ratio: {report['sharpe_ratio']:.2f}")
    print()
    print("--- Trading Activity ---")
    print(f"  Total Trades: {report['total_trades']}")
    print(f"  Circuit Breaker Triggers: {report['circuit_breaker_triggers']}")

    if report['abort_reason']:
        print(f"\n⚠️  Abort Reason: {report['abort_reason']}")

    # Compare to backtest (simulated backtest metrics)
    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON (vs Backtest)")
    print("=" * 70)

    # Simulated backtest metrics (what we expected)
    backtest_metrics = BacktestMetrics(
        total_return=0.10,      # 10% expected
        annual_return=0.10,
        max_drawdown=0.08,      # 8% expected max DD
        sharpe_ratio=1.5,       # Good Sharpe expected
        sortino_ratio=2.0,
        calmar_ratio=1.25,
        total_trades=15,
        win_rate=0.55,
        profit_factor=1.5,
        avg_win=0.02,
        avg_loss=-0.01,
        avg_swap_received=50.0,
        total_swap_profit=750.0,
        swap_contribution_pct=0.15,
    )

    # Create paper trading metrics from protocol results
    paper_metrics = BacktestMetrics(
        total_return=report['total_return'],
        annual_return=report['total_return'],  # Short period
        max_drawdown=report['max_drawdown'],
        sharpe_ratio=report['sharpe_ratio'],
        sortino_ratio=report['sharpe_ratio'] * 1.2,  # Approximate
        calmar_ratio=report['total_return'] / max(report['max_drawdown'], 0.01),
        total_trades=report['total_trades'],
        win_rate=report['win_rate'],
        profit_factor=1.5,  # Approximate
        avg_win=0.02,
        avg_loss=-0.01,
        avg_swap_received=45.0,
        total_swap_profit=675.0,
        swap_contribution_pct=0.12,
    )

    # Run validation comparison
    validator = PaperTradingValidator(
        criteria=ValidationCriteria(
            max_return_degradation=0.30,  # Allow 30% worse return
            max_sharpe_degradation=0.25,  # Allow 25% worse Sharpe
            max_dd_increase=0.20,         # Allow 20% more drawdown
        )
    )

    validation = validator.compare(backtest_metrics, paper_metrics)
    validation_report = validator.generate_report(validation)

    print()
    print(validation_report)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if validation.is_valid and protocol.status == ProtocolStatus.COMPLETED:
        print("\n✅ STRATEGY VALIDATED")
        print("   Paper trading results are within acceptable bounds.")
        print("   Strategy is ready for next stage of evaluation.")
    elif protocol.status == ProtocolStatus.ABORTED:
        print("\n❌ PROTOCOL ABORTED")
        print(f"   Reason: {protocol.abort_reason}")
        print("   Strategy needs adjustment before proceeding.")
    else:
        print("\n⚠️  VALIDATION FAILED")
        print("   Paper trading results degraded too much from backtest.")
        print("   Review failure reasons and adjust strategy.")
        for reason in validation.failure_reasons:
            print(f"   - {reason}")

    return protocol, validation


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    protocol, validation = run_validation_protocol()
