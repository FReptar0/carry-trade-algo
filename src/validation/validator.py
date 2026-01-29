"""Validation tools for comparing backtest to paper trading.

VALIDATION PHILOSOPHY:
======================
Backtests are OPTIMISTIC. They don't account for:
1. Slippage worse than expected
2. Fills at wrong prices
3. Missed signals due to latency
4. Execution failures

Paper trading provides a reality check before real money.

VALIDATION CRITERIA:
====================
A strategy is considered "validated" if paper trading results are
within acceptable degradation thresholds of backtest results.

Typical acceptable degradation:
- Return: 30% worse (if backtest = 10%, paper >= 7%)
- Sharpe: 25% worse
- Max Drawdown: 20% increase

If degradation exceeds thresholds, strategy needs tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from src.backtest.metrics import BacktestMetrics


@dataclass
class ValidationCriteria:
    """Thresholds for validation pass/fail.

    Defaults are reasonable for most strategies.
    Adjust based on strategy characteristics and risk tolerance.
    """

    max_return_degradation: float = 0.30  # 30% max return degradation
    max_sharpe_degradation: float = 0.25  # 25% max Sharpe degradation
    max_dd_increase: float = 0.20  # 20% max increase in drawdown
    min_fill_rate: float = 0.95  # 95% minimum order fill rate
    max_slippage_cost: float = 0.02  # 2% max slippage impact on returns
    max_signal_delay_ms: float = 500  # 500ms max signal-to-fill delay


@dataclass
class ValidationMetrics:
    """Comparison between backtest and paper trading results."""

    # Performance comparison
    backtest_return: float
    paper_return: float
    return_degradation: float  # (backtest - paper) / |backtest|

    backtest_sharpe: float
    paper_sharpe: float
    sharpe_degradation: float

    backtest_max_dd: float
    paper_max_dd: float
    dd_increase: float  # (paper - backtest) / backtest

    # Execution metrics
    total_orders: int
    filled_orders: int
    fill_rate: float
    avg_slippage_pips: float
    slippage_cost_pct: float

    # Timing metrics
    avg_signal_to_fill_ms: float
    missed_signals: int

    # Trade comparison
    backtest_trades: int
    paper_trades: int
    trade_count_match: bool

    # Overall validation
    is_valid: bool
    failure_reasons: list[str]

    @property
    def summary(self) -> str:
        """Get summary string."""
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"Validation {status}\n"
            f"  Return: {self.backtest_return:.2%} -> {self.paper_return:.2%} "
            f"(degradation: {self.return_degradation:.1%})\n"
            f"  Sharpe: {self.backtest_sharpe:.2f} -> {self.paper_sharpe:.2f} "
            f"(degradation: {self.sharpe_degradation:.1%})\n"
            f"  MaxDD: {self.backtest_max_dd:.2%} -> {self.paper_max_dd:.2%}\n"
            f"  Fill Rate: {self.fill_rate:.1%}\n"
            f"  Slippage Cost: {self.slippage_cost_pct:.2%}"
        )


@dataclass
class PaperTradingResult:
    """Results from paper trading session."""

    equity_df: pd.DataFrame
    trades_df: pd.DataFrame
    orders_df: pd.DataFrame
    metrics: BacktestMetrics

    # Execution stats
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    avg_slippage_pips: float = 0.0
    avg_fill_delay_ms: float = 0.0


class PaperTradingValidator:
    """Validates paper trading results against backtest.

    Example:
        validator = PaperTradingValidator()

        metrics = validator.compare(
            backtest_result=bt_result,
            paper_result=paper_result,
        )

        if metrics.is_valid:
            print("Strategy validated!")
        else:
            print(f"Validation failed: {metrics.failure_reasons}")
    """

    def __init__(
        self,
        criteria: ValidationCriteria | None = None,
    ) -> None:
        self.criteria = criteria or ValidationCriteria()

    def compare(
        self,
        backtest_metrics: BacktestMetrics,
        paper_metrics: BacktestMetrics,
        paper_orders: Optional[pd.DataFrame] = None,
    ) -> ValidationMetrics:
        """Compare backtest to paper trading results.

        Args:
            backtest_metrics: Metrics from backtest.
            paper_metrics: Metrics from paper trading.
            paper_orders: Optional order data for execution analysis.

        Returns:
            ValidationMetrics with comparison and pass/fail status.
        """
        # Calculate performance degradation
        return_degradation = self._calc_degradation(
            backtest_metrics.total_return,
            paper_metrics.total_return,
        )

        sharpe_degradation = self._calc_degradation(
            backtest_metrics.sharpe_ratio,
            paper_metrics.sharpe_ratio,
        )

        dd_increase = self._calc_increase(
            backtest_metrics.max_drawdown,
            paper_metrics.max_drawdown,
        )

        # Execution metrics (if order data available)
        if paper_orders is not None and len(paper_orders) > 0:
            total_orders = len(paper_orders)
            filled_orders = len(paper_orders[paper_orders["status"] == "FILLED"])
            fill_rate = filled_orders / total_orders if total_orders > 0 else 1.0

            if "slippage" in paper_orders.columns:
                avg_slippage = paper_orders["slippage"].mean()
            else:
                avg_slippage = 0.0

            if "fill_delay_ms" in paper_orders.columns:
                avg_delay = paper_orders["fill_delay_ms"].mean()
            else:
                avg_delay = 0.0

            slippage_cost = avg_slippage * total_orders / 10000  # Rough estimate
        else:
            total_orders = paper_metrics.total_trades
            filled_orders = paper_metrics.total_trades
            fill_rate = 1.0
            avg_slippage = 0.0
            avg_delay = 0.0
            slippage_cost = 0.0

        # Check validation criteria
        failure_reasons = []

        if return_degradation > self.criteria.max_return_degradation:
            failure_reasons.append(
                f"Return degradation {return_degradation:.1%} > "
                f"{self.criteria.max_return_degradation:.1%} threshold"
            )

        if sharpe_degradation > self.criteria.max_sharpe_degradation:
            failure_reasons.append(
                f"Sharpe degradation {sharpe_degradation:.1%} > "
                f"{self.criteria.max_sharpe_degradation:.1%} threshold"
            )

        if dd_increase > self.criteria.max_dd_increase:
            failure_reasons.append(
                f"Drawdown increase {dd_increase:.1%} > "
                f"{self.criteria.max_dd_increase:.1%} threshold"
            )

        if fill_rate < self.criteria.min_fill_rate:
            failure_reasons.append(
                f"Fill rate {fill_rate:.1%} < "
                f"{self.criteria.min_fill_rate:.1%} threshold"
            )

        is_valid = len(failure_reasons) == 0

        return ValidationMetrics(
            backtest_return=backtest_metrics.total_return,
            paper_return=paper_metrics.total_return,
            return_degradation=return_degradation,
            backtest_sharpe=backtest_metrics.sharpe_ratio,
            paper_sharpe=paper_metrics.sharpe_ratio,
            sharpe_degradation=sharpe_degradation,
            backtest_max_dd=backtest_metrics.max_drawdown,
            paper_max_dd=paper_metrics.max_drawdown,
            dd_increase=dd_increase,
            total_orders=total_orders,
            filled_orders=filled_orders,
            fill_rate=fill_rate,
            avg_slippage_pips=avg_slippage,
            slippage_cost_pct=slippage_cost,
            avg_signal_to_fill_ms=avg_delay,
            missed_signals=total_orders - filled_orders,
            backtest_trades=backtest_metrics.total_trades,
            paper_trades=paper_metrics.total_trades,
            trade_count_match=abs(
                backtest_metrics.total_trades - paper_metrics.total_trades
            )
            <= 2,
            is_valid=is_valid,
            failure_reasons=failure_reasons,
        )

    def _calc_degradation(self, backtest_val: float, paper_val: float) -> float:
        """Calculate performance degradation percentage.

        Degradation = how much worse paper is vs backtest.
        Positive = paper is worse (bad)
        Negative = paper is better (good but suspicious)
        """
        if abs(backtest_val) < 0.001:
            # Backtest was near zero, use absolute difference
            return abs(paper_val - backtest_val)

        return (backtest_val - paper_val) / abs(backtest_val)

    def _calc_increase(self, backtest_val: float, paper_val: float) -> float:
        """Calculate increase in a metric (like drawdown).

        Increase = how much worse paper is vs backtest.
        For drawdown, higher is worse.
        """
        if abs(backtest_val) < 0.001:
            return abs(paper_val)

        return (paper_val - backtest_val) / abs(backtest_val)

    def generate_report(
        self,
        metrics: ValidationMetrics,
    ) -> str:
        """Generate detailed validation report.

        Args:
            metrics: Validation metrics to report.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "PAPER TRADING VALIDATION REPORT",
            "=" * 60,
            "",
            f"Overall Status: {'PASS' if metrics.is_valid else 'FAIL'}",
            "",
            "--- Performance Comparison ---",
            f"  Total Return:",
            f"    Backtest:    {metrics.backtest_return:>8.2%}",
            f"    Paper:       {metrics.paper_return:>8.2%}",
            f"    Degradation: {metrics.return_degradation:>8.1%}",
            "",
            f"  Sharpe Ratio:",
            f"    Backtest:    {metrics.backtest_sharpe:>8.2f}",
            f"    Paper:       {metrics.paper_sharpe:>8.2f}",
            f"    Degradation: {metrics.sharpe_degradation:>8.1%}",
            "",
            f"  Max Drawdown:",
            f"    Backtest:    {metrics.backtest_max_dd:>8.2%}",
            f"    Paper:       {metrics.paper_max_dd:>8.2%}",
            f"    Increase:    {metrics.dd_increase:>8.1%}",
            "",
            "--- Execution Quality ---",
            f"  Orders:        {metrics.filled_orders}/{metrics.total_orders} filled "
            f"({metrics.fill_rate:.1%})",
            f"  Avg Slippage:  {metrics.avg_slippage_pips:.2f} pips",
            f"  Slippage Cost: {metrics.slippage_cost_pct:.2%}",
            f"  Avg Fill Delay:{metrics.avg_signal_to_fill_ms:.0f} ms",
            "",
            "--- Trade Count ---",
            f"  Backtest:      {metrics.backtest_trades}",
            f"  Paper:         {metrics.paper_trades}",
            f"  Match:         {'Yes' if metrics.trade_count_match else 'No'}",
        ]

        if not metrics.is_valid:
            lines.extend([
                "",
                "--- Failure Reasons ---",
            ])
            for reason in metrics.failure_reasons:
                lines.append(f"  - {reason}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
