"""30-day paper trading protocol for strategy validation.

PROTOCOL PHILOSOPHY:
====================
Paper trading for at least 30 days allows you to:
1. See the strategy in different market conditions
2. Verify execution works as expected
3. Build confidence before real money
4. Catch bugs and edge cases

CHECKPOINTS:
============
Day 7:  Early sanity check - is anything obviously wrong?
Day 14: Mid-point review - is performance tracking expectations?
Day 21: Final stretch - any concerns before completion?
Day 30: Final evaluation - pass/fail decision

AUTO-ABORT CONDITIONS:
======================
- Drawdown > 15% (capital preservation)
- 5+ consecutive losing days (something is wrong)
- Circuit breaker triggered 3+ times (strategy too risky)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, Callable

import pandas as pd
import numpy as np


class ProtocolStatus(Enum):
    """Status of the trading protocol."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    DEGRADED = "degraded"  # Defensive mode: manage existing, no new entries
    PAUSED = "paused"
    ABORTED = "aborted"
    COMPLETED = "completed"
    VALIDATED = "validated"  # Completed and passed validation


@dataclass
class ProtocolDay:
    """Results for one day of paper trading."""

    date: date
    starting_equity: float
    ending_equity: float
    daily_pnl: float
    daily_return: float
    trades_opened: int
    trades_closed: int
    max_drawdown_today: float
    regime: str  # Composite regime
    circuit_breaker_triggered: bool
    notes: str = ""

    @property
    def is_profitable(self) -> bool:
        """Check if day was profitable."""
        return self.daily_pnl > 0


@dataclass
class ProtocolCheckpoint:
    """Checkpoint during protocol (days 7, 14, 21, 30)."""

    day: int
    date: date
    cumulative_return: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    sharpe_rolling: float
    total_trades: int
    issues_identified: list[str]
    recommendation: str  # "continue", "pause", "abort"

    @property
    def is_healthy(self) -> bool:
        """Check if checkpoint indicates healthy performance."""
        return self.recommendation == "continue"


@dataclass
class DegradedCriteria:
    """Softer thresholds that trigger DEGRADED mode (warning layer).

    These fire *before* AbortCriteria and shift the protocol into a
    defensive posture: existing positions are managed (stops trail,
    exits fire) but no new entries are allowed.
    """

    max_drawdown: float = 0.10  # 10% drawdown (abort at 15%)
    consecutive_losing_days: int = 3  # 3 losing days (abort at 5)
    circuit_breaker_count: int = 1  # 1 CB trigger in a day (abort at 3 cumulative)
    min_win_rate: float = 0.30  # 30% win rate warning (abort at 20%)


@dataclass
class RecoveryCriteria:
    """Conditions to recover from DEGRADED back to RUNNING."""

    max_drawdown: float = 0.08  # Must improve below 8%
    consecutive_profitable_days: int = 1  # At least 1 profitable day
    no_circuit_breaker_days: int = 1  # 1 full day without CB triggers


@dataclass
class AbortCriteria:
    """Criteria for auto-aborting the protocol."""

    max_drawdown: float = 0.15  # 15% max drawdown
    consecutive_losing_days: int = 5  # 5 losing days in a row
    circuit_breaker_count: int = 3  # 3 circuit breaker triggers
    min_win_rate: float = 0.20  # 20% minimum win rate after 10+ trades


class TradingProtocol:
    """Manages 30-day paper trading validation protocol.

    Example:
        protocol = TradingProtocol(duration_days=30)
        protocol.start()

        for each_day in trading_days:
            day_result = run_paper_trading_day()
            protocol.record_day(day_result)

            abort, reason = protocol.check_abort_conditions()
            if abort:
                print(f"Protocol aborted: {reason}")
                break

            if each_day.day in [7, 14, 21, 30]:
                checkpoint = protocol.run_checkpoint(each_day.day)
                print(checkpoint)

        final = protocol.final_report()
    """

    CHECKPOINT_DAYS = [7, 14, 21, 30]

    def __init__(
        self,
        duration_days: int = 30,
        abort_criteria: AbortCriteria | None = None,
        degraded_criteria: DegradedCriteria | None = None,
        recovery_criteria: RecoveryCriteria | None = None,
        initial_equity: float = 100000.0,
    ) -> None:
        self.duration = duration_days
        self.abort_criteria = abort_criteria or AbortCriteria()
        self.degraded_criteria = degraded_criteria or DegradedCriteria()
        self.recovery_criteria = recovery_criteria or RecoveryCriteria()
        self.initial_equity = initial_equity

        self.days: list[ProtocolDay] = []
        self.checkpoints: list[ProtocolCheckpoint] = []
        self.status = ProtocolStatus.NOT_STARTED
        self.start_date: Optional[date] = None
        self.end_date: Optional[date] = None
        self.abort_reason: Optional[str] = None

        # Degraded mode tracking
        self.degraded_reason: Optional[str] = None
        self.degraded_since: Optional[datetime] = None

    def start(self) -> None:
        """Start the protocol."""
        self.status = ProtocolStatus.RUNNING
        self.start_date = date.today()
        self.days = []
        self.checkpoints = []
        self.abort_reason = None

    def record_day(self, day_result: ProtocolDay) -> None:
        """Record a day's results.

        If a record for the same date already exists, it is replaced
        (last write wins) to prevent duplicate day counts.
        """
        if self.status not in (ProtocolStatus.RUNNING, ProtocolStatus.DEGRADED):
            raise ValueError(f"Cannot record day when status is {self.status}")

        # Replace existing day if same date (prevent duplicates)
        for i, existing in enumerate(self.days):
            if existing.date == day_result.date:
                self.days[i] = day_result
                return

        self.days.append(day_result)

    def check_abort_conditions(self) -> tuple[bool, Optional[str]]:
        """Check if protocol should abort early.

        Returns:
            Tuple of (should_abort, reason).
        """
        if len(self.days) == 0:
            return False, None

        # 1. Check max drawdown
        current_equity = self.days[-1].ending_equity
        peak_equity = max(d.ending_equity for d in self.days)
        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown > self.abort_criteria.max_drawdown:
            return True, f"Max drawdown exceeded: {drawdown:.1%}"

        # 2. Check consecutive losing days
        # Only count days with actual negative PnL (daily_pnl < 0).
        # Days with zero PnL (no activity / break-even / swap-only)
        # are neutral and should not count toward the streak.
        if len(self.days) >= self.abort_criteria.consecutive_losing_days:
            recent = self.days[-self.abort_criteria.consecutive_losing_days :]
            if all(d.daily_pnl < 0 for d in recent):
                return True, f"{len(recent)} consecutive losing days"

        # 3. Check circuit breaker count
        cb_count = sum(1 for d in self.days if d.circuit_breaker_triggered)
        if cb_count >= self.abort_criteria.circuit_breaker_count:
            return True, f"Circuit breaker triggered {cb_count} times"

        # 4. Check win rate after enough trades
        total_trades = sum(d.trades_closed for d in self.days)
        if total_trades >= 10:
            profitable_days = sum(1 for d in self.days if d.is_profitable)
            win_rate = profitable_days / len(self.days)
            if win_rate < self.abort_criteria.min_win_rate:
                return True, f"Win rate too low: {win_rate:.1%}"

        return False, None

    def check_degraded_conditions(self) -> tuple[bool, Optional[str]]:
        """Check if protocol should enter DEGRADED mode.

        These are softer thresholds that fire before abort. They act as
        an early-warning layer: block new entries while managing exits.

        Returns:
            Tuple of (should_degrade, reason).
        """
        if len(self.days) == 0:
            return False, None

        # Already degraded or worse — skip
        if self.status not in (ProtocolStatus.RUNNING,):
            return False, None

        criteria = self.degraded_criteria

        # 1. Drawdown warning (10%)
        current_equity = self.days[-1].ending_equity
        peak_equity = max(d.ending_equity for d in self.days)
        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown > criteria.max_drawdown:
            return (
                True,
                f"Drawdown warning: {drawdown:.1%} > {criteria.max_drawdown:.0%}",
            )

        # 2. Consecutive losing days (3)
        if len(self.days) >= criteria.consecutive_losing_days:
            recent = self.days[-criteria.consecutive_losing_days :]
            if all(d.daily_pnl < 0 for d in recent):
                return True, f"{len(recent)} consecutive losing days (warning)"

        # 3. Circuit breaker triggered today
        cb_count = sum(1 for d in self.days if d.circuit_breaker_triggered)
        if cb_count >= criteria.circuit_breaker_count:
            return True, f"Circuit breaker triggered {cb_count} times (warning)"

        # 4. Low win rate (30%)
        total_trades = sum(d.trades_closed for d in self.days)
        if total_trades >= 10:
            profitable_days = sum(1 for d in self.days if d.is_profitable)
            win_rate = profitable_days / len(self.days)
            if win_rate < criteria.min_win_rate:
                return True, f"Win rate low: {win_rate:.1%} (warning)"

        return False, None

    def check_recovery_conditions(self) -> tuple[bool, Optional[str]]:
        """Check if protocol can recover from DEGRADED to RUNNING.

        Returns:
            Tuple of (can_recover, reason).
        """
        if self.status != ProtocolStatus.DEGRADED:
            return False, None

        if len(self.days) == 0:
            return False, None

        criteria = self.recovery_criteria

        # 1. Drawdown must improve below recovery threshold (8%)
        current_equity = self.days[-1].ending_equity
        peak_equity = max(d.ending_equity for d in self.days)
        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown > criteria.max_drawdown:
            return False, None

        # 2. At least N consecutive profitable days since degradation
        if len(self.days) >= criteria.consecutive_profitable_days:
            recent = self.days[-criteria.consecutive_profitable_days :]
            if not all(d.daily_pnl >= 0 for d in recent):
                return False, None
        else:
            return False, None

        # 3. No circuit breaker triggers in last N days
        if len(self.days) >= criteria.no_circuit_breaker_days:
            recent_cb = self.days[-criteria.no_circuit_breaker_days :]
            if any(d.circuit_breaker_triggered for d in recent_cb):
                return False, None
        else:
            return False, None

        return True, (
            f"Recovery: drawdown={drawdown:.1%}, "
            f"last {criteria.consecutive_profitable_days} day(s) profitable, "
            f"no CB in {criteria.no_circuit_breaker_days} day(s)"
        )

    def degrade(self, reason: str) -> None:
        """Transition protocol to DEGRADED mode.

        Args:
            reason: Why the protocol entered degraded mode.
        """
        self.status = ProtocolStatus.DEGRADED
        self.degraded_reason = reason
        self.degraded_since = datetime.now()

    def recover(self) -> None:
        """Recover from DEGRADED back to RUNNING."""
        self.status = ProtocolStatus.RUNNING
        self.degraded_reason = None
        self.degraded_since = None

    def run_checkpoint(self, day_num: int) -> ProtocolCheckpoint:
        """Run checkpoint analysis.

        Args:
            day_num: Day number (7, 14, 21, or 30).

        Returns:
            ProtocolCheckpoint with analysis.
        """
        if day_num not in self.CHECKPOINT_DAYS:
            raise ValueError(f"Invalid checkpoint day: {day_num}")

        if len(self.days) < day_num:
            raise ValueError(f"Not enough days recorded for day {day_num} checkpoint")

        # Calculate metrics
        days_to_check = self.days[:day_num]

        cumulative_return = (
            days_to_check[-1].ending_equity - self.initial_equity
        ) / self.initial_equity

        peak_equity = max(d.ending_equity for d in days_to_check)
        current_equity = days_to_check[-1].ending_equity
        max_drawdown = max(
            (peak_equity - d.ending_equity) / peak_equity for d in days_to_check
        )
        current_drawdown = (peak_equity - current_equity) / peak_equity

        profitable_days = sum(1 for d in days_to_check if d.is_profitable)
        win_rate = profitable_days / len(days_to_check)

        # Calculate rolling Sharpe
        returns = [d.daily_return for d in days_to_check]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        total_trades = sum(d.trades_closed for d in days_to_check)

        # Identify issues
        issues = []
        if cumulative_return < -0.05:
            issues.append(f"Negative return: {cumulative_return:.1%}")
        if max_drawdown > 0.10:
            issues.append(f"High drawdown: {max_drawdown:.1%}")
        if win_rate < 0.40:
            issues.append(f"Low win rate: {win_rate:.1%}")
        if sharpe < 0:
            issues.append(f"Negative Sharpe: {sharpe:.2f}")

        cb_count = sum(1 for d in days_to_check if d.circuit_breaker_triggered)
        if cb_count > 0:
            issues.append(f"Circuit breaker triggered {cb_count}x")

        # Determine recommendation
        if max_drawdown > self.abort_criteria.max_drawdown:
            recommendation = "abort"
        elif len(issues) >= 3:
            recommendation = "pause"
        elif cumulative_return < -0.10:
            recommendation = "abort"
        else:
            recommendation = "continue"

        checkpoint = ProtocolCheckpoint(
            day=day_num,
            date=days_to_check[-1].date,
            cumulative_return=cumulative_return,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            sharpe_rolling=sharpe,
            total_trades=total_trades,
            issues_identified=issues,
            recommendation=recommendation,
        )

        self.checkpoints.append(checkpoint)
        return checkpoint

    def abort(self, reason: str) -> None:
        """Abort the protocol."""
        self.status = ProtocolStatus.ABORTED
        self.abort_reason = reason
        self.end_date = date.today()

    def complete(self) -> None:
        """Mark protocol as completed."""
        if len(self.days) < self.duration:
            raise ValueError(
                f"Cannot complete: only {len(self.days)}/{self.duration} days recorded"
            )

        self.status = ProtocolStatus.COMPLETED
        self.end_date = self.days[-1].date

    def final_report(self) -> dict:
        """Generate final protocol report.

        Returns:
            Dictionary with complete protocol results.
        """
        if len(self.days) == 0:
            return {"status": self.status.value, "error": "No data recorded"}

        # Calculate final metrics
        final_equity = self.days[-1].ending_equity
        total_return = (final_equity - self.initial_equity) / self.initial_equity

        peak_equity = max(d.ending_equity for d in self.days)
        max_drawdown = max(
            (peak_equity - d.ending_equity) / peak_equity for d in self.days
        )

        profitable_days = sum(1 for d in self.days if d.is_profitable)
        win_rate = profitable_days / len(self.days)

        returns = [d.daily_return for d in self.days]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        total_trades = sum(d.trades_closed for d in self.days)
        cb_triggers = sum(1 for d in self.days if d.circuit_breaker_triggered)

        return {
            "status": self.status.value,
            "duration_days": len(self.days),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_equity": self.initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "total_trades": total_trades,
            "circuit_breaker_triggers": cb_triggers,
            "abort_reason": self.abort_reason,
            "checkpoints": [
                {
                    "day": c.day,
                    "return": c.cumulative_return,
                    "drawdown": c.max_drawdown,
                    "recommendation": c.recommendation,
                }
                for c in self.checkpoints
            ],
        }

    def get_progress(self) -> dict:
        """Get current protocol progress.

        Returns:
            Dictionary with progress information.
        """
        if len(self.days) == 0:
            return {
                "status": self.status.value,
                "days_completed": 0,
                "days_remaining": self.duration,
                "progress_pct": 0.0,
            }

        current_equity = self.days[-1].ending_equity
        current_return = (current_equity - self.initial_equity) / self.initial_equity

        return {
            "status": self.status.value,
            "days_completed": len(self.days),
            "days_remaining": max(0, self.duration - len(self.days)),
            "progress_pct": len(self.days) / self.duration * 100,
            "current_equity": current_equity,
            "current_return": current_return,
            "next_checkpoint": self._next_checkpoint(),
            "degraded_reason": self.degraded_reason,
        }

    def _next_checkpoint(self) -> Optional[int]:
        """Get next checkpoint day."""
        days_done = len(self.days)
        for checkpoint in self.CHECKPOINT_DAYS:
            if checkpoint > days_done:
                return checkpoint
        return None
