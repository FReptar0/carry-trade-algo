"""Circuit breakers (kill switches) for risk management.

Circuit breakers HALT trading when losses exceed predefined limits.
They prevent catastrophic losses during adverse market conditions.

Think of them like the circuit breakers in your home - they trip to
prevent damage when there's too much current (or in our case, losses).

Limits we enforce:
1. DAILY LOSS LIMIT: Stop trading if we lose X% today
2. WEEKLY LOSS LIMIT: Stop trading if we lose Y% this week
3. MAX DRAWDOWN: Stop trading if total drawdown exceeds Z%
4. MAX POSITIONS: Don't open more than N positions simultaneously
5. MAX EXPOSURE PER PAIR: Don't put more than P% in one currency pair

When a limit is violated:
- New entries are blocked
- Optionally, existing positions can be closed
- Alert is generated
- Trading resumes when conditions normalize (or manually)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable

import numpy as np
import pandas as pd


class LimitType(Enum):
    """Type of risk limit."""
    DAILY_LOSS = auto()
    WEEKLY_LOSS = auto()
    MAX_DRAWDOWN = auto()
    MAX_POSITIONS = auto()
    MAX_PAIR_EXPOSURE = auto()


@dataclass
class LimitViolation:
    """Record of a limit violation."""
    limit_type: LimitType
    timestamp: datetime
    current_value: float
    limit_value: float
    message: str

    @property
    def severity(self) -> str:
        """How far over the limit are we?"""
        if self.limit_type in (LimitType.MAX_POSITIONS, LimitType.MAX_PAIR_EXPOSURE):
            over_pct = (self.current_value / self.limit_value - 1) * 100
        else:
            over_pct = (abs(self.current_value) / abs(self.limit_value) - 1) * 100

        if over_pct < 10:
            return "WARNING"
        elif over_pct < 25:
            return "HIGH"
        else:
            return "CRITICAL"


@dataclass
class RiskLimits:
    """Configuration for risk limits.

    All percentages are expressed as decimals (0.03 = 3%).
    """
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_weekly_loss_pct: float = 0.07  # 7% weekly loss limit
    max_drawdown_pct: float = 0.20  # 20% maximum drawdown
    max_positions: int = 3  # Maximum simultaneous positions
    max_exposure_per_pair_pct: float = 0.15  # 15% max in one pair
    close_positions_on_breach: bool = False  # Auto-close on breach?
    cooldown_hours: int = 24  # Hours to wait before re-enabling trading


@dataclass
class PortfolioState:
    """Current state of the portfolio for limit checking."""
    equity: float
    high_water_mark: float  # Peak equity value
    daily_pnl: float  # Today's P&L
    weekly_pnl: float  # This week's P&L
    open_positions: int  # Number of open positions
    exposure_by_pair: dict[str, float] = field(default_factory=dict)  # Pair -> exposure %


class CircuitBreaker:
    """Monitor portfolio state and trigger when limits are breached.

    Example:
        >>> limits = RiskLimits(max_daily_loss_pct=0.03)
        >>> breaker = CircuitBreaker(limits)
        >>> state = PortfolioState(equity=98000, high_water_mark=100000, daily_pnl=-3500, ...)
        >>> breaker.check_limits(state)
        >>> if breaker.is_trading_halted:
        ...     print("Trading halted!")
    """

    def __init__(
        self,
        limits: RiskLimits,
        alert_callback: Callable[[LimitViolation], None] | None = None,
    ) -> None:
        self.limits = limits
        self.alert_callback = alert_callback
        self.violations: list[LimitViolation] = []
        self.trading_halted = False
        self.halt_timestamp: datetime | None = None
        self.initial_equity: float | None = None

    @property
    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted."""
        if not self.trading_halted:
            return False

        # Check if cooldown has passed
        if self.halt_timestamp and self.limits.cooldown_hours > 0:
            cooldown_end = self.halt_timestamp + timedelta(hours=self.limits.cooldown_hours)
            if datetime.now() >= cooldown_end:
                self.trading_halted = False
                return False

        return self.trading_halted

    def set_initial_equity(self, equity: float) -> None:
        """Set initial equity for drawdown calculation."""
        self.initial_equity = equity

    def check_limits(
        self,
        state: PortfolioState,
        timestamp: datetime | None = None,
    ) -> list[LimitViolation]:
        """Check all risk limits against current portfolio state.

        Args:
            state: Current portfolio state.
            timestamp: Timestamp for violation records.

        Returns:
            List of any limit violations found.
        """
        timestamp = timestamp or datetime.now()
        violations = []

        # Daily loss limit
        if state.daily_pnl < 0:
            daily_loss_pct = abs(state.daily_pnl) / (state.equity + abs(state.daily_pnl))
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                violations.append(LimitViolation(
                    limit_type=LimitType.DAILY_LOSS,
                    timestamp=timestamp,
                    current_value=daily_loss_pct,
                    limit_value=self.limits.max_daily_loss_pct,
                    message=f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.limits.max_daily_loss_pct:.1%}",
                ))

        # Weekly loss limit
        if state.weekly_pnl < 0:
            weekly_loss_pct = abs(state.weekly_pnl) / (state.equity + abs(state.weekly_pnl))
            if weekly_loss_pct >= self.limits.max_weekly_loss_pct:
                violations.append(LimitViolation(
                    limit_type=LimitType.WEEKLY_LOSS,
                    timestamp=timestamp,
                    current_value=weekly_loss_pct,
                    limit_value=self.limits.max_weekly_loss_pct,
                    message=f"Weekly loss {weekly_loss_pct:.1%} exceeds limit {self.limits.max_weekly_loss_pct:.1%}",
                ))

        # Maximum drawdown
        drawdown_pct = (state.high_water_mark - state.equity) / state.high_water_mark
        if drawdown_pct >= self.limits.max_drawdown_pct:
            violations.append(LimitViolation(
                limit_type=LimitType.MAX_DRAWDOWN,
                timestamp=timestamp,
                current_value=drawdown_pct,
                limit_value=self.limits.max_drawdown_pct,
                message=f"Drawdown {drawdown_pct:.1%} exceeds limit {self.limits.max_drawdown_pct:.1%}",
            ))

        # Maximum positions
        if state.open_positions > self.limits.max_positions:
            violations.append(LimitViolation(
                limit_type=LimitType.MAX_POSITIONS,
                timestamp=timestamp,
                current_value=float(state.open_positions),
                limit_value=float(self.limits.max_positions),
                message=f"{state.open_positions} positions exceeds limit of {self.limits.max_positions}",
            ))

        # Maximum exposure per pair
        for pair, exposure in state.exposure_by_pair.items():
            if exposure > self.limits.max_exposure_per_pair_pct:
                violations.append(LimitViolation(
                    limit_type=LimitType.MAX_PAIR_EXPOSURE,
                    timestamp=timestamp,
                    current_value=exposure,
                    limit_value=self.limits.max_exposure_per_pair_pct,
                    message=f"{pair} exposure {exposure:.1%} exceeds limit {self.limits.max_exposure_per_pair_pct:.1%}",
                ))

        # Process violations
        if violations:
            self.violations.extend(violations)
            self.trading_halted = True
            self.halt_timestamp = timestamp

            # Call alert callback for each violation
            if self.alert_callback:
                for v in violations:
                    self.alert_callback(v)

        return violations

    def can_open_position(
        self,
        state: PortfolioState,
        pair: str,
        position_size_pct: float,
    ) -> tuple[bool, str | None]:
        """Check if a new position can be opened.

        Args:
            state: Current portfolio state.
            pair: Currency pair for the new position.
            position_size_pct: Size of new position as % of equity.

        Returns:
            Tuple of (allowed, reason_if_blocked).
        """
        if self.is_trading_halted:
            return False, "Trading halted due to limit violation"

        # Check max positions
        if state.open_positions >= self.limits.max_positions:
            return False, f"Maximum positions ({self.limits.max_positions}) reached"

        # Check pair exposure
        current_exposure = state.exposure_by_pair.get(pair, 0.0)
        new_exposure = current_exposure + position_size_pct
        if new_exposure > self.limits.max_exposure_per_pair_pct:
            return False, f"{pair} exposure would be {new_exposure:.1%} (limit: {self.limits.max_exposure_per_pair_pct:.1%})"

        return True, None

    def reset(self) -> None:
        """Reset circuit breaker state (manual override)."""
        self.trading_halted = False
        self.halt_timestamp = None
        self.violations.clear()

    def get_status_report(self, state: PortfolioState) -> dict:
        """Generate a status report for current limits.

        Args:
            state: Current portfolio state.

        Returns:
            Dict with limit utilization percentages.
        """
        # Calculate utilization for each limit
        daily_util = 0.0
        if state.daily_pnl < 0:
            daily_loss_pct = abs(state.daily_pnl) / (state.equity + abs(state.daily_pnl))
            daily_util = daily_loss_pct / self.limits.max_daily_loss_pct

        weekly_util = 0.0
        if state.weekly_pnl < 0:
            weekly_loss_pct = abs(state.weekly_pnl) / (state.equity + abs(state.weekly_pnl))
            weekly_util = weekly_loss_pct / self.limits.max_weekly_loss_pct

        dd_pct = (state.high_water_mark - state.equity) / state.high_water_mark
        dd_util = dd_pct / self.limits.max_drawdown_pct

        pos_util = state.open_positions / self.limits.max_positions

        max_pair_util = 0.0
        for exposure in state.exposure_by_pair.values():
            pair_util = exposure / self.limits.max_exposure_per_pair_pct
            max_pair_util = max(max_pair_util, pair_util)

        return {
            "trading_halted": self.is_trading_halted,
            "daily_loss_utilization": daily_util,
            "weekly_loss_utilization": weekly_util,
            "drawdown_utilization": dd_util,
            "position_utilization": pos_util,
            "pair_exposure_utilization": max_pair_util,
            "violations_count": len(self.violations),
            "current_drawdown_pct": dd_pct,
            "current_daily_pnl_pct": state.daily_pnl / state.equity if state.equity > 0 else 0,
        }
