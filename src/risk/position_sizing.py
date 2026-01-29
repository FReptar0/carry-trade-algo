"""Dynamic position sizing for risk management.

Position sizing determines HOW MUCH to risk on each trade. This is arguably
more important than entry/exit signals - even a great strategy fails with
bad position sizing.

Three methods implemented:

1. KELLY CRITERION
   Mathematically optimal sizing that maximizes long-term growth.
   Formula: f* = (p × b - q) / b
   Where: p = win probability, q = loss probability, b = win/loss ratio

   Problem: Full Kelly is too aggressive (huge drawdowns).
   Solution: Use fractional Kelly (1/4 or 1/2) for safety.

2. FIXED FRACTIONAL
   Risk a fixed percentage of capital on each trade.
   Size = (Capital × Risk%) / Stop Distance

   Simple and effective. Most retail traders use 1-2% risk per trade.
   Drawback: Doesn't adapt to changing win rates.

3. VOLATILITY-BASED (ATR Sizing)
   Size inversely proportional to current volatility.
   When volatility is high, take smaller positions.

   Formula: Size = Target Risk$ / (ATR × Multiplier × Pip Value)

   Adapts to market conditions automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class SizingMethod(Enum):
    """Available position sizing methods."""
    KELLY = auto()
    FIXED_FRACTIONAL = auto()
    VOLATILITY_BASED = auto()


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    method: SizingMethod
    size: float  # Position size as fraction of capital (0.0 to 1.0)
    raw_size: float  # Size before any caps applied
    capped: bool  # Whether size was reduced by caps
    details: dict  # Method-specific calculation details


class PositionSizer:
    """Calculate position sizes using various risk management methods.

    Args:
        max_position_pct: Maximum position size (cap). Default 0.20 (20%).
        kelly_fraction: Fraction of Kelly to use. Default 0.25 (quarter Kelly).
        default_risk_pct: Default risk per trade for fixed fractional. Default 0.02 (2%).

    Example:
        >>> sizer = PositionSizer()
        >>> result = sizer.kelly_criterion(win_rate=0.55, avg_win=0.03, avg_loss=0.02)
        >>> print(f"Kelly says risk {result.size:.1%} of capital")
        Kelly says risk 5.0% of capital
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        kelly_fraction: float = 0.25,
        default_risk_pct: float = 0.02,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.default_risk_pct = default_risk_pct

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> PositionSizeResult:
        """Calculate position size using Kelly Criterion.

        The Kelly Criterion maximizes long-term wealth growth but can be
        extremely aggressive. We use fractional Kelly for safety.

        Args:
            win_rate: Historical win rate (0.0 to 1.0).
            avg_win: Average winning trade return (e.g., 0.03 for 3%).
            avg_loss: Average losing trade return (positive, e.g., 0.02 for 2% loss).

        Returns:
            PositionSizeResult with recommended size.
        """
        if win_rate <= 0 or win_rate >= 1:
            return PositionSizeResult(
                method=SizingMethod.KELLY,
                size=0.0,
                raw_size=0.0,
                capped=False,
                details={"error": "Win rate must be between 0 and 1 exclusive"},
            )

        if avg_loss <= 0:
            return PositionSizeResult(
                method=SizingMethod.KELLY,
                size=0.0,
                raw_size=0.0,
                capped=False,
                details={"error": "avg_loss must be positive"},
            )

        # Kelly formula: f* = (p × b - q) / b
        # Where b = avg_win / avg_loss (win/loss ratio)
        p = win_rate  # Probability of winning
        q = 1 - win_rate  # Probability of losing
        b = avg_win / avg_loss  # Win/loss ratio

        full_kelly = (p * b - q) / b

        # Full Kelly can be negative (don't trade) or very large
        if full_kelly <= 0:
            return PositionSizeResult(
                method=SizingMethod.KELLY,
                size=0.0,
                raw_size=full_kelly,
                capped=False,
                details={
                    "full_kelly": full_kelly,
                    "message": "Kelly suggests no trade (negative edge)",
                    "win_rate": win_rate,
                    "win_loss_ratio": b,
                },
            )

        # Apply fractional Kelly for safety
        fractional_kelly = full_kelly * self.kelly_fraction

        # Cap at maximum position size
        capped = fractional_kelly > self.max_position_pct
        final_size = min(fractional_kelly, self.max_position_pct)

        return PositionSizeResult(
            method=SizingMethod.KELLY,
            size=final_size,
            raw_size=fractional_kelly,
            capped=capped,
            details={
                "full_kelly": full_kelly,
                "fractional_kelly": fractional_kelly,
                "kelly_fraction_used": self.kelly_fraction,
                "win_rate": win_rate,
                "win_loss_ratio": b,
                "expected_edge": p * avg_win - q * avg_loss,
            },
        )

    def fixed_fractional(
        self,
        capital: float,
        stop_loss_distance: float,
        entry_price: float,
        risk_pct: float | None = None,
    ) -> PositionSizeResult:
        """Calculate position size using fixed fractional method.

        Risk a fixed percentage of capital, adjusting size based on
        stop loss distance.

        Args:
            capital: Current account capital.
            stop_loss_distance: Distance to stop loss in price units.
            entry_price: Entry price for the position.
            risk_pct: Risk per trade as decimal. Defaults to default_risk_pct.

        Returns:
            PositionSizeResult with recommended size.
        """
        risk_pct = risk_pct or self.default_risk_pct

        if stop_loss_distance <= 0:
            return PositionSizeResult(
                method=SizingMethod.FIXED_FRACTIONAL,
                size=0.0,
                raw_size=0.0,
                capped=False,
                details={"error": "Stop loss distance must be positive"},
            )

        # Dollar amount we're willing to risk
        risk_amount = capital * risk_pct

        # Convert stop distance to percentage
        stop_pct = stop_loss_distance / entry_price

        # Position size = risk amount / stop distance
        # If stop is 2% and we risk $1000, position = $1000 / 0.02 = $50,000
        # As fraction of capital: $50,000 / capital
        position_value = risk_amount / stop_pct
        raw_size = position_value / capital

        # Cap at maximum
        capped = raw_size > self.max_position_pct
        final_size = min(raw_size, self.max_position_pct)

        return PositionSizeResult(
            method=SizingMethod.FIXED_FRACTIONAL,
            size=final_size,
            raw_size=raw_size,
            capped=capped,
            details={
                "risk_pct": risk_pct,
                "risk_amount": risk_amount,
                "stop_pct": stop_pct,
                "position_value": position_value,
                "capital": capital,
            },
        )

    def volatility_based(
        self,
        capital: float,
        atr: float,
        entry_price: float,
        atr_multiplier: float = 2.0,
        target_risk_pct: float | None = None,
    ) -> PositionSizeResult:
        """Calculate position size based on current volatility (ATR).

        Higher volatility → smaller position.
        Lower volatility → larger position.

        The stop loss is set at ATR × multiplier, then we size to risk
        a fixed percentage of capital.

        Args:
            capital: Current account capital.
            atr: Current Average True Range value.
            entry_price: Entry price for the position.
            atr_multiplier: Stop distance as multiple of ATR. Default 2.0.
            target_risk_pct: Target risk per trade. Defaults to default_risk_pct.

        Returns:
            PositionSizeResult with recommended size.
        """
        target_risk_pct = target_risk_pct or self.default_risk_pct

        if atr <= 0:
            return PositionSizeResult(
                method=SizingMethod.VOLATILITY_BASED,
                size=0.0,
                raw_size=0.0,
                capped=False,
                details={"error": "ATR must be positive"},
            )

        # Stop distance in price units
        stop_distance = atr * atr_multiplier

        # Use fixed fractional with ATR-derived stop
        ff_result = self.fixed_fractional(
            capital=capital,
            stop_loss_distance=stop_distance,
            entry_price=entry_price,
            risk_pct=target_risk_pct,
        )

        # Override method in result
        return PositionSizeResult(
            method=SizingMethod.VOLATILITY_BASED,
            size=ff_result.size,
            raw_size=ff_result.raw_size,
            capped=ff_result.capped,
            details={
                **ff_result.details,
                "atr": atr,
                "atr_multiplier": atr_multiplier,
                "stop_distance": stop_distance,
                "stop_distance_pct": stop_distance / entry_price,
            },
        )

    def recommend(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> dict[SizingMethod, PositionSizeResult]:
        """Get recommendations from all applicable methods.

        Args:
            capital: Current account capital.
            entry_price: Entry price for the position.
            atr: Current ATR value.
            win_rate: Historical win rate (for Kelly).
            avg_win: Average winning trade return (for Kelly).
            avg_loss: Average losing trade return (for Kelly).
            stop_loss_pct: Stop loss percentage (for fixed fractional).

        Returns:
            Dict mapping method to result for all applicable methods.
        """
        results = {}

        # Volatility-based (always applicable if we have ATR)
        results[SizingMethod.VOLATILITY_BASED] = self.volatility_based(
            capital=capital,
            atr=atr,
            entry_price=entry_price,
        )

        # Fixed fractional (if we have stop loss)
        if stop_loss_pct and stop_loss_pct > 0:
            stop_distance = entry_price * stop_loss_pct
            results[SizingMethod.FIXED_FRACTIONAL] = self.fixed_fractional(
                capital=capital,
                stop_loss_distance=stop_distance,
                entry_price=entry_price,
            )

        # Kelly (if we have historical stats)
        if win_rate and avg_win and avg_loss:
            results[SizingMethod.KELLY] = self.kelly_criterion(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )

        return results
