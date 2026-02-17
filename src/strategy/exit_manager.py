"""Centralized exit logic for trading strategies.

EXIT OPTIMIZATION PHILOSOPHY:
=============================
Most traders focus on entries, but exits determine actual profit.

Key principles:
1. Cut losses quickly (but not too quickly - avoid whipsaw)
2. Let winners run (but lock in profit with trailing stops)
3. Adapt to volatility (wider stops in volatile markets)
4. Time-based review (don't hold marginal positions forever)

EXIT TYPES:
===========
1. STOP_LOSS: Initial stop hit (limit loss)
2. TRAILING_STOP: Trailing stop hit (lock profit)
3. TIME_BASED: Held too long without progress
4. PROFIT_TARGET: Scaled take-profit hit
5. SUPPORT_BREAK: Key support level broken
6. REGIME_CHANGE: Market regime turned unfavorable
7. TREND_REVERSAL: Weekly/daily trend reversed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.strategy.indicators import atr
from src.regime.detector import RegimeState, CompositeRegime


class ExitType(Enum):
    """Types of exit signals."""

    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    PROFIT_TARGET = "profit_target"
    SUPPORT_BREAK = "support_break"
    REGIME_CHANGE = "regime_change"
    TREND_REVERSAL = "trend_reversal"
    MANUAL = "manual"


@dataclass
class ExitSignal:
    """An exit recommendation from the exit manager."""

    should_exit: bool
    exit_type: Optional[ExitType] = None
    reason: str = ""
    urgency: float = 0.0  # 0-1, how urgent is the exit
    recommended_price: Optional[float] = None  # For limit exits

    @property
    def is_urgent(self) -> bool:
        """Check if exit is urgent (> 0.7)."""
        return self.urgency > 0.7


@dataclass
class PositionState:
    """State of a position for exit evaluation."""

    entry_price: float
    entry_time: datetime
    current_price: float
    current_time: datetime
    high_water_mark: float  # Highest price since entry
    low_water_mark: float  # Lowest price since entry
    side: str = "long"  # "long" or "short"

    @property
    def return_pct(self) -> float:
        """Current return percentage."""
        if self.side == "long":
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    @property
    def max_favorable_excursion(self) -> float:
        """Maximum favorable price move (MFE)."""
        if self.side == "long":
            return (self.high_water_mark - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.low_water_mark) / self.entry_price

    @property
    def max_adverse_excursion(self) -> float:
        """Maximum adverse price move (MAE)."""
        if self.side == "long":
            return (self.entry_price - self.low_water_mark) / self.entry_price
        else:
            return (self.high_water_mark - self.entry_price) / self.entry_price

    @property
    def hold_duration(self) -> timedelta:
        """Time held."""
        return self.current_time - self.entry_time

    @property
    def hold_days(self) -> float:
        """Days held."""
        return self.hold_duration.total_seconds() / 86400


@dataclass
class ExitManagerConfig:
    """Configuration for exit manager."""

    # Volatility-adjusted stops
    base_atr_mult: float = 3.0
    vol_adjustment_enabled: bool = True
    vol_lookback: int = 60  # Days to compare ATR against

    # Time-based exits
    max_hold_days: int = 30  # Force review after N days
    min_hold_hours: int = 24  # Don't exit within first day
    time_exit_enabled: bool = True

    # Profit-scaled trailing
    profit_scale_enabled: bool = True
    profit_scale_levels: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.02, 1.5),  # At 2% profit, trail by 1.5x ATR
            (0.04, 1.2),  # At 4% profit, trail by 1.2x ATR
            (0.06, 1.0),  # At 6% profit, trail by 1.0x ATR
            (0.10, 0.8),  # At 10% profit, trail by 0.8x ATR (tight)
        ]
    )

    # Support/resistance
    use_sr_exits: bool = False
    sr_lookback: int = 50

    # Regime-based exits
    regime_exit_enabled: bool = True
    exit_on_bad_regime: bool = True
    min_hold_days_for_regime_exit: int = 5  # Regime exits disabled until held N days


class ExitManager:
    """Manages all exit logic for positions.

    Centralizes exit decisions to ensure consistent behavior and
    allow easy optimization of exit parameters.

    Example:
        manager = ExitManager()
        position = PositionState(entry_price=150, ...)

        exit_signal = manager.check_exit(
            position=position,
            price_data=df,
            regime=current_regime,
        )

        if exit_signal.should_exit:
            close_position(reason=exit_signal.reason)
    """

    def __init__(self, config: ExitManagerConfig | None = None) -> None:
        self.config = config or ExitManagerConfig()
        self._stop_price: float = 0.0
        self._trailing_active: bool = False

    def initialize_stop(
        self,
        entry_price: float,
        current_atr: float,
        regime: Optional[RegimeState] = None,
    ) -> float:
        """Calculate initial stop loss price.

        Args:
            entry_price: Position entry price.
            current_atr: Current ATR value.
            regime: Current regime state (optional).

        Returns:
            Initial stop loss price.
        """
        # Base stop multiplier
        mult = self.config.base_atr_mult

        # Adjust for regime
        if regime and self.config.vol_adjustment_enabled:
            if regime.composite_regime == CompositeRegime.TREND_HIGH_VOL:
                mult *= 1.5  # Wider stops in high vol
            elif regime.composite_regime == CompositeRegime.RANGE_HIGH_VOL:
                mult *= 2.0  # Very wide if forced to trade
            elif regime.composite_regime == CompositeRegime.TREND_LOW_VOL:
                mult *= 0.8  # Tighter stops in calm trend

        stop_distance = current_atr * mult
        self._stop_price = entry_price - stop_distance
        self._trailing_active = False

        return self._stop_price

    def check_exit(
        self,
        position: PositionState,
        price_data: pd.DataFrame,
        regime: Optional[RegimeState] = None,
        weekly_trend: Optional[str] = None,
    ) -> ExitSignal:
        """Check all exit conditions and return recommendation.

        Args:
            position: Current position state.
            price_data: Price data for calculations.
            regime: Current regime state.
            weekly_trend: Weekly trend direction ("up", "down", "neutral").

        Returns:
            ExitSignal with recommendation.
        """
        exit_signals = []

        # Calculate current ATR
        if len(price_data) >= 14:
            current_atr = atr(
                price_data["high"],
                price_data["low"],
                price_data["close"],
            ).iloc[-1]
        else:
            current_atr = abs(position.entry_price * 0.02)  # Fallback

        # 1. Check volatility-adjusted stop
        exit_signals.append(
            self._check_vol_adjusted_stop(position, current_atr, price_data)
        )

        # 2. Check profit-scaled trailing
        exit_signals.append(self._check_profit_scaled_trail(position, current_atr))

        # 3. Check time-based exit
        if self.config.time_exit_enabled:
            exit_signals.append(self._check_time_exit(position))

        # 4. Check support break
        if self.config.use_sr_exits:
            exit_signals.append(self._check_support_break(position, price_data))

        # 5. Check regime change
        if self.config.regime_exit_enabled and regime:
            exit_signals.append(self._check_regime_exit(position, regime))

        # 6. Check trend reversal
        if weekly_trend:
            exit_signals.append(self._check_trend_reversal(position, weekly_trend))

        # Return highest urgency exit that should exit
        should_exit_signals = [s for s in exit_signals if s.should_exit]
        if should_exit_signals:
            return max(should_exit_signals, key=lambda x: x.urgency)

        return ExitSignal(should_exit=False, reason="No exit conditions met")

    def _check_vol_adjusted_stop(
        self,
        position: PositionState,
        current_atr: float,
        price_data: pd.DataFrame,
    ) -> ExitSignal:
        """Check volatility-adjusted stop loss."""
        # Calculate historical ATR for comparison
        if len(price_data) >= self.config.vol_lookback:
            hist_atr = (
                atr(
                    price_data["high"],
                    price_data["low"],
                    price_data["close"],
                )
                .iloc[-self.config.vol_lookback : -10]
                .mean()
            )
        else:
            hist_atr = current_atr

        # Adjust stop based on current vs historical volatility
        vol_ratio = current_atr / hist_atr if hist_atr > 0 else 1.0

        if self.config.vol_adjustment_enabled:
            # In high vol, widen stop; in low vol, tighten
            adjusted_mult = self.config.base_atr_mult * max(1.0, vol_ratio)
        else:
            adjusted_mult = self.config.base_atr_mult

        stop_distance = current_atr * adjusted_mult
        dynamic_stop = position.entry_price - stop_distance

        # Use the higher of initial stop and dynamic stop
        effective_stop = (
            max(self._stop_price, dynamic_stop)
            if self._stop_price > 0
            else dynamic_stop
        )

        if position.current_price <= effective_stop:
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.STOP_LOSS,
                reason=f"Stop loss hit at {effective_stop:.2f} (vol-adjusted)",
                urgency=0.95,
                recommended_price=effective_stop,
            )

        return ExitSignal(should_exit=False)

    def _check_profit_scaled_trail(
        self,
        position: PositionState,
        current_atr: float,
    ) -> ExitSignal:
        """Check profit-scaled trailing stop.

        As profit increases, trailing stop tightens to lock in gains.
        """
        if not self.config.profit_scale_enabled:
            return ExitSignal(should_exit=False)

        profit_pct = position.return_pct

        # Find applicable scale level
        trail_mult = None
        for threshold, mult in reversed(self.config.profit_scale_levels):
            if profit_pct >= threshold:
                trail_mult = mult
                break

        if trail_mult is None:
            return ExitSignal(should_exit=False)  # Not enough profit yet

        # Activate trailing
        self._trailing_active = True

        # Calculate trailing stop
        trail_distance = current_atr * trail_mult
        trail_stop = position.high_water_mark - trail_distance

        # Only use trailing stop if it's above breakeven
        if trail_stop <= position.entry_price:
            return ExitSignal(should_exit=False)

        # Update stop price if trailing is higher
        if trail_stop > self._stop_price:
            self._stop_price = trail_stop

        if position.current_price <= trail_stop:
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.TRAILING_STOP,
                reason=f"Trailing stop hit at {trail_stop:.2f} (profit {profit_pct:.1%})",
                urgency=0.90,
                recommended_price=trail_stop,
            )

        return ExitSignal(should_exit=False)

    def _check_time_exit(self, position: PositionState) -> ExitSignal:
        """Check time-based exit conditions."""
        # Don't exit too early
        if position.hold_duration < timedelta(hours=self.config.min_hold_hours):
            return ExitSignal(should_exit=False)

        # Force review after max hold days
        if position.hold_days >= self.config.max_hold_days:
            profit_pct = position.return_pct

            if profit_pct > 0.02:
                # Profitable - tighten trail instead of exiting
                return ExitSignal(
                    should_exit=False,
                    reason=f"Time limit ({self.config.max_hold_days}d) but profitable, tighten trail",
                )
            elif profit_pct < -0.01:
                # Losing - exit
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.TIME_BASED,
                    reason=f"Time limit ({self.config.max_hold_days}d) with loss {profit_pct:.1%}",
                    urgency=0.70,
                )
            else:
                # Marginal - exit
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.TIME_BASED,
                    reason=f"Time limit ({self.config.max_hold_days}d) with marginal profit",
                    urgency=0.60,
                )

        return ExitSignal(should_exit=False)

    def _check_support_break(
        self,
        position: PositionState,
        price_data: pd.DataFrame,
    ) -> ExitSignal:
        """Check if price broke key support level."""
        if len(price_data) < self.config.sr_lookback:
            return ExitSignal(should_exit=False)

        # Find support levels (swing lows)
        lows = price_data["low"].tail(self.config.sr_lookback)
        support_levels = self._find_swing_lows(lows)

        if not support_levels:
            return ExitSignal(should_exit=False)

        # Check if price broke support
        for support in support_levels:
            # Support should be near or below entry (within 5%)
            if support > position.entry_price * 1.05:
                continue

            # Check if broke support (0.5% buffer)
            if position.current_price < support * 0.995:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.SUPPORT_BREAK,
                    reason=f"Broke support at {support:.2f}",
                    urgency=0.80,
                    recommended_price=support,
                )

        return ExitSignal(should_exit=False)

    def _find_swing_lows(self, lows: pd.Series, min_distance: int = 3) -> list[float]:
        """Find swing low levels in price data."""
        swing_lows = []

        for i in range(min_distance, len(lows) - min_distance):
            is_swing_low = True
            current = lows.iloc[i]

            # Check if lower than surrounding bars
            for j in range(1, min_distance + 1):
                if current >= lows.iloc[i - j] or current >= lows.iloc[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append(current)

        return sorted(set(swing_lows))

    def _check_regime_exit(
        self,
        position: PositionState,
        regime: RegimeState,
    ) -> ExitSignal:
        """Check if regime change warrants exit."""
        if not self.config.exit_on_bad_regime:
            return ExitSignal(should_exit=False)

        # Don't fire regime exits on young positions — carry trades need
        # time to develop before hourly regime noise should close them.
        if position.hold_days < self.config.min_hold_days_for_regime_exit:
            return ExitSignal(should_exit=False)

        profit_pct = position.return_pct

        # Exit if regime is worst case and position is losing money
        if regime.composite_regime == CompositeRegime.RANGE_HIGH_VOL:
            if profit_pct < 0.0:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.REGIME_CHANGE,
                    reason=f"Bad regime (RANGE_HIGH_VOL) with {profit_pct:.1%} profit",
                    urgency=0.75,
                )

        # Exit if trend direction reversed and position underwater
        if regime.trend_direction == "down" and position.side == "long":
            if profit_pct < -0.01:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.REGIME_CHANGE,
                    reason=f"Regime trend turned down with {profit_pct:.1%} loss",
                    urgency=0.70,
                )

        return ExitSignal(should_exit=False)

    def _check_trend_reversal(
        self,
        position: PositionState,
        weekly_trend: str,
    ) -> ExitSignal:
        """Check if weekly trend reversed."""
        if position.side == "long" and weekly_trend == "down":
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.TREND_REVERSAL,
                reason="Weekly trend reversed to down",
                urgency=0.85,
            )
        elif position.side == "short" and weekly_trend == "up":
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.TREND_REVERSAL,
                reason="Weekly trend reversed to up",
                urgency=0.85,
            )

        return ExitSignal(should_exit=False)

    def get_current_stop(self) -> float:
        """Get current stop price."""
        return self._stop_price

    def is_trailing_active(self) -> bool:
        """Check if trailing stop is active."""
        return self._trailing_active
