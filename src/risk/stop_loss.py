"""Adaptive stop loss based on volatility (ATR).

Fixed percentage stop losses have a problem:
- In calm markets: 2% stop is too loose, leaving money on the table
- In volatile markets: 2% stop is too tight, causing whipsaw losses

ATR-based stops ADAPT to current market conditions:
- Calculate Average True Range (ATR) over N periods
- Set stop at entry ± (ATR × multiplier)
- Stop widens in volatile markets, tightens in calm markets

Trailing stops:
- Once position is profitable, trail the stop behind price
- Lock in gains while letting winners run
- ATR trailing: trail by ATR × multiplier from the high water mark

This module provides:
1. Initial stop calculation (ATR-based)
2. Trailing stop updates
3. Stop trigger detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd


class StopType(Enum):
    """Type of stop loss."""
    FIXED_PCT = auto()  # Fixed percentage from entry
    ATR_BASED = auto()  # ATR × multiplier from entry
    TRAILING_ATR = auto()  # ATR × multiplier from high water mark


@dataclass
class StopLevel:
    """Current stop loss level for a position."""
    stop_type: StopType
    stop_price: float
    entry_price: float
    is_long: bool
    high_water_mark: float  # Highest price (long) or lowest price (short) since entry
    atr_at_entry: float | None = None
    atr_multiplier: float | None = None
    fixed_pct: float | None = None
    triggered: bool = False

    @property
    def distance_pct(self) -> float:
        """Distance to stop as percentage of current high water mark."""
        if self.is_long:
            return (self.high_water_mark - self.stop_price) / self.high_water_mark
        else:
            return (self.stop_price - self.high_water_mark) / self.high_water_mark


@dataclass
class StopHistory:
    """Track stop level changes over time."""
    timestamps: list = field(default_factory=list)
    stop_prices: list = field(default_factory=list)
    high_water_marks: list = field(default_factory=list)


class AdaptiveStopLoss:
    """Calculate and manage adaptive stop losses.

    Args:
        atr_period: Period for ATR calculation. Default 14.
        atr_multiplier: Stop distance as multiple of ATR. Default 2.0.
        trailing_activation_pct: Profit % to activate trailing. Default 0.02 (2%).
        min_stop_pct: Minimum stop distance. Default 0.01 (1%).
        max_stop_pct: Maximum stop distance. Default 0.10 (10%).

    Example:
        >>> stop_mgr = AdaptiveStopLoss()
        >>> stop = stop_mgr.initial_stop(
        ...     entry_price=150.0,
        ...     is_long=True,
        ...     atr=0.75,
        ... )
        >>> print(f"Stop at {stop.stop_price:.2f}")
        Stop at 148.50
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        trailing_activation_pct: float = 0.02,
        min_stop_pct: float = 0.01,
        max_stop_pct: float = 0.10,
    ) -> None:
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trailing_activation_pct = trailing_activation_pct
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range.

        ATR measures volatility by considering:
        - High - Low (current bar range)
        - |High - Previous Close| (gap up)
        - |Low - Previous Close| (gap down)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.

        Returns:
            Series of ATR values.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def initial_stop(
        self,
        entry_price: float,
        is_long: bool,
        atr: float | None = None,
        fixed_pct: float | None = None,
    ) -> StopLevel:
        """Calculate initial stop level.

        Priority:
        1. If ATR provided, use ATR-based stop
        2. If fixed_pct provided, use fixed percentage
        3. Default to ATR-based with atr_multiplier

        Args:
            entry_price: Entry price of the position.
            is_long: True for long position, False for short.
            atr: Current ATR value (for ATR-based stop).
            fixed_pct: Fixed percentage stop (fallback).

        Returns:
            StopLevel with calculated stop price.
        """
        if atr is not None and atr > 0:
            # ATR-based stop
            stop_distance = atr * self.atr_multiplier
            stop_pct = stop_distance / entry_price

            # Clamp to min/max
            stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_pct))
            stop_distance = entry_price * stop_pct

            if is_long:
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance

            return StopLevel(
                stop_type=StopType.ATR_BASED,
                stop_price=stop_price,
                entry_price=entry_price,
                is_long=is_long,
                high_water_mark=entry_price,
                atr_at_entry=atr,
                atr_multiplier=self.atr_multiplier,
            )

        # Fixed percentage stop
        pct = fixed_pct or self.min_stop_pct
        stop_distance = entry_price * pct

        if is_long:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        return StopLevel(
            stop_type=StopType.FIXED_PCT,
            stop_price=stop_price,
            entry_price=entry_price,
            is_long=is_long,
            high_water_mark=entry_price,
            fixed_pct=pct,
        )

    def update_trailing(
        self,
        current_stop: StopLevel,
        current_price: float,
        current_atr: float | None = None,
    ) -> StopLevel:
        """Update stop for trailing behavior.

        For long positions:
        - If price makes new high, trail stop up
        - Stop never moves down (ratchet effect)

        For short positions:
        - If price makes new low, trail stop down
        - Stop never moves up

        Args:
            current_stop: Current StopLevel.
            current_price: Current market price.
            current_atr: Current ATR (optional, for dynamic trailing).

        Returns:
            Updated StopLevel (new object, original unchanged).
        """
        new_hwm = current_stop.high_water_mark
        new_stop = current_stop.stop_price

        if current_stop.is_long:
            # Update high water mark
            if current_price > new_hwm:
                new_hwm = current_price

                # Check if trailing should activate
                profit_pct = (new_hwm - current_stop.entry_price) / current_stop.entry_price
                if profit_pct >= self.trailing_activation_pct:
                    # Calculate new trailing stop
                    if current_atr and current_atr > 0:
                        trail_distance = current_atr * self.atr_multiplier
                    elif current_stop.atr_at_entry:
                        trail_distance = current_stop.atr_at_entry * self.atr_multiplier
                    else:
                        trail_distance = new_hwm * (current_stop.fixed_pct or self.min_stop_pct)

                    candidate_stop = new_hwm - trail_distance

                    # Only move stop UP (never down)
                    if candidate_stop > new_stop:
                        new_stop = candidate_stop
        else:
            # Short position - update low water mark
            if current_price < new_hwm:
                new_hwm = current_price

                # Check if trailing should activate
                profit_pct = (current_stop.entry_price - new_hwm) / current_stop.entry_price
                if profit_pct >= self.trailing_activation_pct:
                    if current_atr and current_atr > 0:
                        trail_distance = current_atr * self.atr_multiplier
                    elif current_stop.atr_at_entry:
                        trail_distance = current_stop.atr_at_entry * self.atr_multiplier
                    else:
                        trail_distance = new_hwm * (current_stop.fixed_pct or self.min_stop_pct)

                    candidate_stop = new_hwm + trail_distance

                    # Only move stop DOWN (never up) for shorts
                    if candidate_stop < new_stop:
                        new_stop = candidate_stop

        # Return updated stop level
        return StopLevel(
            stop_type=StopType.TRAILING_ATR if current_atr else current_stop.stop_type,
            stop_price=new_stop,
            entry_price=current_stop.entry_price,
            is_long=current_stop.is_long,
            high_water_mark=new_hwm,
            atr_at_entry=current_stop.atr_at_entry,
            atr_multiplier=current_stop.atr_multiplier,
            fixed_pct=current_stop.fixed_pct,
        )

    def check_triggered(
        self,
        stop: StopLevel,
        current_low: float,
        current_high: float,
    ) -> bool:
        """Check if stop loss was triggered.

        For longs: triggered if low <= stop_price
        For shorts: triggered if high >= stop_price

        Args:
            stop: Current StopLevel.
            current_low: Low price of current bar.
            current_high: High price of current bar.

        Returns:
            True if stop was triggered.
        """
        if stop.is_long:
            return current_low <= stop.stop_price
        else:
            return current_high >= stop.stop_price

    def recommend_stop(
        self,
        entry_price: float,
        is_long: bool,
        df: pd.DataFrame,
    ) -> StopLevel:
        """Recommend stop level based on recent data.

        Calculates ATR from the provided data and returns an initial stop.

        Args:
            entry_price: Entry price of the position.
            is_long: True for long, False for short.
            df: Recent price data with high, low, close columns.

        Returns:
            StopLevel with recommended stop.
        """
        atr_series = self.calculate_atr(df)
        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not np.isnan(atr_series.iloc[-1]) else None

        return self.initial_stop(
            entry_price=entry_price,
            is_long=is_long,
            atr=current_atr,
        )
