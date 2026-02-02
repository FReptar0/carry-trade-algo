"""Carry Trade Strategy v3 - Long-Term Trend Following.

KEY INSIGHT FROM V2 FAILURE:
============================
V2 over-filtered. Too many entry conditions = missed opportunities.
Carry trading works best when you HOLD LONGER to accumulate swap income.

V3 PHILOSOPHY:
==============
1. Trade LESS frequently, hold LONGER
2. Follow the BIG trend (weekly, not daily)
3. Use wider stops that respect volatility
4. Let swap income compound
5. Only exit on clear trend reversal

STRUCTURAL CHANGES:
===================
1. Weekly trend filter (200-day SMA proxy for weekly direction)
2. Entry only when daily aligns with weekly
3. Much wider ATR stops (3x instead of 2x)
4. No profit targets - let winners run
5. Exit only on trend break or trailing stop
6. Minimum hold period to accumulate swap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import Signal, Strategy, TradeSignal
from src.strategy.indicators import atr, rsi, sma


@dataclass
class StrategyConfigV3:
    """Configuration optimized for longer-term trend following."""

    # Trend identification
    short_ma_period: int = 50      # Medium-term trend
    long_ma_period: int = 200      # Long-term trend (proxy for weekly)

    # Entry conditions - SIMPLIFIED
    min_swap_threshold: float = 0.0  # Sign-only gate: positive swap = get paid to hold
    max_rsi: int = 75                   # Only avoid extreme overbought

    # Risk management - WIDER
    atr_period: int = 14
    atr_stop_mult: float = 3.0     # Wider stops (was 2.0 in V2)
    trailing_activation_pct: float = 0.05  # Start trailing after 5% profit
    trailing_atr_mult: float = 2.0  # Trail by 2x ATR

    # Position sizing
    position_size_pct: float = 0.08  # Slightly smaller since we hold longer
    max_positions: int = 2


class CarryTradeStrategyV3(Strategy):
    """Long-term trend-following carry trade strategy.

    Philosophy: Follow the big trend, hold for carry, exit on reversal.

    Key differences from V1/V2:
    - Uses 200-day MA for trend (longer-term view)
    - Wider 3x ATR stops (more room to breathe)
    - No take profit - let winners run
    - Trailing stop only after significant profit
    - Minimum conditions for entry (don't over-filter)
    """

    def __init__(self, config: StrategyConfigV3 | None = None) -> None:
        self.config = config or StrategyConfigV3()

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Generate signals with long-term trend following logic."""
        signals: list[TradeSignal] = []

        # Need enough data for 200-day MA
        if len(data) < self.config.long_ma_period + 20:
            return signals

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Calculate indicators
        short_ma = sma(close, self.config.short_ma_period)
        long_ma = sma(close, self.config.long_ma_period)
        rsi_val = rsi(close, period=14)
        atr_val = atr(high, low, close, period=self.config.atr_period)

        # Track position state
        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        max_price = 0.0
        trailing_active = False

        for i in range(self.config.long_ma_period + 10, len(data)):
            ts = data["timestamp"].iloc[i]
            price = close.iloc[i]
            swap = data["swap_long"].iloc[i]

            current_short_ma = short_ma.iloc[i]
            current_long_ma = long_ma.iloc[i]
            current_rsi = rsi_val.iloc[i]
            current_atr = atr_val.iloc[i]

            # Previous values for crossover detection
            prev_short_ma = short_ma.iloc[i-1]
            prev_long_ma = long_ma.iloc[i-1]

            if pd.isna(current_atr) or pd.isna(current_long_ma):
                continue

            if not in_position:
                # ===== ENTRY LOGIC =====
                # Simple conditions - don't over-filter!

                # 1. Positive swap (the carry trade rationale)
                if swap <= 0:
                    continue

                # 2. BOTH MAs trending up (price > short > long)
                if not (price > current_short_ma > current_long_ma):
                    continue

                # 3. Not extremely overbought
                if current_rsi > self.config.max_rsi:
                    continue

                # 4. Recent golden cross OR already in uptrend
                golden_cross = (prev_short_ma <= prev_long_ma and
                               current_short_ma > current_long_ma)
                strong_uptrend = (current_short_ma > current_long_ma and
                                 price > current_short_ma * 1.01)  # Price 1% above short MA

                if golden_cross or strong_uptrend:
                    in_position = True
                    entry_price = price
                    max_price = price
                    trailing_active = False

                    # Wide initial stop
                    stop_price = price - (current_atr * self.config.atr_stop_mult)

                    entry_type = "Golden Cross" if golden_cross else "Strong Uptrend"
                    signals.append(TradeSignal(
                        timestamp=ts,
                        signal=Signal.LONG,
                        reason=f"Entry ({entry_type}): Price={price:.2f}, 50MA={current_short_ma:.2f}, 200MA={current_long_ma:.2f}",
                    ))

            else:
                # ===== EXIT LOGIC =====
                max_price = max(max_price, price)
                profit_pct = (price - entry_price) / entry_price

                exit_signal = None
                exit_reason = ""

                # 1. Initial stop loss
                if price <= stop_price:
                    exit_signal = Signal.CLOSE
                    exit_reason = f"Stop loss at {stop_price:.2f}"

                # 2. Activate trailing stop after good profit
                elif profit_pct >= self.config.trailing_activation_pct and not trailing_active:
                    trailing_active = True
                    # Move stop to breakeven + buffer
                    new_stop = entry_price + (current_atr * 0.5)
                    if new_stop > stop_price:
                        stop_price = new_stop

                # 3. Trail the stop if activated
                if trailing_active and exit_signal is None:
                    trail_stop = max_price - (current_atr * self.config.trailing_atr_mult)
                    if trail_stop > stop_price:
                        stop_price = trail_stop

                    if price <= stop_price:
                        exit_signal = Signal.CLOSE
                        exit_reason = f"Trailing stop at {stop_price:.2f} (locked {profit_pct:.1%} profit)"

                # 4. Death cross - trend reversal (mandatory exit)
                if exit_signal is None:
                    death_cross = (prev_short_ma >= prev_long_ma and
                                  current_short_ma < current_long_ma)
                    if death_cross:
                        exit_signal = Signal.CLOSE
                        exit_reason = "Death cross - trend reversal"

                # 5. Price breaks below long-term MA (trend broken)
                if exit_signal is None and price < current_long_ma:
                    exit_signal = Signal.CLOSE
                    exit_reason = f"Price broke below 200MA ({current_long_ma:.2f})"

                if exit_signal is not None:
                    in_position = False
                    signals.append(TradeSignal(
                        timestamp=ts,
                        signal=exit_signal,
                        reason=exit_reason,
                    ))

        return signals

    def validate_params(self) -> tuple[bool, str]:
        """Validate configuration parameters."""
        if self.config.long_ma_period <= self.config.short_ma_period:
            return False, "Long MA period must be greater than short MA period"
        if self.config.atr_stop_mult <= 0:
            return False, "ATR stop multiplier must be positive"
        return True, "Valid"


# Also create a simple buy-and-hold benchmark for comparison
class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold - benchmark for comparison.

    Just buys when there's positive swap and holds until the end.
    Shows what pure carry income would look like.
    """

    def __init__(self, min_swap: float = 0.0) -> None:
        self.min_swap = min_swap

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Buy at start if positive swap, hold until end."""
        signals: list[TradeSignal] = []

        if len(data) < 50:
            return signals

        # Find first day with positive swap
        for i in range(50, len(data)):
            swap = data["swap_long"].iloc[i]
            if swap > 0:
                signals.append(TradeSignal(
                    timestamp=data["timestamp"].iloc[i],
                    signal=Signal.LONG,
                    reason=f"Buy and hold entry (swap={swap:.4f})",
                ))
                break

        # Close at the end
        if signals:
            signals.append(TradeSignal(
                timestamp=data["timestamp"].iloc[-1],
                signal=Signal.CLOSE,
                reason="End of period",
            ))

        return signals

    def validate_params(self) -> tuple[bool, str]:
        return True, "Valid"
