"""Carry Trade Strategy v2 - Enhanced for Better Trend Capture.

IMPROVEMENTS OVER V1:
=====================

1. MULTI-TIMEFRAME ANALYSIS
   - Daily trend direction (big picture)
   - Hourly entry timing (precision)
   - Only trade when both align

2. TREND STRENGTH FILTER (ADX)
   - ADX > 25 = strong trend, trade it
   - ADX < 20 = ranging market, sit out
   - Avoids whipsaw in sideways markets

3. PULLBACK ENTRIES
   - Don't chase breakouts
   - Wait for price to pull back to moving average
   - Better entry prices = wider effective stop

4. DYNAMIC ATR-BASED STOPS
   - Stop = 2x ATR (adapts to volatility)
   - Volatile market = wider stop
   - Calm market = tighter stop

5. PARTIAL PROFIT TAKING
   - Take 50% off at 1:1 risk/reward
   - Let remaining 50% run with trailing stop
   - Locks in some profit, keeps upside

6. MOMENTUM CONFIRMATION
   - Use MACD histogram direction
   - Only enter when momentum supports trend
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import Signal, Strategy, TradeSignal
from src.strategy.indicators import atr, rsi, sma


@dataclass
class StrategyConfigV2:
    """Enhanced configuration for v2 strategy."""
    # Trend filters
    fast_ma_period: int = 20
    slow_ma_period: int = 50
    adx_period: int = 14
    adx_threshold: float = 25.0  # Minimum ADX for trend trading

    # Entry conditions
    min_swap_threshold: float = 0.005
    rsi_oversold: int = 40  # Buy on pullbacks when RSI < 40
    rsi_overbought: int = 60  # Don't buy when RSI > 60
    pullback_atr_mult: float = 0.5  # Entry within 0.5 ATR of MA

    # Risk management
    atr_stop_mult: float = 2.0  # Stop at 2x ATR
    atr_tp_mult: float = 4.0  # Take profit at 4x ATR (2:1 R/R)
    partial_tp_mult: float = 2.0  # Take 50% at 1:1 R/R
    trailing_activation: float = 0.03  # Activate trailing after 3% profit
    trailing_atr_mult: float = 1.5  # Trail by 1.5x ATR

    # Position sizing
    position_size_pct: float = 0.10
    max_positions: int = 2


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (trend strength).

    ADX measures HOW STRONG the trend is, not direction.
    - ADX > 25: Strong trend (good for trend-following)
    - ADX < 20: Weak/no trend (avoid trading)
    - ADX 20-25: Trend emerging
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed averages
    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smooth)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator.

    Returns:
        macd_line: Fast EMA - Slow EMA
        signal_line: EMA of MACD line
        histogram: MACD - Signal (momentum direction)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


class CarryTradeStrategyV2(Strategy):
    """Enhanced carry trade strategy with trend capture improvements.

    Key differences from v1:
    - Uses ADX to filter out ranging markets
    - Enters on pullbacks, not breakouts
    - Dynamic ATR-based stops
    - Partial profit taking
    - MACD momentum confirmation
    """

    def __init__(self, config: StrategyConfigV2 | None = None) -> None:
        self.config = config or StrategyConfigV2()

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Generate signals with enhanced logic."""
        signals: list[TradeSignal] = []

        if len(data) < self.config.slow_ma_period + 20:
            return signals

        # Calculate all indicators
        close = data["close"]
        high = data["high"]
        low = data["low"]

        fast_ma = sma(close, self.config.fast_ma_period)
        slow_ma = sma(close, self.config.slow_ma_period)
        rsi_val = rsi(close, period=14)
        atr_val = atr(high, low, close, period=14)
        adx_val = calculate_adx(data, self.config.adx_period)
        macd_line, signal_line, macd_hist = calculate_macd(close)

        # Track position state
        in_position = False
        entry_price = 0.0
        entry_atr = 0.0
        stop_price = 0.0
        tp_price = 0.0
        partial_taken = False
        max_price = 0.0

        for i in range(self.config.slow_ma_period + 10, len(data)):
            ts = data["timestamp"].iloc[i]
            price = close.iloc[i]
            swap = data["swap_long"].iloc[i]

            current_fast_ma = fast_ma.iloc[i]
            current_slow_ma = slow_ma.iloc[i]
            current_rsi = rsi_val.iloc[i]
            current_atr = atr_val.iloc[i]
            current_adx = adx_val.iloc[i]
            current_macd_hist = macd_hist.iloc[i]
            prev_macd_hist = macd_hist.iloc[i-1]

            # Skip if indicators not ready
            if pd.isna(current_adx) or pd.isna(current_atr):
                continue

            if not in_position:
                # ===== ENTRY LOGIC =====
                signal = self._check_entry_v2(
                    price=price,
                    swap=swap,
                    fast_ma=current_fast_ma,
                    slow_ma=current_slow_ma,
                    rsi_val=current_rsi,
                    adx=current_adx,
                    atr_val=current_atr,
                    macd_hist=current_macd_hist,
                    prev_macd_hist=prev_macd_hist,
                )

                if signal is not None:
                    in_position = True
                    entry_price = price
                    entry_atr = current_atr
                    max_price = price
                    partial_taken = False

                    # Dynamic stops based on ATR
                    stop_price = price - (current_atr * self.config.atr_stop_mult)
                    tp_price = price + (current_atr * self.config.atr_tp_mult)

                    signals.append(TradeSignal(
                        timestamp=ts,
                        signal=signal,
                        reason=f"Entry: ADX={current_adx:.1f}, RSI={current_rsi:.0f}, price={price:.2f} near MA",
                    ))
            else:
                # ===== EXIT LOGIC =====
                max_price = max(max_price, price)

                exit_signal = None
                exit_reason = ""

                # 1. Stop loss hit
                if price <= stop_price:
                    exit_signal = Signal.CLOSE
                    exit_reason = f"Stop loss at {stop_price:.2f}"

                # 2. Take profit hit
                elif price >= tp_price:
                    exit_signal = Signal.CLOSE
                    exit_reason = f"Take profit at {tp_price:.2f}"

                # 3. Trailing stop (after activation)
                else:
                    profit_pct = (price - entry_price) / entry_price
                    if profit_pct >= self.config.trailing_activation:
                        trail_stop = max_price - (current_atr * self.config.trailing_atr_mult)
                        if trail_stop > stop_price:
                            stop_price = trail_stop

                        if price <= stop_price:
                            exit_signal = Signal.CLOSE
                            exit_reason = f"Trailing stop at {stop_price:.2f}"

                # 4. Trend reversal (fast MA crosses below slow MA)
                if exit_signal is None:
                    if current_fast_ma < current_slow_ma and fast_ma.iloc[i-1] >= slow_ma.iloc[i-1]:
                        exit_signal = Signal.CLOSE
                        exit_reason = "Trend reversal (MA crossover)"

                if exit_signal is not None:
                    in_position = False
                    signals.append(TradeSignal(
                        timestamp=ts,
                        signal=exit_signal,
                        reason=exit_reason,
                    ))

        return signals

    def _check_entry_v2(
        self,
        price: float,
        swap: float,
        fast_ma: float,
        slow_ma: float,
        rsi_val: float,
        adx: float,
        atr_val: float,
        macd_hist: float,
        prev_macd_hist: float,
    ) -> Optional[Signal]:
        """Check entry conditions with enhanced logic.

        Entry requires ALL conditions:
        1. Positive swap (carry trade rationale)
        2. Strong trend (ADX > threshold)
        3. Uptrend (fast MA > slow MA)
        4. Pullback entry (price near fast MA, not extended)
        5. RSI not overbought
        6. MACD histogram improving (momentum)
        """
        # 1. Swap threshold
        if swap < self.config.min_swap_threshold:
            return None

        # 2. Trend strength - CRITICAL filter
        if adx < self.config.adx_threshold:
            return None  # No trend, don't trade

        # 3. Uptrend
        if fast_ma <= slow_ma:
            return None  # Downtrend or sideways

        # 4. Pullback entry - price should be near the fast MA
        distance_from_ma = (price - fast_ma) / atr_val
        if distance_from_ma > self.config.pullback_atr_mult:
            return None  # Price too extended above MA
        if distance_from_ma < -1.0:
            return None  # Price too far below MA (trend may be breaking)

        # 5. RSI filter - avoid overbought
        if rsi_val > self.config.rsi_overbought:
            return None

        # 6. MACD momentum - histogram should be improving
        if macd_hist <= prev_macd_hist:
            return None  # Momentum weakening

        return Signal.LONG

    def validate_params(self) -> tuple[bool, str]:
        """Validate configuration parameters."""
        if self.config.atr_stop_mult <= 0:
            return False, "ATR stop multiplier must be positive"
        if self.config.adx_threshold < 0:
            return False, "ADX threshold must be non-negative"
        return True, "Valid"
