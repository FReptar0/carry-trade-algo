"""Carry Trade Strategy v4 - Multi-Timeframe with Regime Awareness.

V4 IMPROVEMENTS OVER V3:
========================
1. Weekly trend for direction, daily for entry timing
2. Regime-aware parameter adaptation
3. Only trade when weekly and daily agree
4. Dynamic position sizing based on regime confidence

MULTI-TIMEFRAME PHILOSOPHY:
===========================
- WEEKLY: Determines the "big picture" trend direction
- DAILY: Times entries within the weekly trend

Rule: Never trade against the weekly trend.
- If weekly is up, only look for long entries
- If weekly is down, avoid longs entirely (for carry trades)
- If weekly is neutral, wait for clarity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import Signal, Strategy, TradeSignal
from src.strategy.indicators import atr, rsi, sma
from src.regime.detector import RegimeDetector, RegimeState, CompositeRegime
from src.regime.adapters import RegimeAdapter


@dataclass
class StrategyConfigV4:
    """Configuration for multi-timeframe strategy."""

    # Weekly trend settings (from merged data)
    weekly_ma_threshold: float = 0.01  # 1% above weekly MA = uptrend

    # Daily entry settings
    daily_ma_period: int = 20  # Fast daily MA
    daily_pullback_pct: float = 0.02  # Entry on 2% pullback from daily MA

    # Entry conditions
    min_swap_threshold: float = 0.003
    max_rsi: int = 75  # Avoid extreme overbought

    # Risk management (base values, adjusted by regime)
    base_atr_mult: float = 3.0
    trailing_activation_pct: float = 0.04  # Start trailing after 4%
    trailing_atr_mult: float = 2.0

    # Position sizing
    base_position_size_pct: float = 0.08
    max_positions: int = 2

    # Multi-timeframe requirements
    require_weekly_alignment: bool = True
    require_regime_favorable: bool = True


class CarryTradeStrategyV4(Strategy):
    """Multi-timeframe carry trade strategy with regime awareness.

    Key features:
    1. Weekly trend filter (from pre-merged data)
    2. Daily entry timing (pullbacks to daily MA)
    3. Regime-based parameter adaptation
    4. Dynamic stops and position sizing

    Expects data with columns:
    - Standard OHLCV + swap columns
    - weekly_trend: "up", "down", "neutral"
    - weekly_ma: Weekly moving average value
    - weekly_close: Last weekly close
    """

    def __init__(
        self,
        config: StrategyConfigV4 | None = None,
        regime_detector: RegimeDetector | None = None,
        regime_adapter: RegimeAdapter | None = None,
    ) -> None:
        self.config = config or StrategyConfigV4()
        self.regime_detector = regime_detector or RegimeDetector()
        self.regime_adapter = regime_adapter or RegimeAdapter()

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Generate signals using multi-timeframe + regime logic."""
        signals: list[TradeSignal] = []

        # Validate required columns
        required_cols = ["weekly_trend", "weekly_ma", "weekly_close"]
        has_weekly = all(col in data.columns for col in required_cols)

        if not has_weekly:
            # Fall back to single-timeframe if weekly data not available
            return self._generate_signals_single_tf(data)

        # Need enough data for indicators
        min_bars = max(self.config.daily_ma_period + 20, 80)
        if len(data) < min_bars:
            return signals

        # Calculate daily indicators
        close = data["close"]
        high = data["high"]
        low = data["low"]

        daily_ma = sma(close, self.config.daily_ma_period)
        rsi_val = rsi(close, period=14)
        atr_val = atr(high, low, close, period=14)

        # Track position state
        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        max_price = 0.0
        trailing_active = False
        current_regime: Optional[RegimeState] = None

        for i in range(min_bars, len(data)):
            ts = data["timestamp"].iloc[i]
            price = close.iloc[i]
            swap = data["swap_long"].iloc[i]

            # Get current values
            current_daily_ma = daily_ma.iloc[i]
            current_rsi = rsi_val.iloc[i]
            current_atr = atr_val.iloc[i]
            weekly_trend = data["weekly_trend"].iloc[i]
            weekly_ma = data["weekly_ma"].iloc[i]

            # Skip if indicators not ready
            if pd.isna(current_atr) or pd.isna(current_daily_ma):
                continue

            # Detect regime (expensive, do periodically)
            if i % 5 == 0:  # Every 5 bars
                try:
                    current_regime = self.regime_detector.detect(data.iloc[: i + 1])
                except ValueError:
                    current_regime = None

            if not in_position:
                # ===== ENTRY LOGIC =====
                entry_signal = self._check_entry_v4(
                    price=price,
                    swap=swap,
                    daily_ma=current_daily_ma,
                    rsi_val=current_rsi,
                    atr_val=current_atr,
                    weekly_trend=weekly_trend,
                    weekly_ma=weekly_ma,
                    regime=current_regime,
                )

                if entry_signal is not None:
                    in_position = True
                    entry_price = price
                    max_price = price
                    trailing_active = False

                    # Regime-adjusted stop
                    stop_mult = self.config.base_atr_mult
                    if current_regime:
                        stop_mult = self.regime_adapter.get_adjusted_stop(
                            stop_mult, current_regime
                        )

                    stop_price = price - (current_atr * stop_mult)

                    # Build reason string
                    regime_str = (
                        current_regime.composite_regime.value
                        if current_regime
                        else "unknown"
                    )
                    signals.append(
                        TradeSignal(
                            timestamp=ts,
                            signal=entry_signal,
                            reason=f"V4 Entry: weekly={weekly_trend}, regime={regime_str}, price={price:.2f}",
                        )
                    )
            else:
                # ===== EXIT LOGIC =====
                max_price = max(max_price, price)
                profit_pct = (price - entry_price) / entry_price

                exit_signal = None
                exit_reason = ""

                # 1. Stop loss
                if price <= stop_price:
                    exit_signal = Signal.CLOSE
                    exit_reason = f"Stop loss at {stop_price:.2f}"

                # 2. Trailing stop activation
                elif profit_pct >= self.config.trailing_activation_pct:
                    if not trailing_active:
                        trailing_active = True
                        # Move stop to breakeven + buffer
                        new_stop = entry_price + (current_atr * 0.5)
                        stop_price = max(stop_price, new_stop)

                    # Update trailing stop
                    trail_stop = max_price - (current_atr * self.config.trailing_atr_mult)
                    stop_price = max(stop_price, trail_stop)

                    if price <= stop_price:
                        exit_signal = Signal.CLOSE
                        exit_reason = f"Trailing stop at {stop_price:.2f} (+{profit_pct:.1%})"

                # 3. Weekly trend reversal (critical exit)
                if exit_signal is None and weekly_trend == "down":
                    exit_signal = Signal.CLOSE
                    exit_reason = "Weekly trend turned down"

                # 4. Regime turned unfavorable
                if exit_signal is None and current_regime:
                    if current_regime.should_avoid_trading and profit_pct < 0.02:
                        exit_signal = Signal.CLOSE
                        exit_reason = f"Unfavorable regime: {current_regime.composite_regime.value}"

                # 5. Price breaks below daily MA (trend weakness)
                if exit_signal is None:
                    if price < current_daily_ma and close.iloc[i - 1] >= daily_ma.iloc[i - 1]:
                        if profit_pct < 0.01:  # Only exit if not meaningfully profitable
                            exit_signal = Signal.CLOSE
                            exit_reason = "Price broke below daily MA"

                if exit_signal is not None:
                    in_position = False
                    signals.append(
                        TradeSignal(
                            timestamp=ts,
                            signal=exit_signal,
                            reason=exit_reason,
                        )
                    )

        return signals

    def _check_entry_v4(
        self,
        price: float,
        swap: float,
        daily_ma: float,
        rsi_val: float,
        atr_val: float,
        weekly_trend: str,
        weekly_ma: float,
        regime: Optional[RegimeState],
    ) -> Optional[Signal]:
        """Check entry conditions with multi-timeframe + regime logic.

        Entry requires:
        1. Positive swap (carry trade rationale)
        2. Weekly trend is UP (don't fight the big trend)
        3. Daily price pulled back to daily MA (good entry timing)
        4. RSI not overbought
        5. Regime is favorable (optional but recommended)
        """
        # 1. Swap threshold
        if swap < self.config.min_swap_threshold:
            return None

        # 2. Weekly trend filter (CRITICAL)
        if self.config.require_weekly_alignment:
            if weekly_trend != "up":
                return None  # Don't long against weekly downtrend

            # Price should be above weekly MA
            if not pd.isna(weekly_ma) and price < weekly_ma:
                return None

        # 3. Daily pullback entry
        # Price should be near (within 2%) or below daily MA
        distance_from_ma = (price - daily_ma) / daily_ma
        if distance_from_ma > self.config.daily_pullback_pct:
            return None  # Price too extended above daily MA

        # 4. RSI filter
        if rsi_val > self.config.max_rsi:
            return None

        # 5. Regime filter
        if self.config.require_regime_favorable and regime:
            if regime.should_avoid_trading:
                return None
            if regime.trend_direction == "down":
                return None

        return Signal.LONG

    def _generate_signals_single_tf(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Fallback: generate signals without weekly data (like V3)."""
        signals: list[TradeSignal] = []

        if len(data) < 80:
            return signals

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Use longer MA as proxy for weekly trend
        fast_ma = sma(close, 50)
        slow_ma = sma(close, 200)
        rsi_val = rsi(close, period=14)
        atr_val = atr(high, low, close, period=14)

        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        max_price = 0.0

        for i in range(200, len(data)):
            ts = data["timestamp"].iloc[i]
            price = close.iloc[i]
            swap = data["swap_long"].iloc[i]

            current_fast = fast_ma.iloc[i]
            current_slow = slow_ma.iloc[i]
            current_rsi = rsi_val.iloc[i]
            current_atr = atr_val.iloc[i]

            if pd.isna(current_atr):
                continue

            if not in_position:
                # Entry: uptrend + positive swap + not overbought
                if swap >= self.config.min_swap_threshold:
                    if price > current_fast > current_slow:
                        if current_rsi < self.config.max_rsi:
                            in_position = True
                            entry_price = price
                            max_price = price
                            stop_price = price - (current_atr * self.config.base_atr_mult)

                            signals.append(
                                TradeSignal(
                                    timestamp=ts,
                                    signal=Signal.LONG,
                                    reason=f"V4 fallback entry: price={price:.2f}",
                                )
                            )
            else:
                max_price = max(max_price, price)

                # Exit conditions
                if price <= stop_price:
                    in_position = False
                    signals.append(
                        TradeSignal(
                            timestamp=ts,
                            signal=Signal.CLOSE,
                            reason=f"Stop loss at {stop_price:.2f}",
                        )
                    )
                elif current_fast < current_slow:
                    in_position = False
                    signals.append(
                        TradeSignal(
                            timestamp=ts,
                            signal=Signal.CLOSE,
                            reason="Trend reversal (MA crossover)",
                        )
                    )

        return signals

    def validate_params(self) -> tuple[bool, str]:
        """Validate configuration parameters."""
        if self.config.base_atr_mult <= 0:
            return False, "ATR stop multiplier must be positive"
        if self.config.daily_ma_period < 5:
            return False, "Daily MA period must be at least 5"
        return True, "Valid"
