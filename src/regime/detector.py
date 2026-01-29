"""Market regime detection for adaptive trading.

REGIME DETECTION PHILOSOPHY:
============================
Markets alternate between different "regimes" or states:
1. Trending: Price moves consistently in one direction
2. Ranging: Price oscillates between support/resistance
3. High Volatility: Large price swings, more risk
4. Low Volatility: Small price swings, more predictable

The best strategy depends on the current regime:
- Trending + Low Vol: Best for carry trades (trend + predictable)
- Trending + High Vol: Risky but profitable if right direction
- Ranging + Low Vol: Collect swap, avoid directional trades
- Ranging + High Vol: Worst case - avoid trading

This module detects regimes using:
- ADX: Average Directional Index for trend strength
- ATR: Average True Range for volatility level
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.indicators import adx, atr


class VolatilityRegime(Enum):
    """Volatility classification based on ATR vs historical average.

    HIGH: Current ATR > 1.5x the 60-day average (dangerous, widen stops)
    NORMAL: ATR within 0.7x to 1.5x average (typical conditions)
    LOW: Current ATR < 0.7x average (calm, can tighten stops)
    """

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TrendRegime(Enum):
    """Trend classification based on ADX and directional indicators.

    STRONG_UPTREND: ADX > 25 and +DI > -DI (trade long aggressively)
    WEAK_UPTREND: 15 < ADX < 25 and +DI > -DI (trade long cautiously)
    RANGING: ADX < 15 (avoid directional trades)
    WEAK_DOWNTREND: 15 < ADX < 25 and -DI > +DI (avoid longs)
    STRONG_DOWNTREND: ADX > 25 and -DI > +DI (definitely avoid longs)
    """

    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class CompositeRegime(Enum):
    """Combined regime classification (4 states for simplicity).

    These map trend + volatility to actionable trading guidance:

    TREND_LOW_VOL: Best for carry trades
        - Strong trend gives direction
        - Low volatility means predictable moves
        - Recommendation: Trade with full size

    TREND_HIGH_VOL: Profitable but risky
        - Strong trend, but volatile swings
        - Can capture big moves, but need wider stops
        - Recommendation: Reduce size, widen stops

    RANGE_LOW_VOL: Collect carry, avoid trades
        - No clear direction
        - But calm, so existing positions safe
        - Recommendation: Hold existing, no new entries

    RANGE_HIGH_VOL: Worst regime
        - No direction + big swings = whipsaw
        - High chance of stop-outs
        - Recommendation: Do not trade
    """

    TREND_LOW_VOL = "trend_low_vol"
    TREND_HIGH_VOL = "trend_high_vol"
    RANGE_LOW_VOL = "range_low_vol"
    RANGE_HIGH_VOL = "range_high_vol"


@dataclass
class RegimeState:
    """Complete market regime state at a point in time."""

    timestamp: datetime
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    composite_regime: CompositeRegime
    confidence: float  # 0-1 confidence in classification

    # Raw values for debugging/visualization
    adx_value: float
    plus_di: float
    minus_di: float
    atr_value: float
    atr_avg: float  # Historical ATR average for comparison

    @property
    def is_favorable(self) -> bool:
        """Check if regime is favorable for carry trades."""
        return self.composite_regime in (
            CompositeRegime.TREND_LOW_VOL,
            CompositeRegime.TREND_HIGH_VOL,
        )

    @property
    def should_avoid_trading(self) -> bool:
        """Check if trading should be avoided."""
        return self.composite_regime == CompositeRegime.RANGE_HIGH_VOL

    @property
    def trend_direction(self) -> str:
        """Get trend direction as simple string."""
        if self.trend_regime in (TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND):
            return "up"
        elif self.trend_regime in (
            TrendRegime.STRONG_DOWNTREND,
            TrendRegime.WEAK_DOWNTREND,
        ):
            return "down"
        return "neutral"


class RegimeDetector:
    """Detects market regime from OHLC price data.

    Example:
        detector = RegimeDetector()
        regime = detector.detect(df)
        if regime.should_avoid_trading:
            print("Skip trading in this regime")
        elif regime.is_favorable:
            print(f"Good to trade: {regime.composite_regime.value}")
    """

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        atr_lookback: int = 60,
        adx_strong_threshold: float = 25.0,
        adx_weak_threshold: float = 15.0,
        vol_high_mult: float = 1.5,
        vol_low_mult: float = 0.7,
    ) -> None:
        """Initialize regime detector.

        Args:
            adx_period: Period for ADX calculation (default 14).
            atr_period: Period for ATR calculation (default 14).
            atr_lookback: Days to average ATR for volatility comparison (default 60).
            adx_strong_threshold: ADX above this = strong trend (default 25).
            adx_weak_threshold: ADX below this = ranging (default 15).
            vol_high_mult: ATR multiple for high volatility (default 1.5).
            vol_low_mult: ATR multiple for low volatility (default 0.7).
        """
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.adx_strong_threshold = adx_strong_threshold
        self.adx_weak_threshold = adx_weak_threshold
        self.vol_high_mult = vol_high_mult
        self.vol_low_mult = vol_low_mult

    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect current regime from price data.

        Args:
            data: DataFrame with columns: timestamp, high, low, close.
                  Must have at least (atr_lookback + atr_period) rows.

        Returns:
            RegimeState with current regime classification.

        Raises:
            ValueError: If insufficient data.
        """
        min_rows = self.atr_lookback + self.atr_period
        if len(data) < min_rows:
            raise ValueError(f"Need at least {min_rows} rows, got {len(data)}")

        # Calculate indicators
        adx_series, plus_di_series, minus_di_series = adx(
            data["high"], data["low"], data["close"], self.adx_period
        )
        atr_series = atr(
            data["high"], data["low"], data["close"], self.atr_period
        )

        # Get current values
        current_adx = adx_series.iloc[-1]
        current_plus_di = plus_di_series.iloc[-1]
        current_minus_di = minus_di_series.iloc[-1]
        current_atr = atr_series.iloc[-1]

        # Calculate historical ATR average for comparison
        atr_avg = atr_series.iloc[-self.atr_lookback :].mean()

        # Detect volatility regime
        volatility_regime = self._detect_volatility(current_atr, atr_avg)

        # Detect trend regime
        trend_regime = self._detect_trend(
            current_adx, current_plus_di, current_minus_di
        )

        # Combine into composite regime
        composite_regime = self._get_composite(volatility_regime, trend_regime)

        # Calculate confidence (how clear-cut is the classification)
        confidence = self._calculate_confidence(
            current_adx, current_atr, atr_avg
        )

        # Get timestamp
        timestamp = data["timestamp"].iloc[-1]
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        return RegimeState(
            timestamp=timestamp,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            composite_regime=composite_regime,
            confidence=confidence,
            adx_value=float(current_adx) if not pd.isna(current_adx) else 0.0,
            plus_di=float(current_plus_di) if not pd.isna(current_plus_di) else 0.0,
            minus_di=float(current_minus_di) if not pd.isna(current_minus_di) else 0.0,
            atr_value=float(current_atr) if not pd.isna(current_atr) else 0.0,
            atr_avg=float(atr_avg) if not pd.isna(atr_avg) else 0.0,
        )

    def detect_series(self, data: pd.DataFrame) -> list[RegimeState]:
        """Detect regime for each bar in the data.

        Args:
            data: DataFrame with OHLC data.

        Returns:
            List of RegimeState, one per bar (after warmup period).
        """
        min_rows = self.atr_lookback + self.atr_period
        if len(data) < min_rows:
            return []

        regimes = []
        for i in range(min_rows, len(data)):
            subset = data.iloc[: i + 1]
            regime = self.detect(subset)
            regimes.append(regime)

        return regimes

    def _detect_volatility(
        self, current_atr: float, atr_avg: float
    ) -> VolatilityRegime:
        """Classify volatility based on ATR vs historical average."""
        if pd.isna(current_atr) or pd.isna(atr_avg) or atr_avg == 0:
            return VolatilityRegime.NORMAL

        ratio = current_atr / atr_avg

        if ratio >= self.vol_high_mult:
            return VolatilityRegime.HIGH
        elif ratio <= self.vol_low_mult:
            return VolatilityRegime.LOW
        return VolatilityRegime.NORMAL

    def _detect_trend(
        self, adx_value: float, plus_di: float, minus_di: float
    ) -> TrendRegime:
        """Classify trend based on ADX and directional indicators."""
        if pd.isna(adx_value):
            return TrendRegime.RANGING

        is_uptrend = plus_di > minus_di

        if adx_value >= self.adx_strong_threshold:
            return TrendRegime.STRONG_UPTREND if is_uptrend else TrendRegime.STRONG_DOWNTREND
        elif adx_value >= self.adx_weak_threshold:
            return TrendRegime.WEAK_UPTREND if is_uptrend else TrendRegime.WEAK_DOWNTREND
        return TrendRegime.RANGING

    def _get_composite(
        self, vol_regime: VolatilityRegime, trend_regime: TrendRegime
    ) -> CompositeRegime:
        """Combine volatility and trend into composite regime."""
        # Determine if trending (any trend direction, not ranging)
        is_trending = trend_regime not in (TrendRegime.RANGING,)

        # Determine if high volatility
        is_high_vol = vol_regime == VolatilityRegime.HIGH

        if is_trending and not is_high_vol:
            return CompositeRegime.TREND_LOW_VOL
        elif is_trending and is_high_vol:
            return CompositeRegime.TREND_HIGH_VOL
        elif not is_trending and not is_high_vol:
            return CompositeRegime.RANGE_LOW_VOL
        else:
            return CompositeRegime.RANGE_HIGH_VOL

    def _calculate_confidence(
        self, adx_value: float, current_atr: float, atr_avg: float
    ) -> float:
        """Calculate confidence in regime classification.

        Higher confidence when:
        - ADX is clearly above/below thresholds (not borderline)
        - ATR ratio is clearly in high/low/normal range
        """
        if pd.isna(adx_value) or pd.isna(current_atr):
            return 0.0

        # ADX confidence: distance from threshold boundaries
        adx_mid = (self.adx_strong_threshold + self.adx_weak_threshold) / 2
        adx_distance = abs(adx_value - adx_mid)
        adx_range = self.adx_strong_threshold - self.adx_weak_threshold
        adx_conf = min(adx_distance / adx_range, 1.0)

        # Volatility confidence: distance from 1.0 ratio
        if atr_avg > 0:
            vol_ratio = current_atr / atr_avg
            vol_distance = abs(vol_ratio - 1.0)
            vol_conf = min(vol_distance / 0.5, 1.0)  # 0.5 = half range
        else:
            vol_conf = 0.0

        # Average confidence
        return (adx_conf + vol_conf) / 2
