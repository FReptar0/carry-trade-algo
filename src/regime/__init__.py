"""Regime detection module for market condition classification.

This module provides tools to detect whether the market is:
- Trending vs Ranging (using ADX)
- High vs Low volatility (using ATR)

The combination creates a 4-state regime classifier that helps
the strategy adapt to current market conditions.
"""

from src.regime.detector import (
    CompositeRegime,
    RegimeDetector,
    RegimeState,
    TrendRegime,
    VolatilityRegime,
)
from src.regime.adapters import RegimeAdapter, RegimeAdaptedParams

__all__ = [
    "RegimeDetector",
    "RegimeState",
    "VolatilityRegime",
    "TrendRegime",
    "CompositeRegime",
    "RegimeAdapter",
    "RegimeAdaptedParams",
]
