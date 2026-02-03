"""Feature extraction for ML signal filtering.

Computes context features from candle data at the moment a strategy
signal fires. All features are derived from existing indicators.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.indicators import atr, rsi, sma, adx

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "rsi_14",
    "atr_ratio",
    "ma_spread",
    "price_vs_200ma",
    "adx_14",
    "hour_of_day",
    "day_of_week",
    "regime_code",
    "swap_rate",
    "spread_bps",
    "vix",
    "dxy",
    "us10y_yield",
    "sp500_change_pct",
]


class FeatureExtractor:
    """Extracts context features from candle data.

    Args:
        cross_asset_fetcher: Optional CrossAssetFetcher for macro data.
            If None, cross-asset features use defaults.

    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract(df, "USD/JPY")
        >>> print(features["rsi_14"])
    """

    def __init__(self, cross_asset_fetcher=None) -> None:
        self._cross_asset_fetcher = cross_asset_fetcher

    def extract(
        self, df: pd.DataFrame, pair: str
    ) -> dict[str, float]:
        """Extract features from the latest candle data.

        Args:
            df: DataFrame with OHLCV + swap columns.
            pair: Currency pair name.

        Returns:
            Dict of feature_name -> value.
        """
        if df is None or len(df) < 220:
            return self._empty_features()

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        rsi_vals = rsi(close, period=14)
        rsi_14 = float(rsi_vals.iloc[-1]) if not np.isnan(rsi_vals.iloc[-1]) else 50.0

        # ATR ratio (expansion/contraction)
        atr_14 = atr(high, low, close, period=14)
        atr_60 = atr(high, low, close, period=60)
        last_atr_14 = float(atr_14.iloc[-1]) if not np.isnan(atr_14.iloc[-1]) else 1.0
        last_atr_60 = float(atr_60.iloc[-1]) if not np.isnan(atr_60.iloc[-1]) else 1.0
        atr_ratio = last_atr_14 / last_atr_60 if last_atr_60 > 0 else 1.0

        # Moving average spread
        ma_50 = sma(close, 50)
        ma_200 = sma(close, 200)
        price = float(close.iloc[-1])
        ma50_val = float(ma_50.iloc[-1]) if not np.isnan(ma_50.iloc[-1]) else price
        ma200_val = float(ma_200.iloc[-1]) if not np.isnan(ma_200.iloc[-1]) else price

        ma_spread = (ma50_val - ma200_val) / price if price > 0 else 0.0
        price_vs_200ma = (price - ma200_val) / ma200_val if ma200_val > 0 else 0.0

        # ADX
        adx_vals, _, _ = adx(high, low, close, period=14)
        adx_14 = float(adx_vals.iloc[-1]) if not np.isnan(adx_vals.iloc[-1]) else 20.0

        # Time features
        last_ts = df["timestamp"].iloc[-1]
        if isinstance(last_ts, pd.Timestamp):
            hour = last_ts.hour
            dow = last_ts.dayofweek
        else:
            hour = 12
            dow = 2

        # Swap rate
        swap_rate = float(df["swap_long"].iloc[-1]) if "swap_long" in df.columns else 0.0

        # Spread (estimated from high-low of last candle)
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])
        spread_bps = ((last_high - last_low) / price * 10000) if price > 0 else 0.0

        # Regime code (simplified: 0=trend_low, 1=trend_high, 2=range_low, 3=range_high)
        is_trending = adx_14 > 25
        is_high_vol = atr_ratio > 1.2
        if is_trending and not is_high_vol:
            regime_code = 0
        elif is_trending and is_high_vol:
            regime_code = 1
        elif not is_trending and not is_high_vol:
            regime_code = 2
        else:
            regime_code = 3

        features = {
            "rsi_14": rsi_14,
            "atr_ratio": atr_ratio,
            "ma_spread": ma_spread,
            "price_vs_200ma": price_vs_200ma,
            "adx_14": adx_14,
            "hour_of_day": float(hour),
            "day_of_week": float(dow),
            "regime_code": float(regime_code),
            "swap_rate": swap_rate,
            "spread_bps": spread_bps,
        }
        features.update(self._cross_asset_features())
        return features

    def _cross_asset_features(self) -> dict[str, float]:
        """Return cross-asset macro features.

        Uses the fetcher if available, otherwise returns defaults.
        """
        if self._cross_asset_fetcher is not None:
            try:
                data = self._cross_asset_fetcher.get()
                sp500_chg = (
                    (data.sp500 - data.sp500_prev_close)
                    / data.sp500_prev_close
                    * 100.0
                    if data.sp500_prev_close > 0
                    else 0.0
                )
                return {
                    "vix": data.vix,
                    "dxy": data.dxy,
                    "us10y_yield": data.us10y,
                    "sp500_change_pct": sp500_chg,
                }
            except Exception as e:
                logger.debug("Cross-asset fetch failed: %s", e)

        return {
            "vix": 20.0,
            "dxy": 100.0,
            "us10y_yield": 4.0,
            "sp500_change_pct": 0.0,
        }

    def to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in standard order.

        Args:
            features: Feature dict from extract().

        Returns:
            1-D numpy array of feature values.
        """
        return np.array(
            [features.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float64,
        )

    @staticmethod
    def _empty_features() -> dict[str, float]:
        return {name: 0.0 for name in FEATURE_NAMES}
