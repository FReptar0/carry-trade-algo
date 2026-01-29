"""Multi-timeframe data loading and alignment.

MULTI-TIMEFRAME PHILOSOPHY:
===========================
Using multiple timeframes helps us:
1. See the "big picture" (weekly trend direction)
2. Time entries precisely (daily pullbacks)
3. Avoid trading against the larger trend

KEY CHALLENGE: AVOIDING LOOKAHEAD BIAS
======================================
When merging weekly data into daily bars, we must NOT use future weekly closes.
At any given daily bar, we can only know the PREVIOUS week's close.

Example of WRONG approach (lookahead bias):
    Daily bar: Monday Jan 8
    Weekly close used: Friday Jan 12 <-- WRONG! We can't see the future

Example of CORRECT approach:
    Daily bar: Monday Jan 8
    Weekly close used: Friday Jan 5 <-- CORRECT! Last completed week
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from src.data.loader import HistoricalDataLoader
from src.strategy.indicators import sma


@dataclass
class TimeframeData:
    """Data for a single timeframe."""

    interval: str  # "1w", "1d", "4h", "1h"
    data: pd.DataFrame
    indicators: dict[str, pd.Series]  # Pre-calculated indicators


@dataclass
class MultiTimeframeContext:
    """Aligned multi-timeframe data ready for strategy use."""

    daily: pd.DataFrame  # Daily data with weekly context merged in
    weekly_raw: pd.DataFrame  # Raw weekly data for reference

    # Metadata
    pair: str
    start_date: str
    end_date: str


class MultiTimeframeLoader:
    """Loads and aligns data across multiple timeframes.

    Example:
        loader = MultiTimeframeLoader()
        context = loader.load_aligned("USD/JPY", "2021-01-01", "2024-12-31")

        # Daily data now has weekly context columns
        print(context.daily.columns)
        # ['timestamp', 'open', 'high', 'low', 'close', 'volume',
        #  'swap_long', 'swap_short',
        #  'weekly_close', 'weekly_ma', 'weekly_trend']
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        weekly_ma_period: int = 20,  # 20 weeks ~ 100 trading days
    ) -> None:
        """Initialize loader.

        Args:
            cache_dir: Directory for cached data.
            weekly_ma_period: Period for weekly moving average.
        """
        self.base_loader = HistoricalDataLoader(cache_dir=cache_dir)
        self.weekly_ma_period = weekly_ma_period

    def load_aligned(
        self,
        pair: str,
        start: str = "2021-01-01",
        end: str = "2024-12-31",
    ) -> MultiTimeframeContext:
        """Load multi-timeframe data with proper alignment.

        Args:
            pair: Currency pair (e.g., "USD/JPY").
            start: Start date.
            end: End date.

        Returns:
            MultiTimeframeContext with aligned data.

        Raises:
            ValueError: If insufficient data.
        """
        # Load daily data
        daily = self.base_loader.load_pair(pair, start=start, end=end, interval="1d")
        if len(daily) < 50:
            raise ValueError(f"Insufficient daily data: {len(daily)} bars")

        # Resample to weekly
        weekly = self._resample_to_weekly(daily)
        if len(weekly) < self.weekly_ma_period:
            raise ValueError(f"Insufficient weekly data: {len(weekly)} bars")

        # Calculate weekly indicators
        weekly = self._add_weekly_indicators(weekly)

        # Merge weekly context into daily (no lookahead!)
        daily_with_weekly = self._merge_weekly_to_daily(daily, weekly)

        return MultiTimeframeContext(
            daily=daily_with_weekly,
            weekly_raw=weekly,
            pair=pair,
            start_date=start,
            end_date=end,
        )

    def _resample_to_weekly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV to weekly bars.

        Uses Friday close as the weekly close (standard forex convention).
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(daily["timestamp"]):
            daily = daily.copy()
            daily["timestamp"] = pd.to_datetime(daily["timestamp"])

        # Set timestamp as index for resampling
        df = daily.set_index("timestamp")

        # Resample to weekly (week ending Friday)
        weekly = df.resample("W-FRI").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "swap_long": "last",
                "swap_short": "last",
            }
        ).dropna()

        # Reset index to get timestamp back as column
        weekly = weekly.reset_index()
        weekly = weekly.rename(columns={"timestamp": "week_end"})

        return weekly

    def _add_weekly_indicators(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to weekly data."""
        weekly = weekly.copy()

        # Weekly moving average
        weekly["weekly_ma"] = sma(weekly["close"], self.weekly_ma_period)

        # Weekly trend direction
        weekly["weekly_trend"] = "neutral"
        weekly.loc[weekly["close"] > weekly["weekly_ma"], "weekly_trend"] = "up"
        weekly.loc[weekly["close"] < weekly["weekly_ma"], "weekly_trend"] = "down"

        # Weekly momentum (close vs previous week)
        weekly["weekly_momentum"] = weekly["close"].pct_change()

        return weekly

    def _merge_weekly_to_daily(
        self, daily: pd.DataFrame, weekly: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge weekly context into daily data without lookahead bias.

        CRITICAL: At any daily bar, we can only see COMPLETED weeks.
        For a daily bar on Monday Jan 8, we use the week ending Friday Jan 5.
        """
        daily = daily.copy()

        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(daily["timestamp"]):
            daily["timestamp"] = pd.to_datetime(daily["timestamp"])

        # Add columns to hold weekly context
        daily["weekly_close"] = np.nan
        daily["weekly_ma"] = np.nan
        daily["weekly_trend"] = "neutral"
        daily["weekly_high"] = np.nan
        daily["weekly_low"] = np.nan

        # For each daily bar, find the most recent COMPLETED week
        for i, row in daily.iterrows():
            daily_date = row["timestamp"]

            # Find weeks that ended BEFORE this daily bar
            valid_weeks = weekly[weekly["week_end"] < daily_date]

            if len(valid_weeks) == 0:
                continue

            # Use the most recent completed week
            last_week = valid_weeks.iloc[-1]

            daily.at[i, "weekly_close"] = last_week["close"]
            daily.at[i, "weekly_ma"] = last_week["weekly_ma"]
            daily.at[i, "weekly_trend"] = last_week["weekly_trend"]
            daily.at[i, "weekly_high"] = last_week["high"]
            daily.at[i, "weekly_low"] = last_week["low"]

        return daily

    def verify_no_lookahead(self, context: MultiTimeframeContext) -> bool:
        """Verify that no lookahead bias exists in the merged data.

        For each daily bar, check that the weekly_close used is from
        a week that ended BEFORE that daily bar.

        Returns:
            True if no lookahead detected.

        Raises:
            ValueError: If lookahead bias detected.
        """
        daily = context.daily
        weekly = context.weekly_raw

        for _, row in daily.iterrows():
            daily_date = row["timestamp"]
            weekly_close = row["weekly_close"]

            if pd.isna(weekly_close):
                continue

            # Find which week this close came from
            matching_weeks = weekly[weekly["close"] == weekly_close]
            if len(matching_weeks) == 0:
                continue

            week_end = matching_weeks.iloc[-1]["week_end"]

            if week_end >= daily_date:
                raise ValueError(
                    f"Lookahead bias detected! Daily bar {daily_date} "
                    f"uses week ending {week_end}"
                )

        return True


def get_aligned_signals(
    daily_signal: str,
    weekly_trend: str,
    require_alignment: bool = True,
) -> tuple[bool, str]:
    """Check if daily signal aligns with weekly trend.

    Args:
        daily_signal: Daily signal ("long", "short", "neutral")
        weekly_trend: Weekly trend ("up", "down", "neutral")
        require_alignment: Whether to require alignment.

    Returns:
        Tuple of (should_trade, reason).

    Example:
        aligned, reason = get_aligned_signals("long", "up")
        # (True, "Daily long aligned with weekly uptrend")
    """
    if not require_alignment:
        return True, "Alignment not required"

    # Check alignment
    if daily_signal == "long" and weekly_trend == "up":
        return True, "Daily long aligned with weekly uptrend"
    elif daily_signal == "short" and weekly_trend == "down":
        return True, "Daily short aligned with weekly downtrend"
    elif daily_signal == "long" and weekly_trend == "down":
        return False, "Daily long conflicts with weekly downtrend"
    elif daily_signal == "short" and weekly_trend == "up":
        return False, "Daily short conflicts with weekly uptrend"
    elif weekly_trend == "neutral":
        return False, "Weekly trend is neutral, waiting for direction"
    else:
        return False, "No clear alignment"
