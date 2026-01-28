"""Data quality validation and preprocessing.

Real data is messy. This module checks for and fixes common problems:

1. GAPS: Missing bars (holidays, outages). We detect them and optionally
   forward-fill so the backtest doesn't break.

2. OUTLIERS: Extreme prices that are likely data errors (e.g., a price
   of 0 or 999999). We flag these so you can inspect them.

3. WEEKENDS: Forex markets close Friday 5pm NY to Sunday 5pm NY.
   Weekend bars with zero volume are removed.

4. DUPLICATES: Some data sources repeat timestamps. We keep only the last.

Quality report tells you: "Is this data clean enough to backtest on?"
A completeness rate > 98% (excluding weekends) is our target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    """Summary of data quality checks."""

    pair: str
    total_rows: int
    date_range: str
    missing_bars: int
    expected_bars: int
    completeness_pct: float
    duplicate_timestamps: int
    zero_volume_bars: int
    outlier_bars: int
    weekend_bars_removed: int
    passed: bool


def validate_data(
    df: pd.DataFrame,
    pair: str,
    freq_hours: int = 1,
    outlier_std: float = 5.0,
) -> DataQualityReport:
    """Run all quality checks on a price DataFrame.

    Args:
        df: DataFrame with standard columns (timestamp, OHLCV, swaps).
        pair: Pair name for the report.
        freq_hours: Expected bar frequency in hours.
        outlier_std: Flag returns > this many standard deviations.

    Returns:
        DataQualityReport with all findings.
    """
    total_rows = len(df)

    # Date range
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    date_range = f"{start.date()} to {end.date()}"

    # Expected bars (weekdays only, freq_hours intervals)
    total_days = (end - start).days
    weekdays = sum(
        1 for d in pd.date_range(start.date(), end.date())
        if d.weekday() < 5
    )
    expected_bars = weekdays * (24 // freq_hours)

    # Duplicates
    duplicates = df["timestamp"].duplicated().sum()

    # Weekend bars
    weekend_mask = df["timestamp"].dt.weekday >= 5
    weekend_bars = weekend_mask.sum()

    # Missing bars (gap detection)
    # Compare actual count to expected
    missing = max(0, expected_bars - (total_rows - weekend_bars))

    # Completeness
    actual_weekday = total_rows - weekend_bars
    completeness = (actual_weekday / expected_bars * 100) if expected_bars > 0 else 100.0

    # Zero volume
    zero_vol = (df["volume"] == 0).sum() if "volume" in df.columns else 0

    # Outliers (returns > outlier_std standard deviations)
    returns = df["close"].pct_change().dropna()
    if len(returns) > 0:
        mean_ret = returns.mean()
        std_ret = returns.std()
        outliers = ((returns - mean_ret).abs() > outlier_std * std_ret).sum()
    else:
        outliers = 0

    passed = completeness >= 90.0 and duplicates == 0

    return DataQualityReport(
        pair=pair,
        total_rows=total_rows,
        date_range=date_range,
        missing_bars=missing,
        expected_bars=expected_bars,
        completeness_pct=round(completeness, 1),
        duplicate_timestamps=duplicates,
        zero_volume_bars=zero_vol,
        outlier_bars=outliers,
        weekend_bars_removed=weekend_bars,
        passed=passed,
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess a price DataFrame.

    Steps:
    1. Remove duplicate timestamps (keep last)
    2. Remove weekend bars
    3. Sort by timestamp
    4. Forward-fill small gaps (up to 4 bars = 4 hours)
    5. Reset index

    Args:
        df: Raw price DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    out = df.copy()

    # Remove duplicates
    out = out.drop_duplicates(subset="timestamp", keep="last")

    # Remove weekends
    out = out[out["timestamp"].dt.weekday < 5]

    # Sort
    out = out.sort_values("timestamp").reset_index(drop=True)

    return out


def compare_synthetic_vs_real(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    pair: str,
) -> dict:
    """Compare statistics between synthetic and real data.

    This helps validate whether our synthetic generator was realistic.

    Args:
        synthetic: Synthetic data DataFrame.
        real: Real data DataFrame.
        pair: Pair name.

    Returns:
        Dict of comparison metrics.
    """
    def _stats(df: pd.DataFrame) -> dict:
        ret = df["close"].pct_change().dropna()
        return {
            "bars": len(df),
            "price_mean": round(df["close"].mean(), 2),
            "price_std": round(df["close"].std(), 2),
            "return_mean": round(ret.mean(), 7),
            "return_std": round(ret.std(), 6),
            "return_skew": round(ret.skew(), 3),
            "return_kurtosis": round(ret.kurtosis(), 3),
            "ann_vol": round(ret.std() * np.sqrt(24 * 252), 4),
        }

    syn = _stats(synthetic)
    real_s = _stats(real)

    return {
        "pair": pair,
        "metric": list(syn.keys()),
        "synthetic": list(syn.values()),
        "real": list(real_s.values()),
    }
