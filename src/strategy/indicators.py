"""Technical indicators used by trading strategies.

Indicators are math formulas applied to price data to help spot trends
and momentum. Think of them as "lenses" that highlight different
patterns in the raw price action.

SMA (Simple Moving Average):
    Average of the last N prices. If price is ABOVE the SMA, the trend
    is generally "up". If below, "down". We use this as a trend filter --
    only take carry trades in the direction of the trend.

RSI (Relative Strength Index):
    Measures how "overbought" or "oversold" a pair is, on a 0-100 scale.
    - RSI > 70 = overbought = price has gone up too fast, risky to buy
    - RSI < 30 = oversold = price has dropped too fast, risky to sell
    We use this to AVOID entering when the move is already exhausted.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price series (typically close prices).
        period: Number of bars to average over.

    Returns:
        Series with the rolling mean. First (period-1) values are NaN.

    Example:
        If period=3 and prices are [10, 11, 12, 13]:
        SMA = [NaN, NaN, 11.0, 12.0]
    """
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Measures momentum on a 0-100 scale.
    RSI = 100 - (100 / (1 + avg_gain / avg_loss))

    Args:
        series: Price series (typically close prices).
        period: Lookback period (default 14, the standard).

    Returns:
        Series with RSI values (0-100). First values are NaN.
    """
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))

    # Where avg_loss is 0, RSI = 100 (all gains, no losses)
    result = result.fillna(100.0)
    # First `period` values don't have enough data
    result.iloc[:period] = np.nan

    return result


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range -- measures volatility.

    True Range is the largest of:
    1. Current high - current low
    2. |Current high - previous close|
    3. |Current low - previous close|

    ATR = rolling average of True Range. Higher ATR = more volatile.
    Used later for setting stop-loss distances that adapt to market conditions.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period.

    Returns:
        Series with ATR values.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()
