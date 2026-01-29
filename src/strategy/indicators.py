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


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index with Directional Indicators.

    ADX measures TREND STRENGTH (not direction):
    - ADX > 25: Strong trend (good for trend-following)
    - ADX 20-25: Trend emerging
    - ADX < 20: Weak/no trend (ranging market)

    +DI and -DI show trend DIRECTION:
    - +DI > -DI: Uptrend
    - -DI > +DI: Downtrend

    Uses Wilder's smoothing (exponential moving average with alpha = 1/period).

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period (default 14).

    Returns:
        Tuple of (adx, plus_di, minus_di) series.

    Example:
        adx_val, plus_di, minus_di = adx(df['high'], df['low'], df['close'])
        if adx_val.iloc[-1] > 25 and plus_di.iloc[-1] > minus_di.iloc[-1]:
            # Strong uptrend
    """
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM: up move > down move and up move > 0
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    # -DM: down move > up move and down move > 0
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    alpha = 1.0 / period

    # Smoothed True Range
    smoothed_tr = true_range.ewm(alpha=alpha, adjust=False).mean()

    # Smoothed +DM and -DM
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # Directional Indicators
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    # Directional Index
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))

    # ADX = smoothed DX
    adx_values = dx.ewm(alpha=alpha, adjust=False).mean()

    # Set initial values to NaN
    adx_values.iloc[: period * 2] = np.nan
    plus_di.iloc[:period] = np.nan
    minus_di.iloc[:period] = np.nan

    return adx_values, plus_di, minus_di
