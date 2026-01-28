"""Tests for technical indicators."""

import numpy as np
import pandas as pd

from src.strategy.indicators import atr, rsi, sma


class TestSMA:
    def test_simple_case(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert result.iloc[3] == 3.0  # (2+3+4)/3
        assert result.iloc[4] == 4.0  # (3+4+5)/3

    def test_length_preserved(self):
        s = pd.Series(range(100), dtype=float)
        result = sma(s, period=20)
        assert len(result) == 100

    def test_first_values_are_nan(self):
        s = pd.Series(range(50), dtype=float)
        result = sma(s, period=10)
        assert result.iloc[:9].isna().all()
        assert not pd.isna(result.iloc[9])


class TestRSI:
    def test_all_gains_equals_100(self):
        s = pd.Series(range(1, 30), dtype=float)  # monotonically increasing
        result = rsi(s, period=14)
        # After warmup, RSI should be near 100 (all gains)
        assert result.iloc[-1] > 95

    def test_range_0_to_100(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(200).cumsum() + 100)
        result = rsi(s, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_first_values_nan(self):
        s = pd.Series(range(50), dtype=float)
        result = rsi(s, period=14)
        assert result.iloc[:14].isna().all()


class TestATR:
    def test_positive_values(self):
        n = 100
        high = pd.Series(np.random.uniform(101, 105, n))
        low = pd.Series(np.random.uniform(95, 99, n))
        close = pd.Series(np.random.uniform(99, 101, n))
        result = atr(high, low, close, period=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_length_preserved(self):
        n = 50
        h = pd.Series(np.ones(n) * 2)
        l = pd.Series(np.ones(n) * 1)
        c = pd.Series(np.ones(n) * 1.5)
        result = atr(h, l, c, period=14)
        assert len(result) == n
