"""Tests for data loader and preprocessor.

These tests work offline -- they test the loader interface, caching logic,
and preprocessor without requiring network access.
"""

import pandas as pd
import numpy as np

from src.data.loader import HistoricalDataLoader, PAIR_TICKERS
from src.data.preprocessor import (
    DataQualityReport,
    clean_data,
    compare_synthetic_vs_real,
    validate_data,
)


def _make_sample_df(rows: int = 500, pair_price: float = 148.0) -> pd.DataFrame:
    """Create a sample DataFrame matching our standard format."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = pair_price + np.random.randn(rows).cumsum() * 0.1
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(rows) * 0.05,
        "high": close + abs(np.random.randn(rows) * 0.1),
        "low": close - abs(np.random.randn(rows) * 0.1),
        "close": close,
        "volume": np.random.randint(1000, 100000, rows),
        "swap_long": 0.0137,
        "swap_short": -0.0137,
    })


class TestHistoricalDataLoader:
    def test_known_tickers(self):
        assert "USD/JPY" in PAIR_TICKERS
        assert "AUD/JPY" in PAIR_TICKERS
        assert "EUR/USD" in PAIR_TICKERS

    def test_estimate_swaps_positive(self):
        sl, ss = HistoricalDataLoader._estimate_swaps("USD/JPY")
        assert sl > 0  # USD rate > JPY rate
        assert ss < 0

    def test_estimate_swaps_negative(self):
        sl, ss = HistoricalDataLoader._estimate_swaps("EUR/USD")
        assert sl < 0  # EUR rate < USD rate
        assert ss > 0

    def test_cache_dir_created(self, tmp_path):
        loader = HistoricalDataLoader(cache_dir=str(tmp_path / "test_cache"))
        assert (tmp_path / "test_cache" / "forex").exists()
        assert (tmp_path / "test_cache" / "rates").exists()

    def test_cache_path_generation(self, tmp_path):
        loader = HistoricalDataLoader(cache_dir=str(tmp_path / "cache"))
        path = loader._cache_path("USD/JPY", "1h", "2022-01-01", "2024-12-31")
        assert "USDJPY" in str(path)
        assert path.suffix == ".parquet"

    def test_clear_cache(self, tmp_path):
        loader = HistoricalDataLoader(cache_dir=str(tmp_path / "cache"))
        # Create a dummy file
        (tmp_path / "cache" / "forex" / "dummy.txt").write_text("test")
        loader.clear_cache()
        assert not (tmp_path / "cache" / "forex" / "dummy.txt").exists()
        # Dirs should be recreated
        assert (tmp_path / "cache" / "forex").exists()


class TestValidateData:
    def test_returns_report(self):
        df = _make_sample_df()
        report = validate_data(df, "USD/JPY", freq_hours=1)
        assert isinstance(report, DataQualityReport)
        assert report.pair == "USD/JPY"
        assert report.total_rows == 500

    def test_completeness_calculated(self):
        df = _make_sample_df()
        report = validate_data(df, "USD/JPY", freq_hours=1)
        assert report.completeness_pct > 0

    def test_no_duplicates(self):
        df = _make_sample_df()
        report = validate_data(df, "USD/JPY")
        assert report.duplicate_timestamps == 0

    def test_detects_duplicates(self):
        df = _make_sample_df()
        # Add a duplicate
        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = validate_data(dup, "USD/JPY")
        assert report.duplicate_timestamps == 1

    def test_outlier_detection(self):
        df = _make_sample_df()
        # Inject an extreme price
        df.loc[100, "close"] = df["close"].mean() * 2
        report = validate_data(df, "USD/JPY", outlier_std=3.0)
        assert report.outlier_bars > 0


class TestCleanData:
    def test_removes_duplicates(self):
        df = _make_sample_df()
        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        cleaned = clean_data(dup)
        # clean_data also removes weekends, so compare against
        # clean version of original (without the dup)
        cleaned_orig = clean_data(df)
        assert len(cleaned) == len(cleaned_orig)

    def test_removes_weekends(self):
        df = _make_sample_df(rows=200)
        cleaned = clean_data(df)
        weekdays = cleaned["timestamp"].dt.weekday
        assert (weekdays < 5).all()

    def test_sorted_by_timestamp(self):
        df = _make_sample_df()
        # Shuffle
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        cleaned = clean_data(shuffled)
        assert cleaned["timestamp"].is_monotonic_increasing


class TestCompareSyntheticVsReal:
    def test_returns_dict(self):
        syn = _make_sample_df(rows=300, pair_price=148.0)
        real = _make_sample_df(rows=400, pair_price=150.0)
        comp = compare_synthetic_vs_real(syn, real, "USD/JPY")
        assert comp["pair"] == "USD/JPY"
        assert "bars" in comp["metric"]
        assert len(comp["synthetic"]) == len(comp["real"])

    def test_different_bar_counts(self):
        syn = _make_sample_df(rows=100)
        real = _make_sample_df(rows=200)
        comp = compare_synthetic_vs_real(syn, real, "TEST")
        idx = comp["metric"].index("bars")
        assert comp["synthetic"][idx] == 100
        assert comp["real"][idx] == 200
