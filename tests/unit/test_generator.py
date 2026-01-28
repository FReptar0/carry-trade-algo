"""Tests for synthetic data generator."""

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_PAIRS, PairConfig
from src.data.generator import SyntheticDataGenerator


def _make_generator(seed: int = 42) -> SyntheticDataGenerator:
    return SyntheticDataGenerator(seed=seed)


def _default_pair() -> PairConfig:
    return DEFAULT_PAIRS[0]  # USD/JPY


class TestGeneratePair:
    def test_returns_dataframe(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        expected = ["timestamp", "open", "high", "low", "close", "volume", "swap_long", "swap_short"]
        assert list(df.columns) == expected

    def test_no_weekend_bars(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        weekdays = df["timestamp"].dt.weekday
        assert (weekdays < 5).all(), "Found weekend bars"

    def test_high_ge_low(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        assert (df["high"] >= df["low"]).all()

    def test_high_ge_open_and_close(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_low_le_open_and_close(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_positive_volume(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        assert (df["volume"] > 0).all()

    def test_prices_within_range(self):
        gen = _make_generator()
        pair = _default_pair()
        df = gen.generate_pair(pair, days=365)
        # Allow small margin for generation noise
        margin = (pair.price_max - pair.price_min) * 0.05
        assert df["close"].min() >= pair.price_min - margin
        assert df["close"].max() <= pair.price_max + margin

    def test_reproducibility(self):
        df1 = _make_generator(seed=123).generate_pair(_default_pair(), days=30)
        df2 = _make_generator(seed=123).generate_pair(_default_pair(), days=30)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = _make_generator(seed=1).generate_pair(_default_pair(), days=30)
        df2 = _make_generator(seed=2).generate_pair(_default_pair(), days=30)
        assert not df1["close"].equals(df2["close"])

    def test_year_data_has_sufficient_bars(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=365)
        # ~260 weekdays * 24 hours = ~6240 bars
        assert len(df) > 5000


class TestSwapCalculation:
    def test_positive_carry_long(self):
        """USD/JPY: USD rate > JPY rate => swap_long > 0."""
        gen = _make_generator()
        pair = DEFAULT_PAIRS[0]  # USD/JPY
        df = gen.generate_pair(pair, days=10)
        assert df["swap_long"].iloc[0] > 0
        assert df["swap_short"].iloc[0] < 0

    def test_negative_carry_long(self):
        """EUR/USD: EUR rate < USD rate => swap_long < 0."""
        gen = _make_generator()
        pair = DEFAULT_PAIRS[2]  # EUR/USD
        df = gen.generate_pair(pair, days=10)
        assert df["swap_long"].iloc[0] < 0
        assert df["swap_short"].iloc[0] > 0

    def test_swap_values_are_constant(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30)
        assert df["swap_long"].nunique() == 1
        assert df["swap_short"].nunique() == 1


class TestDynamicRates:
    def test_dynamic_rates_produces_varying_swaps(self):
        """With dynamic rates, swap values should change over time."""
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=365, dynamic_rates=True)
        # Dynamic swaps should have more than 1 unique value
        assert df["swap_long"].nunique() > 1

    def test_dynamic_rates_columns_match_static(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30, dynamic_rates=True)
        expected = ["timestamp", "open", "high", "low", "close", "volume", "swap_long", "swap_short"]
        assert list(df.columns) == expected

    def test_dynamic_rates_no_nan(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=365, dynamic_rates=True)
        assert not df.isna().any().any()

    def test_static_swaps_are_constant(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30, dynamic_rates=False)
        assert df["swap_long"].nunique() == 1
        assert df["swap_short"].nunique() == 1


class TestGenerateAllPairs:
    def test_returns_dict(self):
        gen = _make_generator()
        result = gen.generate_all_pairs(days=30)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_keys_match_pair_names(self):
        gen = _make_generator()
        result = gen.generate_all_pairs(days=30)
        assert set(result.keys()) == {"USD/JPY", "AUD/JPY", "EUR/USD"}

    def test_custom_configs(self):
        gen = _make_generator()
        custom = [DEFAULT_PAIRS[0]]
        result = gen.generate_all_pairs(pair_configs=custom, days=10)
        assert len(result) == 1
        assert "USD/JPY" in result

    def test_dynamic_rates_passthrough(self):
        gen = _make_generator()
        result = gen.generate_all_pairs(days=365, dynamic_rates=True)
        for name, df in result.items():
            assert df["swap_long"].nunique() > 1, f"{name} swaps not dynamic"


class TestEdgeCases:
    def test_single_day(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=1)
        assert len(df) > 0

    def test_all_three_pairs_generate(self):
        """Each default pair should produce valid data."""
        gen = _make_generator()
        for pair in DEFAULT_PAIRS:
            df = gen.generate_pair(pair, days=30)
            assert len(df) > 0
            assert (df["high"] >= df["low"]).all(), f"{pair.name} high < low"

    def test_four_hour_frequency(self):
        gen = _make_generator()
        df = gen.generate_pair(_default_pair(), days=30, freq_hours=4)
        # 4h bars = 6 per day, ~22 weekdays in 30 days = ~132
        assert len(df) < 200
