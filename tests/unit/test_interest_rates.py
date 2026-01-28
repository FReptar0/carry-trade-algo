"""Tests for interest rate simulator."""

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_PAIRS, PairConfig
from src.data.interest_rates import InterestRateSimulator


def _make_sim(seed: int = 42) -> InterestRateSimulator:
    return InterestRateSimulator(seed=seed)


class TestGenerateRatePath:
    def test_returns_correct_length(self):
        sim = _make_sim()
        path = sim.generate_rate_path(initial_rate=0.05, long_run_mean=0.04, days=365)
        assert len(path) == 365

    def test_starts_at_initial_rate(self):
        sim = _make_sim()
        path = sim.generate_rate_path(initial_rate=0.0525, long_run_mean=0.04, days=100)
        assert path[0] == 0.0525

    def test_rates_non_negative(self):
        """Interest rates should never go below zero (floor at 0%)."""
        sim = _make_sim()
        path = sim.generate_rate_path(
            initial_rate=0.001,
            long_run_mean=0.0,
            volatility=0.01,
            days=500,
        )
        assert (path >= 0.0).all()

    def test_rates_change_at_meeting_intervals(self):
        """Rates should only change at meeting dates, not between them."""
        sim = _make_sim()
        path = sim.generate_rate_path(
            initial_rate=0.05,
            long_run_mean=0.03,
            volatility=0.005,
            days=100,
            meeting_interval_days=42,
        )
        # Between day 1 and day 41, rate should be constant
        assert np.all(path[0:42] == path[0])

    def test_step_size_quantization(self):
        """Rate changes should be multiples of step_size_bps."""
        sim = _make_sim()
        path = sim.generate_rate_path(
            initial_rate=0.05,
            long_run_mean=0.03,
            volatility=0.01,
            days=365,
            meeting_interval_days=42,
            step_size_bps=25,
        )
        unique_rates = np.unique(path)
        # All rate differences from initial should be multiples of 0.0025
        step = 0.0025
        for rate in unique_rates:
            diff = abs(rate - 0.05)
            remainder = diff % step
            assert remainder < 1e-10 or abs(remainder - step) < 1e-10, (
                f"Rate {rate} is not a 25bps multiple from 0.05"
            )

    def test_mean_reversion_pulls_toward_target(self):
        """With strong mean reversion and long horizon, rates should approach the mean."""
        sim = _make_sim()
        path = sim.generate_rate_path(
            initial_rate=0.10,
            long_run_mean=0.03,
            mean_reversion_speed=0.15,
            volatility=0.001,
            days=1000,
            meeting_interval_days=42,
        )
        # Final rate should be closer to 0.03 than the initial 0.10
        assert abs(path[-1] - 0.03) < abs(0.10 - 0.03)

    def test_reproducibility(self):
        p1 = _make_sim(seed=99).generate_rate_path(0.05, 0.04, days=100)
        p2 = _make_sim(seed=99).generate_rate_path(0.05, 0.04, days=100)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = _make_sim(seed=1).generate_rate_path(0.05, 0.03, volatility=0.01, days=365)
        p2 = _make_sim(seed=2).generate_rate_path(0.05, 0.03, volatility=0.01, days=365)
        assert not np.array_equal(p1, p2)


class TestGeneratePairRates:
    def test_returns_dataframe(self):
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=100)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=100)
        expected = ["date", "rate_base", "rate_quote", "rate_differential", "swap_long", "swap_short"]
        assert list(df.columns) == expected

    def test_correct_row_count(self):
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=200)
        assert len(df) == 200

    def test_differential_equals_base_minus_quote(self):
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=100)
        computed = (df["rate_base"] - df["rate_quote"]).round(4)
        pd.testing.assert_series_equal(computed, df["rate_differential"], check_names=False)

    def test_swap_long_positive_when_base_higher(self):
        """USD/JPY: USD rate (5.25%) >> JPY rate (0.25%) => swap_long > 0 initially."""
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=10)
        assert df["swap_long"].iloc[0] > 0

    def test_swap_long_and_short_opposite_signs(self):
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=100)
        # Where differential != 0, signs should be opposite
        nonzero = df["rate_differential"] != 0
        assert (
            (df.loc[nonzero, "swap_long"] * df.loc[nonzero, "swap_short"]) <= 0
        ).all()

    def test_rates_are_non_negative(self):
        sim = _make_sim()
        df = sim.generate_pair_rates(DEFAULT_PAIRS[0], days=365)
        assert (df["rate_base"] >= 0).all()
        assert (df["rate_quote"] >= 0).all()


class TestGenerateAllPairRates:
    def test_returns_dict_with_all_pairs(self):
        sim = _make_sim()
        result = sim.generate_all_pair_rates(DEFAULT_PAIRS, days=50)
        assert len(result) == 3
        assert set(result.keys()) == {"USD/JPY", "AUD/JPY", "EUR/USD"}

    def test_each_value_is_dataframe(self):
        sim = _make_sim()
        result = sim.generate_all_pair_rates(DEFAULT_PAIRS, days=50)
        for name, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
            assert len(df) == 50
