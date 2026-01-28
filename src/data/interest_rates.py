"""Simulated interest rate generator for carry trade backtesting.

Central banks set policy rates that determine the "carry" in carry trades.
This module simulates how those rates evolve over time:

- Rates follow a mean-reverting process (Vasicek model), reflecting how
  central banks tend to normalize rates toward a long-run equilibrium.
- Discrete rate decisions (hikes/cuts of 25bps) are layered on top,
  simulating scheduled central bank meetings.
- The swap rate a trader receives each day is derived from the rate
  differential between the two currencies in the pair.

Financial concepts:
    - **Policy rate**: The overnight interest rate set by a central bank
      (e.g., Fed Funds Rate, BOJ rate). This is the base cost of borrowing.
    - **Rate differential**: base_rate - quote_rate. A positive differential
      means long positions earn swap; negative means they pay.
    - **Basis point (bps)**: 1/100th of a percent. A 25bps hike means
      the rate goes from e.g. 5.00% to 5.25%.
    - **Vasicek model**: dr = a(b - r)dt + sigma*dW, where a = mean reversion
      speed, b = long-run mean, sigma = volatility, dW = random shock.
"""

import numpy as np
import pandas as pd

from src.config.settings import PairConfig


class InterestRateSimulator:
    """Simulates central bank interest rate paths with regime changes.

    Generates daily interest rate series for two currencies in a forex pair,
    including discrete central bank rate decisions.

    Args:
        seed: Random seed for reproducibility.

    Example:
        >>> sim = InterestRateSimulator(seed=42)
        >>> rates = sim.generate_rate_path(
        ...     initial_rate=0.0525,
        ...     long_run_mean=0.04,
        ...     volatility=0.005,
        ...     days=365,
        ...     meeting_interval_days=42,
        ... )
        >>> rates.shape
        (365,)
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_rate_path(
        self,
        initial_rate: float,
        long_run_mean: float,
        volatility: float = 0.005,
        mean_reversion_speed: float = 0.05,
        days: int = 365,
        meeting_interval_days: int = 42,
        step_size_bps: int = 25,
    ) -> np.ndarray:
        """Generate a daily interest rate path using a Vasicek-inspired model.

        The rate evolves continuously via mean reversion + noise, but actual
        rate *changes* only happen at discrete "meeting" dates (simulating
        central bank decisions). Between meetings, the rate is held constant.

        Args:
            initial_rate: Starting annual interest rate (e.g., 0.0525 = 5.25%).
            long_run_mean: Equilibrium rate the process reverts toward.
            volatility: Standard deviation of daily rate shocks (pre-discretization).
            mean_reversion_speed: How fast rates pull back to long_run_mean.
                Higher = faster reversion. Range: 0.01 (slow) to 0.2 (fast).
            days: Number of calendar days to simulate.
            meeting_interval_days: Days between central bank meetings.
                Real-world: ~42 days (Fed meets ~8x/year).
            step_size_bps: Rate changes are rounded to this many basis points.
                Real-world: usually 25bps, sometimes 50bps in aggressive cycles.

        Returns:
            Array of daily interest rates with length = days.
        """
        rates = np.zeros(days)
        rates[0] = initial_rate

        # Generate the underlying continuous process first
        continuous = np.zeros(days)
        continuous[0] = initial_rate
        for i in range(1, days):
            dr = (
                mean_reversion_speed * (long_run_mean - continuous[i - 1])
                + volatility * self.rng.normal()
            )
            continuous[i] = max(continuous[i - 1] + dr, 0.0)

        # Discretize: only change at meeting dates, in step_size_bps increments
        current_rate = initial_rate
        step = step_size_bps / 10_000  # Convert bps to decimal

        for i in range(days):
            if i > 0 and i % meeting_interval_days == 0:
                # Central bank decides based on where continuous process points
                target = continuous[i]
                diff = target - current_rate
                # Round to nearest step_size_bps
                n_steps = round(diff / step)
                current_rate = current_rate + n_steps * step
                current_rate = max(current_rate, 0.0)
            rates[i] = current_rate

        return rates

    def generate_pair_rates(
        self,
        pair_config: PairConfig,
        days: int = 365,
    ) -> pd.DataFrame:
        """Generate daily interest rates for both currencies in a pair.

        Also computes the daily swap values derived from the rate differential.

        Args:
            pair_config: Forex pair configuration with initial rates.
            days: Number of calendar days.

        Returns:
            DataFrame with columns: date, rate_base, rate_quote,
            rate_differential, swap_long, swap_short.
        """
        dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

        base_rates = self.generate_rate_path(
            initial_rate=pair_config.interest_rate_base,
            long_run_mean=pair_config.interest_rate_base * 0.9,
            volatility=0.003,
            days=days,
        )

        quote_rates = self.generate_rate_path(
            initial_rate=pair_config.interest_rate_quote,
            long_run_mean=pair_config.interest_rate_quote * 1.1,
            volatility=0.002,
            days=days,
        )

        differential = base_rates - quote_rates
        swap_long = np.round(differential / 365 * 100, 4)
        swap_short = np.round(-differential / 365 * 100, 4)

        return pd.DataFrame(
            {
                "date": dates,
                "rate_base": np.round(base_rates, 4),
                "rate_quote": np.round(quote_rates, 4),
                "rate_differential": np.round(differential, 4),
                "swap_long": swap_long,
                "swap_short": swap_short,
            }
        )

    def generate_all_pair_rates(
        self,
        pair_configs: list[PairConfig],
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Generate interest rate data for multiple pairs.

        Args:
            pair_configs: List of pair configurations.
            days: Number of calendar days.

        Returns:
            Dictionary mapping pair name to rate DataFrame.
        """
        return {cfg.name: self.generate_pair_rates(cfg, days) for cfg in pair_configs}
