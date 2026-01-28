"""Synthetic forex data generator for educational backtesting.

Generates realistic OHLCV + swap data using geometric Brownian motion
with configurable trend, volatility, and interest rate differentials.
"""

import numpy as np
import pandas as pd

from src.config.settings import DEFAULT_PAIRS, PairConfig
from src.data.interest_rates import InterestRateSimulator


class SyntheticDataGenerator:
    """Generates synthetic forex data with swap rates.

    Args:
        seed: Random seed for reproducibility.

    Example:
        >>> gen = SyntheticDataGenerator(seed=42)
        >>> df = gen.generate_pair(pair_config, days=365)
        >>> df.columns.tolist()
        ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'swap_long', 'swap_short']
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_pair(
        self,
        pair_config: PairConfig,
        days: int = 365,
        freq_hours: int = 1,
        dynamic_rates: bool = False,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV + swap data for one forex pair.

        Args:
            pair_config: Configuration for the pair to generate.
            days: Number of calendar days to generate.
            freq_hours: Bar frequency in hours (default 1 = hourly).
            dynamic_rates: If True, generate time-varying interest rates
                using the InterestRateSimulator. Swaps will change over time
                as simulated central banks adjust policy. If False, swaps
                are static (derived from initial rate config).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close,
            volume, swap_long, swap_short.
        """
        bars_per_day = 24 // freq_hours
        total_bars = days * bars_per_day

        timestamps = pd.date_range(
            start="2024-01-01",
            periods=total_bars,
            freq=f"{freq_hours}h",
        )

        # Filter out weekends (forex market closed Sat-Sun)
        timestamps = timestamps[timestamps.weekday < 5]
        total_bars = len(timestamps)

        closes = self._generate_prices(pair_config, total_bars)
        opens, highs, lows = self._generate_ohlc(closes, pair_config.volatility)
        volumes = self._generate_volume(total_bars)

        if dynamic_rates:
            swap_long, swap_short = self._dynamic_swaps(
                pair_config, days, timestamps
            )
        else:
            sl, ss = self._calculate_swaps(pair_config)
            swap_long = np.full(total_bars, sl)
            swap_short = np.full(total_bars, ss)

        df = pd.DataFrame(
            {
                "timestamp": timestamps[:total_bars],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "swap_long": swap_long,
                "swap_short": swap_short,
            }
        )
        return df

    def generate_all_pairs(
        self,
        pair_configs: list[PairConfig] | None = None,
        days: int = 365,
        dynamic_rates: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Generate synthetic data for multiple pairs.

        Args:
            pair_configs: List of pair configs. Uses defaults if None.
            days: Number of calendar days to generate.
            dynamic_rates: If True, use time-varying interest rates.

        Returns:
            Dictionary mapping pair name to its DataFrame.
        """
        configs = pair_configs or DEFAULT_PAIRS
        return {
            cfg.name: self.generate_pair(cfg, days, dynamic_rates=dynamic_rates)
            for cfg in configs
        }

    def _dynamic_swaps(
        self,
        config: PairConfig,
        days: int,
        timestamps: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate time-varying swap rates using simulated interest rates.

        Maps daily rate changes onto hourly bars. Each bar within the same
        calendar day shares that day's swap value.
        """
        rate_sim = InterestRateSimulator(seed=self.rng.integers(0, 2**31))
        rates_df = rate_sim.generate_pair_rates(config, days)

        # Map daily swaps onto hourly timestamps by date
        date_to_swap_long = dict(
            zip(rates_df["date"].dt.date, rates_df["swap_long"])
        )
        date_to_swap_short = dict(
            zip(rates_df["date"].dt.date, rates_df["swap_short"])
        )

        # Fallback to static swap for any date not in the rate sim
        static_long, static_short = self._calculate_swaps(config)
        swap_long = np.array(
            [date_to_swap_long.get(ts.date(), static_long) for ts in timestamps]
        )
        swap_short = np.array(
            [date_to_swap_short.get(ts.date(), static_short) for ts in timestamps]
        )
        return swap_long, swap_short

    def _generate_prices(
        self, config: PairConfig, total_bars: int
    ) -> np.ndarray:
        """Generate close prices using geometric Brownian motion with mean reversion."""
        mid_price = (config.price_min + config.price_max) / 2
        price_range = config.price_max - config.price_min

        # Drift based on trend
        drift_map = {"uptrend": 0.0001, "downtrend": -0.0001, "sideways": 0.0}
        drift = drift_map.get(config.trend, 0.0)

        # Mean reversion strength
        reversion_speed = 0.002

        prices = np.zeros(total_bars)
        prices[0] = mid_price + self.rng.uniform(-price_range * 0.1, price_range * 0.1)

        for i in range(1, total_bars):
            shock = self.rng.normal(0, config.volatility)
            reversion = reversion_speed * (mid_price - prices[i - 1])
            prices[i] = prices[i - 1] * (1 + drift + shock) + reversion

            # Soft clamp to configured range
            if prices[i] < config.price_min:
                prices[i] = config.price_min + abs(self.rng.normal(0, price_range * 0.01))
            elif prices[i] > config.price_max:
                prices[i] = config.price_max - abs(self.rng.normal(0, price_range * 0.01))

        return prices

    def _generate_ohlc(
        self, closes: np.ndarray, volatility: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate open, high, low from close prices."""
        n = len(closes)
        intra_vol = volatility * 0.5

        opens = np.zeros(n)
        opens[0] = closes[0] * (1 + self.rng.normal(0, intra_vol * 0.3))
        opens[1:] = closes[:-1] * (1 + self.rng.normal(0, intra_vol * 0.1, n - 1))

        # High is max of open/close plus a positive wick
        bar_max = np.maximum(opens, closes)
        bar_min = np.minimum(opens, closes)
        highs = bar_max * (1 + np.abs(self.rng.normal(0, intra_vol, n)))
        lows = bar_min * (1 - np.abs(self.rng.normal(0, intra_vol, n)))

        return opens, highs, lows

    def _generate_volume(self, total_bars: int) -> np.ndarray:
        """Generate realistic volume with intraday patterns."""
        base_volume = 1_000_000
        noise = self.rng.lognormal(mean=0, sigma=0.5, size=total_bars)
        return (base_volume * noise).astype(int)

    @staticmethod
    def _calculate_swaps(config: PairConfig) -> tuple[float, float]:
        """Calculate daily swap rates from interest rate differential.

        Swap formula (simplified):
            swap_long = (rate_base - rate_quote) / 365 * contract_size_factor
            swap_short = (rate_quote - rate_base) / 365 * contract_size_factor

        A positive swap_long means the trader earns by going long
        (base currency has higher rate).
        """
        rate_diff = config.interest_rate_base - config.interest_rate_quote
        # Normalize to daily pips-like value (simplified for education)
        swap_long = round(rate_diff / 365 * 100, 4)
        swap_short = round(-rate_diff / 365 * 100, 4)
        return swap_long, swap_short
