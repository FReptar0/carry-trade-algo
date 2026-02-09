"""Historical data loader using yfinance.

This replaces synthetic data with REAL market data.

What changes vs synthetic:
    - Prices reflect actual market moves (trends, crashes, rallies)
    - Volume is real trading activity
    - Gaps exist (holidays, data outages)
    - Volatility clusters realistically (calm → storm → calm)

Data sources:
    - yfinance: Free historical forex data from Yahoo Finance
    - FRED API: Central bank interest rates (optional, needs API key)

Caching:
    Downloads are saved as parquet files so you only download once.
    Subsequent runs load from disk in milliseconds instead of waiting
    for the network.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Yahoo Finance ticker symbols for forex pairs
PAIR_TICKERS: dict[str, str] = {
    "USD/JPY": "USDJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/USD": "EURUSD=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "NZD/JPY": "NZDJPY=X",
    "CAD/JPY": "CADJPY=X",
}

# FRED series IDs for central bank policy rates
FRED_RATE_SERIES: dict[str, str] = {
    "USD": "FEDFUNDS",
    "JPY": "IRSTCI01JPM156N",
    "EUR": "ECBDFR",
    "AUD": "IRSTCI01AUM156N",
    "GBP": "IUDSOIA",
}


class HistoricalDataLoader:
    """Downloads and caches real forex data from Yahoo Finance.

    Args:
        cache_dir: Directory to store cached parquet files.

    Example:
        >>> loader = HistoricalDataLoader()
        >>> df = loader.load_pair("USD/JPY", start="2022-01-01", end="2024-12-31")
        >>> df.columns.tolist()
        ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'swap_long', 'swap_short']
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "forex").mkdir(exist_ok=True)
        (self.cache_dir / "rates").mkdir(exist_ok=True)

    def load_pair(
        self,
        pair: str,
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        interval: str = "1h",
        force_download: bool = False,
    ) -> pd.DataFrame:
        """Load historical data for a forex pair.

        First checks cache. If not cached (or force_download=True),
        downloads from Yahoo Finance and saves to cache.

        Args:
            pair: Forex pair name (e.g., "USD/JPY").
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).
            interval: Bar interval ("1h", "1d", etc.).
                Note: yfinance limits hourly data to ~2 years back.
            force_download: If True, skip cache and re-download.

        Returns:
            DataFrame with standardized columns matching our synthetic format.
        """
        cache_path = self._cache_path(pair, interval, start, end)

        if not force_download and cache_path.exists():
            logger.info(f"Loading {pair} from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

        logger.info(f"Downloading {pair} from Yahoo Finance...")
        df = self._download_pair(pair, start, end, interval)

        if len(df) > 0:
            df.to_parquet(cache_path, index=False)
            self._update_metadata(pair, interval, start, end, len(df))
            logger.info(f"Cached {len(df)} bars to {cache_path}")

        return df

    def load_multiple_pairs(
        self,
        pairs: list[str] | None = None,
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        interval: str = "1h",
        force_download: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Load data for multiple pairs.

        Args:
            pairs: List of pair names. Defaults to USD/JPY, AUD/JPY, EUR/USD.
            start: Start date.
            end: End date.
            interval: Bar interval.
            force_download: Re-download even if cached.

        Returns:
            Dict mapping pair name to DataFrame.
        """
        pairs = pairs or ["USD/JPY", "AUD/JPY", "EUR/USD"]
        result = {}
        for pair in pairs:
            try:
                df = self.load_pair(pair, start, end, interval, force_download)
                if len(df) > 0:
                    result[pair] = df
                else:
                    logger.warning(f"No data returned for {pair}")
            except Exception as e:
                logger.error(f"Failed to load {pair}: {e}")
        return result

    def load_interest_rates(
        self,
        currency: str,
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        fred_api_key: str | None = None,
    ) -> pd.Series | None:
        """Load central bank interest rate from FRED.

        Requires a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html

        Args:
            currency: Currency code (e.g., "USD", "JPY").
            start: Start date.
            end: End date.
            fred_api_key: Your FRED API key. If None, returns None.

        Returns:
            Series of daily interest rates, or None if no API key.
        """
        if fred_api_key is None:
            logger.info(f"No FRED API key provided. Skipping {currency} rates.")
            return None

        series_id = FRED_RATE_SERIES.get(currency)
        if series_id is None:
            logger.warning(f"No FRED series ID for {currency}")
            return None

        cache_path = self.cache_dir / "rates" / f"{series_id}.parquet"
        if cache_path.exists():
            logger.info(f"Loading {currency} rates from cache")
            return pd.read_parquet(cache_path).squeeze()

        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_api_key)
            rates = fred.get_series(series_id, start, end)
            rates = rates.dropna() / 100  # Convert percentage to decimal
            rates.to_frame("rate").to_parquet(cache_path)
            return rates
        except Exception as e:
            logger.error(f"FRED download failed for {currency}: {e}")
            return None

    def _download_pair(
        self, pair: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Download from yfinance and standardize columns."""
        import yfinance as yf

        ticker = PAIR_TICKERS.get(pair)
        if ticker is None:
            raise ValueError(f"Unknown pair: {pair}. Known: {list(PAIR_TICKERS.keys())}")

        raw = yf.download(
            ticker, start=start, end=end, interval=interval, progress=False
        )

        if raw.empty:
            return pd.DataFrame()

        # yfinance returns MultiIndex columns for single ticker too
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = pd.DataFrame({
            "timestamp": raw.index,
            "open": raw["Open"].values,
            "high": raw["High"].values,
            "low": raw["Low"].values,
            "close": raw["Close"].values,
            "volume": raw["Volume"].values if "Volume" in raw.columns else 0,
        }).reset_index(drop=True)

        # Remove timezone info for consistency
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # Add placeholder swap columns (will be filled from rates data)
        # Use approximate static rates for now
        swap_long, swap_short = self._estimate_swaps(pair)
        df["swap_long"] = swap_long
        df["swap_short"] = swap_short

        return df

    @staticmethod
    def _estimate_swaps(pair: str) -> tuple[float, float]:
        """Estimate swap values based on known historical rate differentials.

        These are rough approximations of 2022-2024 average rates.
        In a production system, you'd compute from actual daily rates.
        """
        # Approximate annual rate differentials (2022-2024 averages)
        differentials = {
            "USD/JPY": 0.05,    # Fed ~5% vs BOJ ~0%
            "AUD/JPY": 0.04,    # RBA ~4% vs BOJ ~0%
            "EUR/USD": -0.01,   # ECB ~4% vs Fed ~5%
            "EUR/JPY": 0.035,   # ECB ~4% vs BOJ ~0%
            "GBP/JPY": 0.05,    # BOE ~5% vs BOJ ~0%
            "NZD/JPY": 0.05,    # RBNZ ~5% vs BOJ ~0%
            "CAD/JPY": 0.05,    # BOC ~5% vs BOJ ~0%
        }
        diff = differentials.get(pair, 0.0)
        swap_long = round(diff / 365 * 100, 4)
        swap_short = round(-diff / 365 * 100, 4)
        return swap_long, swap_short

    def _cache_path(
        self, pair: str, interval: str, start: str, end: str
    ) -> Path:
        """Generate cache file path."""
        safe_pair = pair.replace("/", "")
        return self.cache_dir / "forex" / f"{safe_pair}_{interval}_{start}_{end}.parquet"

    def _update_metadata(
        self, pair: str, interval: str, start: str, end: str, bar_count: int
    ) -> None:
        """Update cache metadata JSON."""
        meta_path = self.cache_dir / "forex" / "metadata.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        meta[pair] = {
            "interval": interval,
            "start": start,
            "end": end,
            "bars": bar_count,
            "cached_at": datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    def clear_cache(self) -> None:
        """Delete all cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
            (self.cache_dir / "forex").mkdir()
            (self.cache_dir / "rates").mkdir()
            logger.info("Cache cleared")
