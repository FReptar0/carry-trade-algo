"""Cross-asset macro data fetcher (VIX, DXY, 10Y Yield, S&P 500).

Fetches macro indicators from Yahoo Finance V8 API using only stdlib
(urllib). Results are cached per tick so multiple pairs share one
HTTP round-trip.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Yahoo Finance V8 chart endpoint (free, no auth)
_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Symbol mapping
SYMBOLS = {
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "sp500": "^GSPC",
}

# Defaults when fetch fails
DEFAULTS = {
    "vix": 20.0,
    "dxy": 100.0,
    "us10y": 4.0,
    "sp500": 5000.0,
}


@dataclass
class CrossAssetData:
    """Container for cross-asset macro data.

    Attributes:
        vix: CBOE Volatility Index.
        dxy: US Dollar Index.
        dxy_prev_close: DXY previous day close (for daily change calc).
        us10y: US 10-Year Treasury yield.
        sp500: S&P 500 price.
        sp500_prev_close: S&P 500 previous close (for change calc).
        fetched_at: Unix timestamp when data was fetched.
        is_default: True if data is from defaults (fetch failed).
    """

    vix: float
    dxy: float
    dxy_prev_close: float
    us10y: float
    sp500: float
    sp500_prev_close: float
    fetched_at: float
    is_default: bool


class CrossAssetFetcher:
    """Fetches and caches cross-asset macro data from Yahoo Finance.

    Args:
        cache_ttl: Cache time-to-live in seconds (default 5 min).
        timeout: HTTP request timeout in seconds.

    Example:
        >>> fetcher = CrossAssetFetcher()
        >>> data = fetcher.get()
        >>> print(data.vix, data.dxy)
    """

    def __init__(
        self, cache_ttl: int = 300, timeout: int = 10
    ) -> None:
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cache: Optional[CrossAssetData] = None

    def get(self) -> CrossAssetData:
        """Return cached data or fetch fresh.

        Returns:
            CrossAssetData with current macro values.
        """
        now = time.time()
        if (
            self._cache is not None
            and (now - self._cache.fetched_at) < self.cache_ttl
        ):
            return self._cache

        data = self._fetch_all()
        self._cache = data
        return data

    def invalidate(self) -> None:
        """Clear the cache, forcing a fresh fetch on next get()."""
        self._cache = None

    def _fetch_all(self) -> CrossAssetData:
        """Fetch all 4 symbols, falling back to defaults on failure.

        Returns:
            CrossAssetData with fetched or default values.
        """
        results: dict[str, float] = {}
        prev_closes: dict[str, float] = {}
        any_success = False

        for key, symbol in SYMBOLS.items():
            price, prev_close = self._fetch_symbol(symbol)
            if price is not None:
                results[key] = price
                any_success = True
            else:
                results[key] = DEFAULTS[key]
            if prev_close is not None:
                prev_closes[key] = prev_close

        sp500_prev = prev_closes.get("sp500", results["sp500"])
        dxy_prev = prev_closes.get("dxy", results["dxy"])

        data = CrossAssetData(
            vix=results["vix"],
            dxy=results["dxy"],
            dxy_prev_close=dxy_prev,
            us10y=results["us10y"],
            sp500=results["sp500"],
            sp500_prev_close=sp500_prev,
            fetched_at=time.time(),
            is_default=not any_success,
        )

        if any_success:
            logger.info(
                "Cross-asset data: VIX=%.1f DXY=%.2f 10Y=%.2f SP500=%.0f",
                data.vix,
                data.dxy,
                data.us10y,
                data.sp500,
            )
        else:
            logger.warning("Cross-asset fetch failed, using defaults")

        return data

    def _fetch_symbol(
        self, symbol: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Fetch a single symbol from Yahoo Finance V8 API.

        Args:
            symbol: Yahoo Finance symbol (e.g. "^VIX").

        Returns:
            Tuple of (current_price, previous_close) or (None, None).
        """
        url = _BASE_URL.format(symbol=symbol)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(
                req, timeout=self.timeout
            ) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            meta = raw["chart"]["result"][0]["meta"]
            price = float(meta["regularMarketPrice"])
            prev_close = float(
                meta.get("chartPreviousClose", meta.get("previousClose", price))
            )
            return price, prev_close

        except Exception as e:
            logger.debug("Failed to fetch %s: %s", symbol, e)
            return None, None
