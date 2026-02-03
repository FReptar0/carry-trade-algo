"""Tests for CrossAssetFetcher."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.data.cross_asset import (
    CrossAssetData,
    CrossAssetFetcher,
    DEFAULTS,
    SYMBOLS,
)


def _make_yahoo_response(price: float, prev_close: float = 0.0) -> bytes:
    """Build a fake Yahoo V8 JSON response."""
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "regularMarketPrice": price,
                            "chartPreviousClose": prev_close or price,
                        }
                    }
                ]
            }
        }
    ).encode("utf-8")


class TestCrossAssetFetcher:
    def test_cache_hit(self):
        """Second call within TTL returns the same object."""
        fetcher = CrossAssetFetcher(cache_ttl=300)
        fake = CrossAssetData(
            vix=18.0,
            dxy=102.0,
            us10y=4.2,
            sp500=5100.0,
            sp500_prev_close=5050.0,
            fetched_at=time.time(),
            is_default=False,
        )
        fetcher._cache = fake
        result = fetcher.get()
        assert result is fake

    def test_cache_expiry(self):
        """After TTL, re-fetches instead of returning stale cache."""
        fetcher = CrossAssetFetcher(cache_ttl=1)
        old = CrossAssetData(
            vix=18.0,
            dxy=102.0,
            us10y=4.2,
            sp500=5100.0,
            sp500_prev_close=5050.0,
            fetched_at=time.time() - 10,
            is_default=False,
        )
        fetcher._cache = old

        with patch.object(
            fetcher, "_fetch_all", return_value=CrossAssetData(
                vix=22.0,
                dxy=101.0,
                us10y=4.3,
                sp500=5200.0,
                sp500_prev_close=5100.0,
                fetched_at=time.time(),
                is_default=False,
            )
        ) as mock_fetch:
            result = fetcher.get()
            mock_fetch.assert_called_once()
            assert result.vix == 22.0

    def test_invalidate_clears_cache(self):
        """invalidate() forces a fresh fetch on next get()."""
        fetcher = CrossAssetFetcher(cache_ttl=300)
        fetcher._cache = CrossAssetData(
            vix=18.0,
            dxy=102.0,
            us10y=4.2,
            sp500=5100.0,
            sp500_prev_close=5050.0,
            fetched_at=time.time(),
            is_default=False,
        )
        fetcher.invalidate()
        assert fetcher._cache is None

    def test_network_failure_returns_defaults(self):
        """When all HTTP calls fail, returns default values."""
        fetcher = CrossAssetFetcher(cache_ttl=300)
        with patch.object(
            fetcher, "_fetch_symbol", return_value=(None, None)
        ):
            data = fetcher.get()
            assert data.vix == DEFAULTS["vix"]
            assert data.dxy == DEFAULTS["dxy"]
            assert data.us10y == DEFAULTS["us10y"]
            assert data.sp500 == DEFAULTS["sp500"]
            assert data.is_default is True

    def test_json_parsing_extracts_price(self):
        """_fetch_symbol extracts regularMarketPrice from Yahoo JSON."""
        fetcher = CrossAssetFetcher()

        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_yahoo_response(25.5, 24.0)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            price, prev_close = fetcher._fetch_symbol("^VIX")
            assert price == 25.5
            assert prev_close == 24.0

    def test_all_four_symbols_fetched(self):
        """All 4 symbols are requested during a fetch."""
        fetcher = CrossAssetFetcher()
        calls = []

        def fake_fetch(symbol):
            calls.append(symbol)
            return 100.0, 99.0

        with patch.object(fetcher, "_fetch_symbol", side_effect=fake_fetch):
            data = fetcher.get()

        assert set(calls) == set(SYMBOLS.values())
        assert not data.is_default

    def test_partial_failure(self):
        """If 1 symbol fails, others still succeed."""
        fetcher = CrossAssetFetcher()

        def partial_fetch(symbol):
            if symbol == "^VIX":
                return None, None
            return 100.0, 99.0

        with patch.object(
            fetcher, "_fetch_symbol", side_effect=partial_fetch
        ):
            data = fetcher.get()

        assert data.vix == DEFAULTS["vix"]  # Failed — default
        assert data.dxy == 100.0  # Succeeded
        assert data.us10y == 100.0
        assert data.sp500 == 100.0
        assert data.is_default is False  # At least one succeeded

    def test_fetch_symbol_handles_exception(self):
        """_fetch_symbol returns (None, None) on HTTP error."""
        fetcher = CrossAssetFetcher(timeout=1)
        with patch(
            "urllib.request.urlopen",
            side_effect=Exception("connection refused"),
        ):
            price, prev_close = fetcher._fetch_symbol("^VIX")
            assert price is None
            assert prev_close is None
