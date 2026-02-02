"""Tests for the OANDA broker adapter.

Uses mocked API calls since we don't want actual OANDA connections
during testing.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.broker.oanda import (
    OandaBroker,
    OandaConfig,
    _from_oanda_instrument,
    _to_oanda_instrument,
)
from src.broker.orders import OrderSide, OrderStatus


class TestOandaConfig:
    """Tests for OandaConfig validation."""

    def test_practice_environment_allowed(self):
        config = OandaConfig(
            access_token="test-token",
            account_id="101-001-123-001",
            environment="practice",
        )
        assert config.environment == "practice"

    def test_live_environment_rejected(self):
        with pytest.raises(ValueError, match="practice"):
            OandaConfig(
                access_token="test-token",
                account_id="101-001-123-001",
                environment="live",
            )

    def test_default_instruments(self):
        config = OandaConfig(
            access_token="test-token",
            account_id="101-001-123-001",
        )
        assert "USD_JPY" in config.instruments
        assert "AUD_JPY" in config.instruments


class TestInstrumentConversion:
    """Tests for instrument name conversion helpers."""

    def test_to_oanda(self):
        assert _to_oanda_instrument("USD/JPY") == "USD_JPY"
        assert _to_oanda_instrument("AUD/JPY") == "AUD_JPY"
        assert _to_oanda_instrument("EUR/USD") == "EUR_USD"

    def test_from_oanda(self):
        assert _from_oanda_instrument("USD_JPY") == "USD/JPY"
        assert _from_oanda_instrument("AUD_JPY") == "AUD/JPY"


@pytest.fixture
def mock_broker():
    """Create an OandaBroker with a mocked API client."""
    config = OandaConfig(
        access_token="test-token",
        account_id="101-001-123-001",
    )
    broker = OandaBroker(config)
    broker.client = MagicMock()
    return broker


class TestFetchCandles:
    """Tests for candle fetching."""

    def test_returns_dataframe(self, mock_broker):
        mock_broker.client.request.return_value = {
            "candles": [
                {
                    "complete": True,
                    "time": "2026-02-01T12:00:00Z",
                    "mid": {
                        "o": "155.00",
                        "h": "155.50",
                        "l": "154.80",
                        "c": "155.30",
                    },
                    "volume": 1000,
                },
                {
                    "complete": True,
                    "time": "2026-02-01T13:00:00Z",
                    "mid": {
                        "o": "155.30",
                        "h": "155.60",
                        "l": "155.10",
                        "c": "155.40",
                    },
                    "volume": 800,
                },
            ]
        }

        df = mock_broker.fetch_candles("USD/JPY", count=2)
        assert df is not None
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert df["close"].iloc[0] == 155.30

    def test_skips_incomplete_candles(self, mock_broker):
        mock_broker.client.request.return_value = {
            "candles": [
                {
                    "complete": True,
                    "time": "2026-02-01T12:00:00Z",
                    "mid": {"o": "155", "h": "156", "l": "154", "c": "155.5"},
                    "volume": 100,
                },
                {
                    "complete": False,
                    "time": "2026-02-01T13:00:00Z",
                    "mid": {"o": "155.5", "h": "156", "l": "155", "c": "155.8"},
                    "volume": 50,
                },
            ]
        }

        df = mock_broker.fetch_candles("USD/JPY")
        assert df is not None
        assert len(df) == 1

    def test_returns_none_on_empty(self, mock_broker):
        mock_broker.client.request.return_value = {"candles": []}
        df = mock_broker.fetch_candles("USD/JPY")
        assert df is None


class TestGetCurrentPrice:
    """Tests for current price fetching."""

    def test_returns_bid_ask_mid(self, mock_broker):
        mock_broker.client.request.return_value = {
            "prices": [
                {
                    "bids": [{"price": "155.400"}],
                    "asks": [{"price": "155.420"}],
                }
            ]
        }

        price = mock_broker.get_current_price("USD/JPY")
        assert price is not None
        assert price["bid"] == 155.400
        assert price["ask"] == 155.420
        assert price["mid"] == pytest.approx(155.410)

    def test_returns_none_on_empty(self, mock_broker):
        mock_broker.client.request.return_value = {"prices": []}
        assert mock_broker.get_current_price("USD/JPY") is None


class TestSubmitMarketOrder:
    """Tests for market order submission."""

    def test_buy_order_filled(self, mock_broker):
        mock_broker.client.request.return_value = {
            "orderFillTransaction": {
                "id": "12345",
                "orderID": "12344",
                "price": "155.500",
                "units": "10000",
                "commission": "0.00",
            }
        }

        order = mock_broker.submit_market_order(
            "USD/JPY", OrderSide.BUY, 10000
        )
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == 155.500
        assert order.side == OrderSide.BUY

    def test_sell_order_filled(self, mock_broker):
        mock_broker.client.request.return_value = {
            "orderFillTransaction": {
                "id": "12346",
                "price": "155.300",
                "units": "-10000",
                "commission": "0.00",
            }
        }

        order = mock_broker.submit_market_order(
            "USD/JPY", OrderSide.SELL, 10000
        )
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.SELL

    def test_order_rejected(self, mock_broker):
        mock_broker.client.request.return_value = {
            "orderRejectTransaction": {
                "rejectReason": "INSUFFICIENT_MARGIN"
            }
        }

        order = mock_broker.submit_market_order(
            "USD/JPY", OrderSide.BUY, 10000
        )
        assert order is not None
        assert order.status == OrderStatus.REJECTED
        assert "INSUFFICIENT_MARGIN" in order.reject_reason


class TestClosePosition:
    """Tests for position closing."""

    def test_close_long_position(self, mock_broker):
        mock_broker.client.request.return_value = {
            "longOrderFillTransaction": {
                "id": "12347",
                "price": "156.000",
                "units": "-10000",
                "commission": "0.00",
            }
        }

        order = mock_broker.close_position("USD/JPY")
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.SELL

    def test_close_no_position(self, mock_broker):
        mock_broker.client.request.return_value = {}
        order = mock_broker.close_position("USD/JPY")
        assert order is None


class TestGetAccountState:
    """Tests for account state queries."""

    def test_returns_account_info(self, mock_broker):
        mock_broker.client.request.return_value = {
            "account": {
                "balance": "100000.00",
                "NAV": "100500.00",
                "unrealizedPL": "500.00",
                "marginUsed": "3000.00",
                "marginAvailable": "97500.00",
                "openPositionCount": "2",
            }
        }

        state = mock_broker.get_account_state()
        assert state is not None
        assert state["balance"] == 100000.00
        assert state["equity"] == 100500.00
        assert state["unrealized_pnl"] == 500.00
        assert state["open_positions"] == 2


class TestGetAllPositions:
    """Tests for position queries."""

    def test_returns_long_position(self, mock_broker):
        mock_broker.client.request.return_value = {
            "positions": [
                {
                    "instrument": "USD_JPY",
                    "long": {
                        "units": "10000",
                        "averagePrice": "155.500",
                        "unrealizedPL": "300.00",
                        "financing": "-5.00",
                    },
                    "short": {
                        "units": "0",
                        "averagePrice": "0",
                        "unrealizedPL": "0",
                    },
                }
            ]
        }

        positions = mock_broker.get_all_positions()
        assert positions is not None
        assert len(positions) == 1
        assert positions[0]["pair"] == "USD/JPY"
        assert positions[0]["side"] == "BUY"
        assert positions[0]["units"] == 10000

    def test_returns_empty_for_no_positions(self, mock_broker):
        mock_broker.client.request.return_value = {"positions": []}
        positions = mock_broker.get_all_positions()
        assert positions is not None
        assert len(positions) == 0
