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

        order = mock_broker.submit_market_order("USD/JPY", OrderSide.BUY, 10000)
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

        order = mock_broker.submit_market_order("USD/JPY", OrderSide.SELL, 10000)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.SELL

    def test_order_rejected(self, mock_broker):
        mock_broker.client.request.return_value = {
            "orderRejectTransaction": {"rejectReason": "INSUFFICIENT_MARGIN"}
        }

        order = mock_broker.submit_market_order("USD/JPY", OrderSide.BUY, 10000)
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


class TestSubmitLimitOrder:
    """Tests for limit order submission."""

    def test_immediate_fill(self, mock_broker):
        """Limit order that fills immediately (price at or above ask)."""
        mock_broker.client.request.return_value = {
            "orderFillTransaction": {
                "id": "20001",
                "orderID": "20000",
                "price": "155.500",
                "units": "10000",
                "commission": "0.00",
                "tradeOpened": {"tradeID": "T500"},
            }
        }

        order = mock_broker.submit_limit_order(
            "USD/JPY", OrderSide.BUY, 10000, price=155.500, stop_loss_price=154.0
        )
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == 155.500
        assert order.trade_id == "T500"
        assert order.side == OrderSide.BUY

    def test_pending_order(self, mock_broker):
        """Limit order that stays pending (bid below ask)."""
        mock_broker.client.request.return_value = {
            "orderCreateTransaction": {
                "id": "20010",
                "type": "LIMIT_ORDER",
                "instrument": "USD_JPY",
                "units": "10000",
                "price": "155.400",
                "timeInForce": "GTC",
            }
        }

        order = mock_broker.submit_limit_order(
            "USD/JPY", OrderSide.BUY, 10000, price=155.400
        )
        assert order is not None
        assert order.status == OrderStatus.SUBMITTED
        assert order.trade_id == "20010"  # OANDA order ID stored here

    def test_rejection(self, mock_broker):
        """Limit order rejected by broker."""
        mock_broker.client.request.return_value = {
            "orderRejectTransaction": {"rejectReason": "INSUFFICIENT_MARGIN"}
        }

        order = mock_broker.submit_limit_order(
            "USD/JPY", OrderSide.BUY, 10000, price=155.500
        )
        assert order is not None
        assert order.status == OrderStatus.REJECTED
        assert "INSUFFICIENT_MARGIN" in order.reject_reason

    def test_sell_side_negates_units(self, mock_broker):
        """SELL limit should send negative units to OANDA."""
        mock_broker.client.request.return_value = {
            "orderCreateTransaction": {
                "id": "20020",
                "type": "LIMIT_ORDER",
                "instrument": "USD_JPY",
                "units": "-10000",
                "price": "156.000",
                "timeInForce": "GTC",
            }
        }

        order = mock_broker.submit_limit_order(
            "USD/JPY", OrderSide.SELL, 10000, price=156.000
        )
        assert order is not None
        assert order.status == OrderStatus.SUBMITTED
        assert order.side == OrderSide.SELL


class TestGetPendingOrders:
    """Tests for pending order queries."""

    def test_returns_pending_orders(self, mock_broker):
        """Should parse pending order list from OANDA."""
        mock_broker.client.request.return_value = {
            "orders": [
                {
                    "id": "30001",
                    "type": "LIMIT",
                    "instrument": "USD_JPY",
                    "units": "10000",
                    "price": "155.400",
                    "timeInForce": "GTC",
                    "stopLossOnFill": {"price": "154.000"},
                    "createTime": "2026-02-01T12:00:00Z",
                },
                {
                    "id": "30002",
                    "type": "LIMIT",
                    "instrument": "AUD_JPY",
                    "units": "5000",
                    "price": "98.500",
                    "timeInForce": "GTC",
                    "createTime": "2026-02-01T13:00:00Z",
                },
            ]
        }

        orders = mock_broker.get_pending_orders()
        assert orders is not None
        assert len(orders) == 2
        assert orders[0]["order_id"] == "30001"
        assert orders[0]["pair"] == "USD/JPY"
        assert orders[0]["units"] == 10000
        assert orders[0]["price"] == 155.400
        assert orders[0]["stop_loss_price"] == 154.0
        assert orders[1]["order_id"] == "30002"
        assert orders[1]["pair"] == "AUD/JPY"
        assert orders[1]["stop_loss_price"] is None  # No SL on this one

    def test_returns_empty_list(self, mock_broker):
        """Should return empty list when no pending orders."""
        mock_broker.client.request.return_value = {"orders": []}

        orders = mock_broker.get_pending_orders()
        assert orders is not None
        assert len(orders) == 0


class TestCancelOrder:
    """Tests for order cancellation."""

    def test_cancel_success(self, mock_broker):
        """Should return True on successful cancel."""
        mock_broker.client.request.return_value = {
            "orderCancelTransaction": {
                "orderID": "30001",
                "reason": "CLIENT_REQUEST",
            }
        }

        result = mock_broker.cancel_order("30001")
        assert result is True

    def test_cancel_rejection(self, mock_broker):
        """Should return False when cancel is rejected."""
        mock_broker.client.request.return_value = {
            "orderCancelRejectTransaction": {
                "orderID": "30001",
                "rejectReason": "ORDER_DOESNT_EXIST",
            }
        }

        result = mock_broker.cancel_order("30001")
        assert result is False

    def test_cancel_unexpected_response(self, mock_broker):
        """Should return False on unexpected response."""
        mock_broker.client.request.return_value = {}

        result = mock_broker.cancel_order("30001")
        assert result is False


class TestGetOpenTrades:
    """Tests for open trade queries and openTime parsing."""

    def test_open_time_extracted(self, mock_broker):
        """Should parse OANDA openTime into a datetime."""
        mock_broker.client.request.return_value = {
            "trades": [
                {
                    "id": "12345",
                    "instrument": "USD_JPY",
                    "currentUnits": "10000",
                    "price": "155.000",
                    "unrealizedPL": "50.00",
                    "financing": "-1.25",
                    "openTime": "2026-02-03T14:23:17.000000000Z",
                    "stopLossOrder": {
                        "price": "153.500",
                    },
                }
            ]
        }

        trades = mock_broker.get_open_trades()
        assert trades is not None
        assert len(trades) == 1

        trade = trades[0]
        assert trade["trade_id"] == "12345"
        assert trade["pair"] == "USD/JPY"
        assert trade["units"] == 10000
        assert trade["price"] == 155.0
        assert trade["stop_loss_price"] == 153.5
        assert trade["open_time"] is not None

        from datetime import datetime, timezone

        expected = datetime(2026, 2, 3, 14, 23, 17, tzinfo=timezone.utc)
        assert trade["open_time"].year == expected.year
        assert trade["open_time"].month == expected.month
        assert trade["open_time"].day == expected.day
        assert trade["open_time"].hour == expected.hour
        assert trade["open_time"].minute == expected.minute
        assert trade["open_time"].second == expected.second

    def test_open_time_missing_graceful(self, mock_broker):
        """Should return open_time=None when openTime is absent."""
        mock_broker.client.request.return_value = {
            "trades": [
                {
                    "id": "12346",
                    "instrument": "AUD_JPY",
                    "currentUnits": "5000",
                    "price": "98.000",
                    "unrealizedPL": "10.00",
                    "financing": "0.00",
                    # No openTime field
                }
            ]
        }

        trades = mock_broker.get_open_trades()
        assert trades is not None
        assert len(trades) == 1
        assert trades[0]["open_time"] is None
        assert trades[0]["stop_loss_price"] is None

    def test_open_time_malformed(self, mock_broker):
        """Should return open_time=None for unparseable openTime string."""
        mock_broker.client.request.return_value = {
            "trades": [
                {
                    "id": "12347",
                    "instrument": "GBP_JPY",
                    "currentUnits": "1000",
                    "price": "212.000",
                    "unrealizedPL": "0.00",
                    "financing": "0.00",
                    "openTime": "not-a-timestamp",
                }
            ]
        }

        trades = mock_broker.get_open_trades()
        assert trades is not None
        assert len(trades) == 1
        assert trades[0]["open_time"] is None

    def test_multiple_trades_returned(self, mock_broker):
        """Should return all trades with their respective openTimes."""
        mock_broker.client.request.return_value = {
            "trades": [
                {
                    "id": "100",
                    "instrument": "AUD_JPY",
                    "currentUnits": "5000",
                    "price": "98.000",
                    "unrealizedPL": "10.00",
                    "financing": "-0.50",
                    "openTime": "2026-02-01T08:00:00.000000000Z",
                },
                {
                    "id": "101",
                    "instrument": "AUD_JPY",
                    "currentUnits": "3000",
                    "price": "98.500",
                    "unrealizedPL": "5.00",
                    "financing": "-0.30",
                    "openTime": "2026-02-03T10:15:00.000000000Z",
                },
            ]
        }

        trades = mock_broker.get_open_trades()
        assert trades is not None
        assert len(trades) == 2
        assert trades[0]["open_time"] is not None
        assert trades[1]["open_time"] is not None
        # First trade should be earlier
        assert trades[0]["open_time"] < trades[1]["open_time"]

    def test_empty_trades(self, mock_broker):
        """Should return empty list when no trades."""
        mock_broker.client.request.return_value = {"trades": []}

        trades = mock_broker.get_open_trades()
        assert trades is not None
        assert len(trades) == 0
