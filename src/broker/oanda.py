"""OANDA v20 API broker adapter.

Wraps the oandapyV20 library to provide a clean interface for:
- Fetching historical candles
- Getting current prices
- Submitting market and limit orders
- Managing pending orders (query and cancel)
- Closing positions
- Querying account state and swap rates

Practice account only. Every API call retries up to 3 times
with exponential backoff. Failures are logged but never crash
the system.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders_api
import oandapyV20.endpoints.positions as positions_api
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades_api
from oandapyV20.contrib.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossDetails,
)

from src.broker.orders import Fill, Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds


@dataclass
class OandaConfig:
    """Configuration for the OANDA broker connection.

    Attributes:
        access_token: OANDA API access token.
        account_id: OANDA account ID (e.g., "101-001-12345678-001").
        environment: "practice" or "live". Only practice is supported.
        instruments: List of OANDA instrument names to trade.
    """

    access_token: str
    account_id: str
    environment: str = "practice"
    instruments: list[str] = field(default_factory=lambda: ["USD_JPY", "AUD_JPY"])

    def __post_init__(self) -> None:
        if self.environment != "practice":
            raise ValueError(
                "Only practice environment is supported. "
                "This system is for paper trading only."
            )


def _to_oanda_instrument(pair: str) -> str:
    """Convert 'USD/JPY' to 'USD_JPY'."""
    return pair.replace("/", "_")


def _from_oanda_instrument(instrument: str) -> str:
    """Convert 'USD_JPY' to 'USD/JPY'."""
    return instrument.replace("_", "/")


def _retry(func):
    """Decorator for retrying API calls with exponential backoff."""

    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except oandapyV20.exceptions.V20Error as e:
                last_exc = e
                # Don't retry auth errors
                if "401" in str(e) or "403" in str(e):
                    logger.critical("OANDA auth error: %s", e)
                    raise
                wait = BACKOFF_BASE**attempt
                logger.warning(
                    "OANDA API error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    wait,
                )
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                wait = BACKOFF_BASE**attempt
                logger.warning(
                    "Unexpected error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    wait,
                )
                time.sleep(wait)
        logger.error(
            "All %d retries exhausted. Last error: %s",
            MAX_RETRIES,
            last_exc,
        )
        return None

    return wrapper


class OandaBroker:
    """OANDA v20 API adapter for paper trading.

    Args:
        config: OandaConfig with credentials and settings.

    Example:
        >>> config = OandaConfig(
        ...     access_token="your-token",
        ...     account_id="101-001-12345678-001",
        ... )
        >>> broker = OandaBroker(config)
        >>> candles = broker.fetch_candles("USD/JPY", count=300)
        >>> price = broker.get_current_price("USD/JPY")
    """

    def __init__(self, config: OandaConfig) -> None:
        self.config = config
        self.client = oandapyV20.API(
            access_token=config.access_token,
            environment=config.environment,
        )
        self.account_id = config.account_id

    @_retry
    def fetch_candles(
        self,
        pair: str,
        count: int = 500,
        granularity: str = "H1",
    ) -> Optional[pd.DataFrame]:
        """Fetch historical candlestick data.

        Args:
            pair: Currency pair (e.g., "USD/JPY").
            count: Number of candles to fetch (max 5000).
            granularity: Candle period ("M1", "H1", "D", etc.).

        Returns:
            DataFrame with columns matching the project standard:
            timestamp, open, high, low, close, volume.
            Returns None on failure.
        """
        instrument = _to_oanda_instrument(pair)
        params = {
            "count": min(count, 5000),
            "granularity": granularity,
            "price": "M",  # Mid prices
        }

        endpoint = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = self.client.request(endpoint)
        candles = response.get("candles", [])

        if not candles:
            logger.warning("No candles returned for %s", pair)
            return None

        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append(
                {
                    "timestamp": pd.Timestamp(c["time"]),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c["volume"]),
                }
            )

        if not rows:
            return None

        df = pd.DataFrame(rows)
        logger.info("Fetched %d candles for %s (%s)", len(df), pair, granularity)
        return df

    @_retry
    def get_current_price(self, pair: str) -> Optional[dict[str, float]]:
        """Get current bid/ask/mid price for a pair.

        Args:
            pair: Currency pair (e.g., "USD/JPY").

        Returns:
            Dict with keys 'bid', 'ask', 'mid', or None on failure.
        """
        instrument = _to_oanda_instrument(pair)
        params = {"instruments": instrument}

        endpoint = pricing.PricingInfo(accountID=self.account_id, params=params)
        response = self.client.request(endpoint)
        prices = response.get("prices", [])

        if not prices:
            return None

        p = prices[0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
        }

    @_retry
    def submit_market_order(
        self,
        pair: str,
        side: OrderSide,
        units: int,
        stop_loss_price: Optional[float] = None,
    ) -> Optional[Order]:
        """Submit a market order, optionally with a broker-side stop-loss.

        Args:
            pair: Currency pair (e.g., "USD/JPY").
            side: BUY or SELL.
            units: Number of units (positive). Converted to negative
                for sell orders per OANDA convention.
            stop_loss_price: If provided, OANDA attaches a stop-loss
                order to the trade at fill time. This means the stop
                is enforced server-side even if the bot is offline.

        Returns:
            Filled Order object (with trade_id set), or None on failure.
        """
        instrument = _to_oanda_instrument(pair)
        signed_units = units if side == OrderSide.BUY else -units

        # Build stop-loss on-fill payload if requested
        sl_on_fill = None
        if stop_loss_price is not None:
            sl_on_fill = StopLossDetails(price=round(stop_loss_price, 3)).data
            logger.info(
                "Attaching stop-loss @ %.3f to %s order for %s",
                stop_loss_price,
                side.name,
                pair,
            )

        # Use the SDK's MarketOrderRequest for clean payload construction
        mo = MarketOrderRequest(
            instrument=instrument,
            units=signed_units,
            stopLossOnFill=sl_on_fill,
        )

        endpoint = orders_api.OrderCreate(accountID=self.account_id, data=mo.data)
        response = self.client.request(endpoint)

        fill_data = response.get("orderFillTransaction")
        if not fill_data:
            reject = response.get("orderRejectTransaction", {})
            reason = reject.get("rejectReason", "Unknown rejection")
            logger.error("Order rejected for %s: %s", pair, reason)
            order = Order(
                pair=pair,
                side=side,
                order_type=OrderType.MARKET,
                quantity=float(units),
                status=OrderStatus.REJECTED,
                reject_reason=reason,
            )
            return order

        fill_price = float(fill_data["price"])
        fill_units = abs(int(fill_data["units"]))

        # Extract the OANDA trade ID — needed for stop-loss updates
        trade_id = None
        trade_opened = fill_data.get("tradeOpened")
        if trade_opened:
            trade_id = str(trade_opened["tradeID"])

        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(units),
            status=OrderStatus.FILLED,
            filled_at=datetime.now(),
            trade_id=trade_id,
            fills=[
                Fill(
                    fill_id=fill_data["id"],
                    order_id=fill_data.get("orderID", ""),
                    timestamp=datetime.now(),
                    price=fill_price,
                    quantity=float(fill_units),
                    commission=float(fill_data.get("commission", "0")),
                    slippage=0.0,
                )
            ],
        )

        logger.info(
            "Order filled: %s %s %d units @ %.5f (trade_id=%s, stop=%.3f)",
            side.name,
            pair,
            fill_units,
            fill_price,
            trade_id or "N/A",
            stop_loss_price or 0.0,
        )
        return order

    @_retry
    def close_position(
        self, pair: str, side: str = "long", units: int = 0
    ) -> Optional[Order]:
        """Close a position (fully or partially) for a pair.

        Args:
            pair: Currency pair (e.g., "USD/JPY").
            side: Which side to close ("long" or "short"). Defaults to "long"
                since the carry trade strategy only holds longs.
            units: Number of units to close. 0 means close ALL units.

        Returns:
            Order representing the close, or None on failure.
        """
        instrument = _to_oanda_instrument(pair)
        # Only close the specified side to avoid rejection when the other
        # side doesn't exist. OANDA rejects the whole request if you try
        # to close a non-existent short alongside an existing long.
        unit_str = str(units) if units > 0 else "ALL"
        if side == "long":
            data = {"longUnits": unit_str}
        else:
            data = {"shortUnits": unit_str}

        endpoint = positions_api.PositionClose(
            accountID=self.account_id,
            instrument=instrument,
            data=data,
        )
        response = self.client.request(endpoint)

        # Check for long close
        long_txn = response.get("longOrderFillTransaction")
        short_txn = response.get("shortOrderFillTransaction")
        fill_txn = long_txn or short_txn

        if not fill_txn:
            logger.warning("No position to close for %s", pair)
            return None

        fill_price = float(fill_txn["price"])
        fill_units = abs(int(fill_txn["units"]))
        side = OrderSide.SELL if long_txn else OrderSide.BUY

        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(fill_units),
            status=OrderStatus.FILLED,
            filled_at=datetime.now(),
            fills=[
                Fill(
                    fill_id=fill_txn["id"],
                    order_id=fill_txn.get("orderID", ""),
                    timestamp=datetime.now(),
                    price=fill_price,
                    quantity=float(fill_units),
                    commission=float(fill_txn.get("commission", "0")),
                    slippage=0.0,
                )
            ],
        )

        logger.info(
            "Position closed: %s %d units @ %.5f",
            pair,
            fill_units,
            fill_price,
        )
        return order

    @_retry
    def get_account_state(self) -> Optional[dict]:
        """Get current account summary.

        Returns:
            Dict with keys: balance, equity (NAV), unrealized_pnl,
            margin_used, margin_available, open_positions.
            Returns None on failure.
        """
        endpoint = accounts.AccountSummary(accountID=self.account_id)
        response = self.client.request(endpoint)
        acct = response.get("account", {})

        return {
            "balance": float(acct.get("balance", 0)),
            "equity": float(acct.get("NAV", 0)),
            "unrealized_pnl": float(acct.get("unrealizedPL", 0)),
            "margin_used": float(acct.get("marginUsed", 0)),
            "margin_available": float(acct.get("marginAvailable", 0)),
            "open_positions": int(acct.get("openPositionCount", 0)),
        }

    @_retry
    def get_all_positions(self) -> Optional[list[dict]]:
        """Get all open positions.

        Returns:
            List of position dicts with keys: pair, side, units,
            avg_price, unrealized_pnl, financing.
            Returns None on failure.
        """
        endpoint = positions_api.OpenPositions(accountID=self.account_id)
        response = self.client.request(endpoint)
        raw_positions = response.get("positions", [])

        result = []
        for pos in raw_positions:
            instrument = pos["instrument"]
            pair = _from_oanda_instrument(instrument)

            long_units = int(pos["long"]["units"])
            short_units = int(pos["short"]["units"])

            if long_units > 0:
                result.append(
                    {
                        "pair": pair,
                        "side": "BUY",
                        "units": long_units,
                        "avg_price": float(pos["long"]["averagePrice"]),
                        "unrealized_pnl": float(pos["long"]["unrealizedPL"]),
                        "financing": float(pos["long"].get("financing", 0)),
                    }
                )
            if short_units < 0:
                result.append(
                    {
                        "pair": pair,
                        "side": "SELL",
                        "units": abs(short_units),
                        "avg_price": float(pos["short"]["averagePrice"]),
                        "unrealized_pnl": float(pos["short"]["unrealizedPL"]),
                        "financing": float(pos["short"].get("financing", 0)),
                    }
                )

        return result

    @_retry
    def get_swap_rates(self, pair: str) -> Optional[tuple[float, float]]:
        """Get current financing/swap rates for a pair.

        Queries the OANDA v20 /accounts/{id}/instruments endpoint
        to retrieve actual financing rates. Falls back to (0, 0)
        if the instrument data doesn't include financing info.

        Args:
            pair: Currency pair (e.g., "USD/JPY").

        Returns:
            Tuple of (long_rate, short_rate) as daily decimals,
            or None on failure.
        """
        instrument = _to_oanda_instrument(pair)

        try:
            endpoint = accounts.AccountInstruments(
                accountID=self.account_id,
                params={"instruments": instrument},
            )
            response = self.client.request(endpoint)
            instruments_data = response.get("instruments", [])

            if not instruments_data:
                logger.debug(
                    "No instrument data for %s, returning defaults",
                    pair,
                )
                return (0.0, 0.0)

            inst = instruments_data[0]
            financing = inst.get("financing", {})

            long_rate = float(financing.get("longRate", "0"))
            short_rate = float(financing.get("shortRate", "0"))

            # Convert from annualized to daily rate
            daily_long = long_rate / 365.0
            daily_short = short_rate / 365.0

            logger.debug(
                "Swap rates for %s: long=%.6f short=%.6f (daily)",
                pair,
                daily_long,
                daily_short,
            )
            return (daily_long, daily_short)

        except Exception as e:
            logger.warning(
                "Could not fetch financing rates for %s: %s. Returning defaults.",
                pair,
                e,
            )
            return (0.0, 0.0)

    @_retry
    def get_open_trades(self) -> Optional[list[dict]]:
        """Get all open trades with their stop-loss order info.

        Unlike get_all_positions() which aggregates by instrument,
        this returns individual trades — each with its own trade_id,
        stop-loss state, and units. Needed for per-trade stop management.

        Returns:
            List of trade dicts with keys: trade_id, pair, units,
            price (open price), unrealized_pnl, stop_loss_price
            (or None if no stop). Returns None on failure.
        """
        endpoint = trades_api.OpenTrades(accountID=self.account_id)
        response = self.client.request(endpoint)
        raw_trades = response.get("trades", [])

        result = []
        for t in raw_trades:
            sl_order = t.get("stopLossOrder")
            sl_price = None
            if sl_order:
                sl_price = float(sl_order["price"])

            result.append(
                {
                    "trade_id": str(t["id"]),
                    "pair": _from_oanda_instrument(t["instrument"]),
                    "units": int(t["currentUnits"]),
                    "price": float(t["price"]),
                    "unrealized_pnl": float(t.get("unrealizedPL", "0")),
                    "financing": float(t.get("financing", "0")),
                    "stop_loss_price": sl_price,
                }
            )

        return result

    @_retry
    def modify_trade_stop_loss(self, trade_id: str, stop_price: float) -> bool:
        """Create or replace the stop-loss order on an existing trade.

        Uses the OANDA TradeCRCDO (Create/Replace/Cancel Dependent
        Orders) endpoint. If a stop-loss already exists on the trade,
        it is atomically replaced.

        Args:
            trade_id: OANDA trade ID (from orderFillTransaction).
            stop_price: New stop-loss trigger price. Rounded to 3
                decimal places (appropriate for JPY pairs).

        Returns:
            True if the stop was successfully set/updated, False on
            failure.
        """
        data = {
            "stopLoss": {
                "timeInForce": "GTC",
                "price": f"{round(stop_price, 3):.3f}",
            }
        }

        endpoint = trades_api.TradeCRCDO(
            accountID=self.account_id,
            tradeID=trade_id,
            data=data,
        )
        response = self.client.request(endpoint)

        # Check for successful stop-loss order creation/replacement
        sl_txn = response.get("stopLossOrderTransaction")
        if sl_txn:
            logger.info(
                "Stop-loss set on trade %s @ %.3f",
                trade_id,
                stop_price,
            )
            return True

        # Check for rejection
        sl_reject = response.get("stopLossOrderRejectTransaction")
        if sl_reject:
            reason = sl_reject.get("rejectReason", "unknown")
            logger.error(
                "Stop-loss rejected on trade %s: %s",
                trade_id,
                reason,
            )
            return False

        logger.warning(
            "Unexpected response modifying stop for trade %s: %s",
            trade_id,
            list(response.keys()),
        )
        return False

    @_retry
    def submit_limit_order(
        self,
        pair: str,
        side: OrderSide,
        units: int,
        price: float,
        stop_loss_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Optional[Order]:
        """Submit a limit order for better fill prices.

        Places a GTC limit order at the specified price with an optional
        broker-side stop-loss attached on fill. Unlike market orders,
        limit orders only execute when the price reaches the limit level,
        potentially saving 1-2 pips of spread cost per entry.

        Args:
            pair: Currency pair (e.g., "USD/JPY").
            side: BUY or SELL.
            units: Number of units (positive).
            price: Limit price. For BUY, order fills at this price
                or lower. Rounded to 3 decimals (JPY pairs).
            stop_loss_price: If provided, attaches a stop-loss on fill.
            time_in_force: "GTC" (Good Till Cancelled) by default.

        Returns:
            Order object with status SUBMITTED (pending fill) and
            the OANDA order ID stored in trade_id, or None on failure.
            The order may also be immediately filled if the limit price
            is at or above the current ask (BUY) — in that case the
            status will be FILLED.
        """
        instrument = _to_oanda_instrument(pair)
        signed_units = units if side == OrderSide.BUY else -units
        rounded_price = round(price, 3)

        # Build stop-loss on-fill payload
        sl_on_fill = None
        if stop_loss_price is not None:
            sl_on_fill = StopLossDetails(price=round(stop_loss_price, 3)).data
            logger.info(
                "Limit order: attaching stop-loss @ %.3f for %s",
                stop_loss_price,
                pair,
            )

        lo = LimitOrderRequest(
            instrument=instrument,
            units=signed_units,
            price=f"{rounded_price:.3f}",
            stopLossOnFill=sl_on_fill,
            timeInForce=time_in_force,
        )

        endpoint = orders_api.OrderCreate(accountID=self.account_id, data=lo.data)
        response = self.client.request(endpoint)

        # Case 1: Immediate fill (limit at or better than market)
        fill_data = response.get("orderFillTransaction")
        if fill_data:
            fill_price = float(fill_data["price"])
            fill_units = abs(int(fill_data["units"]))
            trade_id = None
            trade_opened = fill_data.get("tradeOpened")
            if trade_opened:
                trade_id = str(trade_opened["tradeID"])

            order = Order(
                pair=pair,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=float(units),
                limit_price=rounded_price,
                status=OrderStatus.FILLED,
                filled_at=datetime.now(),
                trade_id=trade_id,
                fills=[
                    Fill(
                        fill_id=fill_data["id"],
                        order_id=fill_data.get("orderID", ""),
                        timestamp=datetime.now(),
                        price=fill_price,
                        quantity=float(fill_units),
                        commission=float(fill_data.get("commission", "0")),
                        slippage=0.0,
                    )
                ],
            )
            logger.info(
                "Limit order IMMEDIATELY filled: %s %s %d units @ %.5f "
                "(limit=%.3f, trade_id=%s)",
                side.name,
                pair,
                fill_units,
                fill_price,
                rounded_price,
                trade_id or "N/A",
            )
            return order

        # Case 2: Order created but not yet filled (pending)
        order_txn = response.get("orderCreateTransaction")
        if order_txn:
            oanda_order_id = str(order_txn.get("id", ""))
            order = Order(
                pair=pair,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=float(units),
                limit_price=rounded_price,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(),
                trade_id=oanda_order_id,  # Store OANDA order ID
            )
            logger.info(
                "Limit order PENDING: %s %s %d units @ %.3f (order_id=%s, stop=%.3f)",
                side.name,
                pair,
                units,
                rounded_price,
                oanda_order_id,
                stop_loss_price or 0.0,
            )
            return order

        # Case 3: Rejection
        reject = response.get("orderRejectTransaction", {})
        reason = reject.get("rejectReason", "Unknown rejection")
        logger.error("Limit order rejected for %s: %s", pair, reason)
        return Order(
            pair=pair,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=float(units),
            limit_price=rounded_price,
            status=OrderStatus.REJECTED,
            reject_reason=reason,
        )

    @_retry
    def get_pending_orders(self) -> Optional[list[dict]]:
        """Get all pending (unfilled) orders from OANDA.

        Queries the account's open orders. Useful for tracking limit
        orders that haven't filled yet.

        Returns:
            List of order dicts with keys: order_id, pair, units,
            price, order_type, time_in_force, stop_loss_price.
            Returns None on failure.
        """
        endpoint = orders_api.OrdersPending(accountID=self.account_id)
        response = self.client.request(endpoint)
        raw_orders = response.get("orders", [])

        result = []
        for o in raw_orders:
            sl_on_fill = o.get("stopLossOnFill")
            sl_price = None
            if sl_on_fill:
                sl_price = float(sl_on_fill.get("price", "0"))

            result.append(
                {
                    "order_id": str(o["id"]),
                    "pair": _from_oanda_instrument(o.get("instrument", "")),
                    "units": abs(int(o.get("units", "0"))),
                    "price": float(o.get("price", "0")),
                    "order_type": o.get("type", "UNKNOWN"),
                    "time_in_force": o.get("timeInForce", "GTC"),
                    "stop_loss_price": sl_price,
                    "create_time": o.get("createTime", ""),
                }
            )

        return result

    @_retry
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by OANDA order ID.

        Args:
            order_id: The OANDA order ID to cancel.

        Returns:
            True if cancelled successfully, False on failure.
        """
        endpoint = orders_api.OrderCancel(
            accountID=self.account_id,
            orderID=order_id,
        )
        response = self.client.request(endpoint)

        cancel_txn = response.get("orderCancelTransaction")
        if cancel_txn:
            logger.info(
                "Order %s cancelled (reason: %s)",
                order_id,
                cancel_txn.get("reason", "CLIENT_REQUEST"),
            )
            return True

        reject_txn = response.get("orderCancelRejectTransaction")
        if reject_txn:
            reason = reject_txn.get("rejectReason", "unknown")
            logger.error("Cancel rejected for order %s: %s", order_id, reason)
            return False

        logger.warning(
            "Unexpected cancel response for order %s: %s",
            order_id,
            list(response.keys()),
        )
        return False
