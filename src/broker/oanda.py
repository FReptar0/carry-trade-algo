"""OANDA v20 API broker adapter.

Wraps the oandapyV20 library to provide a clean interface for:
- Fetching historical candles
- Getting current prices
- Submitting market orders
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
    instruments: list[str] = field(
        default_factory=lambda: ["USD_JPY", "AUD_JPY"]
    )

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
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "OANDA API error (attempt %d/%d): %s. "
                    "Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    wait,
                )
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Unexpected error (attempt %d/%d): %s. "
                    "Retrying in %.1fs",
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

        endpoint = instruments.InstrumentsCandles(
            instrument=instrument, params=params
        )
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
        logger.info(
            "Fetched %d candles for %s (%s)", len(df), pair, granularity
        )
        return df

    @_retry
    def get_current_price(
        self, pair: str
    ) -> Optional[dict[str, float]]:
        """Get current bid/ask/mid price for a pair.

        Args:
            pair: Currency pair (e.g., "USD/JPY").

        Returns:
            Dict with keys 'bid', 'ask', 'mid', or None on failure.
        """
        instrument = _to_oanda_instrument(pair)
        params = {"instruments": instrument}

        endpoint = pricing.PricingInfo(
            accountID=self.account_id, params=params
        )
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
    ) -> Optional[Order]:
        """Submit a market order.

        Args:
            pair: Currency pair (e.g., "USD/JPY").
            side: BUY or SELL.
            units: Number of units (positive). Converted to negative
                for sell orders per OANDA convention.

        Returns:
            Filled Order object, or None on failure.
        """
        instrument = _to_oanda_instrument(pair)
        signed_units = units if side == OrderSide.BUY else -units

        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(signed_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }

        endpoint = orders_api.OrderCreate(
            accountID=self.account_id, data=data
        )
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

        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(units),
            status=OrderStatus.FILLED,
            filled_at=datetime.now(),
            fills=[
                Fill(
                    fill_id=fill_data["id"],
                    order_id=fill_data.get("orderID", ""),
                    timestamp=datetime.now(),
                    price=fill_price,
                    quantity=float(fill_units),
                    commission=float(
                        fill_data.get("commission", "0")
                    ),
                    slippage=0.0,
                )
            ],
        )

        logger.info(
            "Order filled: %s %s %d units @ %.5f",
            side.name,
            pair,
            fill_units,
            fill_price,
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
        side = (
            OrderSide.SELL
            if long_txn
            else OrderSide.BUY
        )

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
                    commission=float(
                        fill_txn.get("commission", "0")
                    ),
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
        endpoint = accounts.AccountSummary(
            accountID=self.account_id
        )
        response = self.client.request(endpoint)
        acct = response.get("account", {})

        return {
            "balance": float(acct.get("balance", 0)),
            "equity": float(acct.get("NAV", 0)),
            "unrealized_pnl": float(
                acct.get("unrealizedPL", 0)
            ),
            "margin_used": float(acct.get("marginUsed", 0)),
            "margin_available": float(
                acct.get("marginAvailable", 0)
            ),
            "open_positions": int(
                acct.get("openPositionCount", 0)
            ),
        }

    @_retry
    def get_all_positions(self) -> Optional[list[dict]]:
        """Get all open positions.

        Returns:
            List of position dicts with keys: pair, side, units,
            avg_price, unrealized_pnl, financing.
            Returns None on failure.
        """
        endpoint = positions_api.OpenPositions(
            accountID=self.account_id
        )
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
                        "avg_price": float(
                            pos["long"]["averagePrice"]
                        ),
                        "unrealized_pnl": float(
                            pos["long"]["unrealizedPL"]
                        ),
                        "financing": float(
                            pos["long"].get("financing", 0)
                        ),
                    }
                )
            if short_units < 0:
                result.append(
                    {
                        "pair": pair,
                        "side": "SELL",
                        "units": abs(short_units),
                        "avg_price": float(
                            pos["short"]["averagePrice"]
                        ),
                        "unrealized_pnl": float(
                            pos["short"]["unrealizedPL"]
                        ),
                        "financing": float(
                            pos["short"].get("financing", 0)
                        ),
                    }
                )

        return result

    @_retry
    def get_swap_rates(
        self, pair: str
    ) -> Optional[tuple[float, float]]:
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

            long_rate = float(
                financing.get("longRate", "0")
            )
            short_rate = float(
                financing.get("shortRate", "0")
            )

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
                "Could not fetch financing rates for %s: %s. "
                "Returning defaults.",
                pair,
                e,
            )
            return (0.0, 0.0)
