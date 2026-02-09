"""Order types and execution for paper trading.

This module defines the order lifecycle:

1. ORDER CREATED: Client wants to buy/sell
2. ORDER SUBMITTED: Sent to broker (simulated latency)
3. ORDER FILLED: Executed at price (with slippage)
4. ORDER REJECTED: Not enough margin, invalid params, etc.

Order types supported:
- MARKET: Execute immediately at current price (+ slippage)
- LIMIT: Execute only if price reaches target
- STOP: Trigger market order when price hits stop level
- STOP_LIMIT: Trigger limit order when price hits stop level

Real brokers also have:
- IOC (Immediate or Cancel)
- FOK (Fill or Kill)
- GTC (Good Till Cancelled)
We keep it simple for educational purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional
import uuid


class OrderType(Enum):
    """Type of order."""

    MARKET = auto()  # Execute at current price
    LIMIT = auto()  # Execute at target price or better
    STOP = auto()  # Trigger market when price hits stop
    STOP_LIMIT = auto()  # Trigger limit when price hits stop


class OrderSide(Enum):
    """Buy or sell."""

    BUY = auto()
    SELL = auto()


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = auto()  # Created, not yet submitted
    SUBMITTED = auto()  # Sent to broker, awaiting fill
    PARTIALLY_FILLED = auto()  # Some quantity filled
    FILLED = auto()  # Fully executed
    CANCELLED = auto()  # Cancelled by user
    REJECTED = auto()  # Rejected by broker (margin, invalid, etc.)
    EXPIRED = auto()  # Limit/stop expired without fill


@dataclass
class Fill:
    """Record of an order execution.

    A single order can have multiple fills (partial fills).
    """

    fill_id: str
    order_id: str
    timestamp: datetime
    price: float  # Actual execution price (after slippage)
    quantity: float
    commission: float
    slippage: float  # Price impact in pips

    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return self.price * self.quantity + self.commission


@dataclass
class Order:
    """An order to buy or sell a currency pair.

    Attributes:
        order_id: Unique identifier
        pair: Currency pair (e.g., "USD/JPY")
        side: BUY or SELL
        order_type: MARKET, LIMIT, STOP, STOP_LIMIT
        quantity: Amount to trade
        limit_price: Target price for LIMIT orders
        stop_price: Trigger price for STOP orders
        status: Current order status
        created_at: When order was created
        submitted_at: When order was sent to broker
        filled_at: When order was fully filled
        fills: List of fills (partial or full)
    """

    pair: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fills: list[Fill] = field(default_factory=list)
    reject_reason: Optional[str] = None
    trade_id: Optional[str] = None  # OANDA trade ID for stop-loss mgmt

    @property
    def filled_quantity(self) -> float:
        """Total quantity filled so far."""
        return sum(f.quantity for f in self.fills)

    @property
    def remaining_quantity(self) -> float:
        """Quantity remaining to be filled."""
        return self.quantity - self.filled_quantity

    @property
    def avg_fill_price(self) -> Optional[float]:
        """Volume-weighted average fill price."""
        if not self.fills:
            return None
        total_value = sum(f.price * f.quantity for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        return total_value / total_qty if total_qty > 0 else None

    @property
    def total_commission(self) -> float:
        """Total commission paid across all fills."""
        return sum(f.commission for f in self.fills)

    @property
    def total_slippage(self) -> float:
        """Total slippage across all fills."""
        return sum(f.slippage for f in self.fills)

    def is_active(self) -> bool:
        """Check if order is still active (could be filled)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    def can_cancel(self) -> bool:
        """Check if order can be cancelled."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "order_id": self.order_id,
            "pair": self.pair,
            "side": self.side.name,
            "type": self.order_type.name,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "total_commission": self.total_commission,
            "trade_id": self.trade_id,
        }


@dataclass
class Position:
    """An open position in a currency pair.

    Tracks entry, current P&L, and accumulated swap.
    """

    pair: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    accumulated_swap: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized P&L (excluding swap)."""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def total_pnl(self) -> float:
        """Total P&L including accumulated swap."""
        return self.unrealized_pnl + self.accumulated_swap

    @property
    def return_pct(self) -> float:
        """Return as percentage of entry value."""
        entry_value = self.entry_price * self.quantity
        return self.total_pnl / entry_value if entry_value > 0 else 0.0

    def update_price(self, price: float) -> None:
        """Update current market price."""
        self.current_price = price

    def add_swap(self, swap: float) -> None:
        """Add overnight swap credit/debit."""
        self.accumulated_swap += swap

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "pair": self.pair,
            "side": self.side.name,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "accumulated_swap": self.accumulated_swap,
            "total_pnl": self.total_pnl,
        }
