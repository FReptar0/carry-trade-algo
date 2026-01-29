"""Broker simulator for paper trading.

This simulates a real forex broker with:
- Order submission and execution
- Realistic slippage (price impact)
- Commission charges
- Margin requirements
- Swap (overnight interest) accrual
- Latency simulation

Why simulate a broker?
1. Test strategy in realistic conditions before live trading
2. Experience order rejection (insufficient margin, invalid orders)
3. Understand how slippage affects actual returns
4. Practice with swap accrual timing

This is a PAPER TRADING simulator - no real money, no real orders.
The goal is education and testing, not profit.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import uuid

from src.broker.orders import Order, OrderType, OrderSide, OrderStatus, Fill, Position


logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """Configuration for broker simulation."""
    initial_balance: float = 100000.0
    leverage: float = 50.0  # 50:1 leverage (typical for forex)
    min_slippage_pips: float = 0.5
    max_slippage_pips: float = 2.0
    commission_per_lot: float = 7.0  # $7 per 100k lot (round trip)
    lot_size: float = 100000.0  # Standard lot = 100k units
    min_order_size: float = 0.01  # 0.01 lot = 1000 units
    margin_call_level: float = 0.50  # 50% margin call
    stop_out_level: float = 0.30  # 30% stop out
    order_latency_ms: tuple[int, int] = (50, 200)  # Min/max latency
    swap_time_utc: int = 21  # 21:00 UTC (5pm NY)


@dataclass
class AccountState:
    """Current state of the trading account."""
    balance: float  # Cash balance
    equity: float  # Balance + unrealized P&L
    margin_used: float  # Margin tied up in positions
    free_margin: float  # Equity - margin_used
    margin_level: float  # Equity / margin_used (%)
    unrealized_pnl: float
    daily_pnl: float
    positions_count: int


class BrokerSimulator:
    """Simulates a forex broker for paper trading.

    Example:
        >>> broker = BrokerSimulator()
        >>> order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)
        >>> broker.execute_orders({"USD/JPY": 150.50})
        >>> print(broker.get_account_state())
    """

    def __init__(
        self,
        config: BrokerConfig | None = None,
        on_fill: Callable[[Fill], None] | None = None,
        on_position_update: Callable[[Position], None] | None = None,
    ) -> None:
        self.config = config or BrokerConfig()
        self.on_fill = on_fill
        self.on_position_update = on_position_update

        # Account state
        self.balance = self.config.initial_balance
        self.positions: dict[str, Position] = {}  # pair -> Position
        self.pending_orders: list[Order] = []
        self.order_history: list[Order] = []
        self.fill_history: list[Fill] = []

        # Tracking
        self.daily_pnl = 0.0
        self.last_swap_date: datetime | None = None
        self.prices: dict[str, float] = {}  # Current prices

        # Swap rates (approximate, per lot per night)
        self.swap_rates = {
            "USD/JPY": (1.37, -1.37),  # long, short
            "AUD/JPY": (1.10, -1.10),
            "EUR/USD": (-0.50, 0.50),
            "EUR/JPY": (0.85, -0.85),
            "GBP/JPY": (1.25, -1.25),
        }

    def submit_market_order(
        self,
        pair: str,
        side: OrderSide,
        lots: float,
    ) -> Order:
        """Submit a market order.

        Args:
            pair: Currency pair to trade.
            side: BUY or SELL.
            lots: Size in lots (1 lot = 100k units).

        Returns:
            Order object with status SUBMITTED or REJECTED.
        """
        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.MARKET,
            quantity=lots * self.config.lot_size,
        )

        # Check minimum size
        if lots < self.config.min_order_size:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Order size {lots} below minimum {self.config.min_order_size}"
            self.order_history.append(order)
            logger.warning(f"Order rejected: {order.reject_reason}")
            return order

        # Simulate latency
        latency_ms = random.randint(*self.config.order_latency_ms)
        time.sleep(latency_ms / 1000)

        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order)
        logger.info(f"Order submitted: {order.to_dict()}")

        return order

    def submit_limit_order(
        self,
        pair: str,
        side: OrderSide,
        lots: float,
        limit_price: float,
    ) -> Order:
        """Submit a limit order."""
        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=lots * self.config.lot_size,
            limit_price=limit_price,
        )

        if lots < self.config.min_order_size:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Order size below minimum"
            self.order_history.append(order)
            return order

        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order)
        logger.info(f"Limit order submitted: {order.to_dict()}")

        return order

    def submit_stop_order(
        self,
        pair: str,
        side: OrderSide,
        lots: float,
        stop_price: float,
    ) -> Order:
        """Submit a stop order."""
        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.STOP,
            quantity=lots * self.config.lot_size,
            stop_price=stop_price,
        )

        if lots < self.config.min_order_size:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Order size below minimum"
            self.order_history.append(order)
            return order

        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order)
        logger.info(f"Stop order submitted: {order.to_dict()}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: ID of order to cancel.

        Returns:
            True if cancelled, False if not found or can't cancel.
        """
        for order in self.pending_orders:
            if order.order_id == order_id and order.can_cancel():
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                self.order_history.append(order)
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False

    def execute_orders(self, prices: dict[str, float]) -> list[Fill]:
        """Execute pending orders at current prices.

        Call this on each price update to check for fills.

        Args:
            prices: Dict mapping pair to current price.

        Returns:
            List of fills that occurred.
        """
        self.prices = prices
        fills = []
        orders_to_remove = []

        for order in self.pending_orders:
            if order.pair not in prices:
                continue

            price = prices[order.pair]
            fill = self._try_fill_order(order, price)

            if fill:
                fills.append(fill)
                if order.status == OrderStatus.FILLED:
                    orders_to_remove.append(order)

        # Remove filled orders
        for order in orders_to_remove:
            self.pending_orders.remove(order)
            self.order_history.append(order)

        # Update positions with new prices
        self._update_positions(prices)

        return fills

    def _try_fill_order(self, order: Order, price: float) -> Optional[Fill]:
        """Attempt to fill an order at current price."""
        should_fill = False
        fill_price = price

        if order.order_type == OrderType.MARKET:
            should_fill = True
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and price <= order.limit_price:
                should_fill = True
                fill_price = order.limit_price  # Fill at limit or better
            elif order.side == OrderSide.SELL and price >= order.limit_price:
                should_fill = True
                fill_price = order.limit_price
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and price >= order.stop_price:
                should_fill = True
            elif order.side == OrderSide.SELL and price <= order.stop_price:
                should_fill = True

        if not should_fill:
            return None

        # Check margin
        required_margin = self._calculate_margin(order.quantity, fill_price)
        if required_margin > self._free_margin():
            order.status = OrderStatus.REJECTED
            order.reject_reason = "Insufficient margin"
            logger.warning(f"Order rejected: insufficient margin for {order.order_id}")
            return None

        # Apply slippage
        slippage_pips = random.uniform(
            self.config.min_slippage_pips,
            self.config.max_slippage_pips
        )

        # Slippage always works against you
        pip_value = self._pip_value(order.pair)
        if order.side == OrderSide.BUY:
            fill_price += slippage_pips * pip_value
        else:
            fill_price -= slippage_pips * pip_value

        # Calculate commission
        lots = order.quantity / self.config.lot_size
        commission = lots * self.config.commission_per_lot

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            slippage=slippage_pips,
        )

        order.fills.append(fill)
        order.filled_at = fill.timestamp
        order.status = OrderStatus.FILLED

        # Update balance for commission
        self.balance -= commission

        # Update or create position
        self._update_position_from_fill(order, fill)

        self.fill_history.append(fill)
        logger.info(f"Order filled: {fill}")

        if self.on_fill:
            self.on_fill(fill)

        return fill

    def _update_position_from_fill(self, order: Order, fill: Fill) -> None:
        """Update positions based on a fill."""
        pair = order.pair

        if pair in self.positions:
            pos = self.positions[pair]
            if pos.side == order.side:
                # Adding to position
                new_qty = pos.quantity + fill.quantity
                new_avg = (pos.entry_price * pos.quantity + fill.price * fill.quantity) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_avg
            else:
                # Closing or reversing
                if fill.quantity >= pos.quantity:
                    # Close position
                    realized_pnl = pos.unrealized_pnl + pos.accumulated_swap
                    self.balance += realized_pnl
                    self.daily_pnl += realized_pnl

                    remaining = fill.quantity - pos.quantity
                    del self.positions[pair]

                    if remaining > 0:
                        # Open reverse position
                        self.positions[pair] = Position(
                            pair=pair,
                            side=order.side,
                            quantity=remaining,
                            entry_price=fill.price,
                            entry_time=fill.timestamp,
                            current_price=fill.price,
                        )
                else:
                    # Partial close
                    close_ratio = fill.quantity / pos.quantity
                    realized_pnl = pos.unrealized_pnl * close_ratio + pos.accumulated_swap * close_ratio
                    self.balance += realized_pnl
                    self.daily_pnl += realized_pnl
                    pos.quantity -= fill.quantity
                    pos.accumulated_swap *= (1 - close_ratio)
        else:
            # New position
            self.positions[pair] = Position(
                pair=pair,
                side=order.side,
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_time=fill.timestamp,
                current_price=fill.price,
            )

    def _update_positions(self, prices: dict[str, float]) -> None:
        """Update all positions with current prices."""
        for pair, pos in self.positions.items():
            if pair in prices:
                pos.update_price(prices[pair])
                if self.on_position_update:
                    self.on_position_update(pos)

    def accrue_swap(self, current_time: datetime) -> dict[str, float]:
        """Accrue overnight swap for all positions.

        Call this at swap time (typically 5pm NY / 21:00 UTC).

        Args:
            current_time: Current datetime.

        Returns:
            Dict mapping pair to swap amount credited/debited.
        """
        if self.last_swap_date and self.last_swap_date.date() == current_time.date():
            return {}  # Already accrued today

        swaps = {}
        for pair, pos in self.positions.items():
            if pair not in self.swap_rates:
                continue

            long_rate, short_rate = self.swap_rates[pair]
            lots = pos.quantity / self.config.lot_size

            if pos.side == OrderSide.BUY:
                swap = long_rate * lots
            else:
                swap = short_rate * lots

            pos.add_swap(swap)
            swaps[pair] = swap
            logger.info(f"Swap accrued for {pair}: {swap:.2f}")

        self.last_swap_date = current_time
        return swaps

    def close_position(self, pair: str) -> Optional[Order]:
        """Close an entire position.

        Args:
            pair: Currency pair to close.

        Returns:
            The closing order, or None if no position.
        """
        if pair not in self.positions:
            return None

        pos = self.positions[pair]
        side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        lots = pos.quantity / self.config.lot_size

        return self.submit_market_order(pair, side, lots)

    def close_all_positions(self) -> list[Order]:
        """Close all open positions."""
        orders = []
        for pair in list(self.positions.keys()):
            order = self.close_position(pair)
            if order:
                orders.append(order)
        return orders

    def get_account_state(self) -> AccountState:
        """Get current account state."""
        unrealized = sum(p.unrealized_pnl + p.accumulated_swap for p in self.positions.values())
        equity = self.balance + unrealized
        margin_used = self._margin_used()
        free_margin = equity - margin_used
        margin_level = (equity / margin_used * 100) if margin_used > 0 else float('inf')

        return AccountState(
            balance=self.balance,
            equity=equity,
            margin_used=margin_used,
            free_margin=free_margin,
            margin_level=margin_level,
            unrealized_pnl=unrealized,
            daily_pnl=self.daily_pnl,
            positions_count=len(self.positions),
        )

    def _calculate_margin(self, quantity: float, price: float) -> float:
        """Calculate margin required for a position."""
        notional = quantity * price
        return notional / self.config.leverage

    def _margin_used(self) -> float:
        """Calculate total margin used by open positions."""
        total = 0.0
        for pair, pos in self.positions.items():
            price = self.prices.get(pair, pos.entry_price)
            total += self._calculate_margin(pos.quantity, price)
        return total

    def _free_margin(self) -> float:
        """Calculate free margin available."""
        state = self.get_account_state()
        return state.free_margin

    def _pip_value(self, pair: str) -> float:
        """Get pip value for a pair."""
        # For JPY pairs, 1 pip = 0.01
        # For other pairs, 1 pip = 0.0001
        if "JPY" in pair:
            return 0.01
        return 0.0001

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L counter (call at start of day)."""
        self.daily_pnl = 0.0

    def get_position(self, pair: str) -> Optional[Position]:
        """Get position for a pair."""
        return self.positions.get(pair)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())
