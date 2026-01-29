"""Tests for broker simulator and paper trading components (Phase 6).

Tests cover:
- Order creation and lifecycle
- Market order execution with slippage
- Limit and stop orders
- Position management
- Swap accrual
- Margin calculations
- Performance monitoring
"""

from datetime import datetime, timedelta
import pytest

from src.broker.orders import Order, OrderType, OrderSide, OrderStatus, Fill, Position
from src.broker.simulator import BrokerSimulator, BrokerConfig, AccountState
from src.monitoring.performance import PerformanceMonitor, TradeRecord


# ============== Order Tests ==============

class TestOrder:
    def test_order_creation(self):
        """Order is created with correct defaults."""
        order = Order(
            pair="USD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000,
        )
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.is_active()

    def test_filled_quantity(self):
        """Filled quantity is calculated from fills."""
        order = Order(
            pair="USD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000,
        )
        order.fills.append(Fill(
            fill_id="f1",
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=150.0,
            quantity=5000,
            commission=3.5,
            slippage=1.0,
        ))
        assert order.filled_quantity == 5000
        assert order.remaining_quantity == 5000

    def test_avg_fill_price(self):
        """Average fill price is volume-weighted."""
        order = Order(
            pair="USD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000,
        )
        order.fills.append(Fill("f1", order.order_id, datetime.now(), 150.0, 6000, 0, 0))
        order.fills.append(Fill("f2", order.order_id, datetime.now(), 150.5, 4000, 0, 0))
        # (150 * 6000 + 150.5 * 4000) / 10000 = 150.2
        assert order.avg_fill_price == pytest.approx(150.2, rel=1e-5)


class TestPosition:
    def test_long_position_pnl(self):
        """Long position P&L is calculated correctly."""
        pos = Position(
            pair="USD/JPY",
            side=OrderSide.BUY,
            quantity=10000,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=151.0,
        )
        # Price went up $1 on 10000 units = $10000 profit
        assert pos.unrealized_pnl == pytest.approx(10000, rel=1e-5)

    def test_short_position_pnl(self):
        """Short position P&L is calculated correctly."""
        pos = Position(
            pair="USD/JPY",
            side=OrderSide.SELL,
            quantity=10000,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=149.0,  # Price went down (good for short)
        )
        assert pos.unrealized_pnl == pytest.approx(10000, rel=1e-5)

    def test_swap_accrual(self):
        """Swap is added to position."""
        pos = Position(
            pair="USD/JPY",
            side=OrderSide.BUY,
            quantity=10000,
            entry_price=150.0,
            entry_time=datetime.now(),
        )
        pos.add_swap(1.37)
        pos.add_swap(1.37)
        assert pos.accumulated_swap == pytest.approx(2.74, rel=1e-5)


# ============== Broker Simulator Tests ==============

class TestBrokerSimulator:
    @pytest.fixture
    def broker(self):
        """Create broker with test config."""
        config = BrokerConfig(
            initial_balance=100000,
            leverage=50.0,
            min_slippage_pips=0.0,  # No slippage for predictable tests
            max_slippage_pips=0.0,
            commission_per_lot=7.0,
            order_latency_ms=(0, 0),  # No latency for tests
        )
        return BrokerSimulator(config)

    def test_initial_state(self, broker):
        """Broker starts with correct initial state."""
        state = broker.get_account_state()
        assert state.balance == 100000
        assert state.equity == 100000
        assert state.positions_count == 0

    def test_market_order_submitted(self, broker):
        """Market order is submitted correctly."""
        order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)
        assert order.status == OrderStatus.SUBMITTED
        assert len(broker.pending_orders) == 1

    def test_order_rejected_below_minimum(self, broker):
        """Order below minimum size is rejected."""
        order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.001)  # Below 0.01 min
        assert order.status == OrderStatus.REJECTED
        assert "minimum" in order.reject_reason.lower()

    def test_market_order_filled(self, broker):
        """Market order is filled at current price."""
        order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)
        fills = broker.execute_orders({"USD/JPY": 150.0})

        assert len(fills) == 1
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(150.0, rel=1e-3)

    def test_position_created_after_fill(self, broker):
        """Position is created after order fill."""
        broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)
        broker.execute_orders({"USD/JPY": 150.0})

        pos = broker.get_position("USD/JPY")
        assert pos is not None
        assert pos.side == OrderSide.BUY
        assert pos.quantity == 10000

    def test_commission_charged(self, broker):
        """Commission is deducted from balance."""
        initial = broker.balance
        broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)  # 0.1 lot
        broker.execute_orders({"USD/JPY": 150.0})

        # 0.1 lot * $7/lot = $0.70 commission
        assert broker.balance == pytest.approx(initial - 0.70, rel=1e-2)

    def test_limit_order_not_filled_immediately(self, broker):
        """Limit order waits for price to reach target."""
        order = broker.submit_limit_order("USD/JPY", OrderSide.BUY, 0.1, limit_price=149.0)
        fills = broker.execute_orders({"USD/JPY": 150.0})  # Current price above limit

        assert len(fills) == 0
        assert order.status == OrderStatus.SUBMITTED

    def test_limit_order_filled_at_target(self, broker):
        """Limit order fills when price reaches target."""
        order = broker.submit_limit_order("USD/JPY", OrderSide.BUY, 0.1, limit_price=149.0)
        fills = broker.execute_orders({"USD/JPY": 148.5})  # Price dropped below limit

        assert len(fills) == 1
        assert order.status == OrderStatus.FILLED

    def test_stop_order_triggers(self, broker):
        """Stop order triggers when price hits stop level."""
        order = broker.submit_stop_order("USD/JPY", OrderSide.SELL, 0.1, stop_price=149.0)
        fills = broker.execute_orders({"USD/JPY": 148.5})  # Price dropped to stop

        assert len(fills) == 1
        assert order.status == OrderStatus.FILLED

    def test_cancel_order(self, broker):
        """Pending order can be cancelled."""
        order = broker.submit_limit_order("USD/JPY", OrderSide.BUY, 0.1, limit_price=149.0)
        result = broker.cancel_order(order.order_id)

        assert result is True
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.pending_orders) == 0

    def test_close_position(self, broker):
        """Position can be closed."""
        broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)
        broker.execute_orders({"USD/JPY": 150.0})

        close_order = broker.close_position("USD/JPY")
        broker.execute_orders({"USD/JPY": 151.0})  # Close at higher price

        assert close_order is not None
        assert broker.get_position("USD/JPY") is None

    def test_swap_accrual(self, broker):
        """Swap is accrued for overnight positions."""
        broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)  # 0.1 lot (smaller to fit margin)
        broker.execute_orders({"USD/JPY": 150.0})

        swaps = broker.accrue_swap(datetime.now())
        assert "USD/JPY" in swaps
        assert swaps["USD/JPY"] > 0  # Long USD/JPY earns positive swap


class TestMarginCalculation:
    def test_margin_required(self):
        """Margin is calculated correctly."""
        config = BrokerConfig(leverage=50.0)
        broker = BrokerSimulator(config)

        # 100k position at 150 price with 50:1 leverage
        # Notional = 100000 * 150 = 15,000,000
        # Margin = 15,000,000 / 50 = 300,000
        margin = broker._calculate_margin(100000, 150.0)
        assert margin == pytest.approx(300000, rel=1e-5)

    def test_order_rejected_insufficient_margin(self):
        """Order is rejected if margin insufficient."""
        config = BrokerConfig(
            initial_balance=1000,  # Very small balance
            leverage=50.0,
            order_latency_ms=(0, 0),
        )
        broker = BrokerSimulator(config)

        # Try to open huge position
        order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 10.0)  # 10 lots
        fills = broker.execute_orders({"USD/JPY": 150.0})

        assert len(fills) == 0
        assert order.status == OrderStatus.REJECTED
        assert "margin" in order.reject_reason.lower()


# ============== Performance Monitor Tests ==============

class TestPerformanceMonitor:
    def test_initial_state(self):
        """Monitor starts with correct initial state."""
        monitor = PerformanceMonitor(initial_equity=100000)
        snapshot = monitor.get_snapshot()
        assert snapshot.equity == 100000
        assert snapshot.drawdown == 0.0
        assert snapshot.max_drawdown == 0.0

    def test_drawdown_calculation(self):
        """Drawdown is calculated correctly."""
        monitor = PerformanceMonitor(initial_equity=100000)
        monitor.update(
            equity=95000,
            balance=95000,
            unrealized_pnl=0,
            daily_pnl=-5000,
            positions_count=0,
        )
        snapshot = monitor.get_snapshot()
        # 5% drawdown from 100k to 95k
        assert snapshot.drawdown == pytest.approx(0.05, rel=1e-3)

    def test_high_water_mark_updates(self):
        """High water mark updates on new highs."""
        monitor = PerformanceMonitor(initial_equity=100000)
        monitor.update(equity=105000, balance=105000, unrealized_pnl=0, daily_pnl=5000, positions_count=0)
        monitor.update(equity=103000, balance=103000, unrealized_pnl=0, daily_pnl=3000, positions_count=0)

        snapshot = monitor.get_snapshot()
        assert snapshot.high_water_mark == 105000

    def test_trade_recording(self):
        """Trades are recorded correctly."""
        monitor = PerformanceMonitor(initial_equity=100000)
        trade = TradeRecord(
            timestamp=datetime.now(),
            pair="USD/JPY",
            side="BUY",
            entry_price=150.0,
            exit_price=151.0,
            quantity=10000,
            pnl=1000,
            pnl_pct=0.01,
            duration=timedelta(hours=5),
            swap_earned=1.37,
        )
        monitor.record_trade(trade)

        stats = monitor.get_trade_stats()
        assert stats["total_trades"] == 1
        assert stats["win_rate"] == 1.0

    def test_win_rate_calculation(self):
        """Win rate is calculated correctly."""
        monitor = PerformanceMonitor(initial_equity=100000)

        # 3 wins, 2 losses = 60% win rate
        for pnl in [100, 200, -50, 150, -100]:
            trade = TradeRecord(
                timestamp=datetime.now(),
                pair="USD/JPY",
                side="BUY",
                entry_price=150.0,
                exit_price=150.5 if pnl > 0 else 149.5,
                quantity=10000,
                pnl=pnl,
                pnl_pct=pnl / 100000,
                duration=timedelta(hours=1),
                swap_earned=0,
            )
            monitor.record_trade(trade)

        stats = monitor.get_trade_stats()
        assert stats["win_rate"] == pytest.approx(0.6, rel=1e-3)

    def test_profit_factor_calculation(self):
        """Profit factor is calculated correctly."""
        monitor = PerformanceMonitor(initial_equity=100000)

        # Gross profit = 450, Gross loss = 150 -> PF = 3.0
        for pnl in [100, 200, -50, 150, -100]:
            trade = TradeRecord(
                timestamp=datetime.now(),
                pair="USD/JPY",
                side="BUY",
                entry_price=150.0,
                exit_price=150.5,
                quantity=10000,
                pnl=pnl,
                pnl_pct=pnl / 100000,
                duration=timedelta(hours=1),
                swap_earned=0,
            )
            monitor.record_trade(trade)

        stats = monitor.get_trade_stats()
        assert stats["profit_factor"] == pytest.approx(3.0, rel=1e-3)

    def test_alert_triggered(self):
        """Alert is triggered when threshold exceeded."""
        alerts = []
        def alert_callback(alert_type, message, value):
            alerts.append((alert_type, message, value))

        monitor = PerformanceMonitor(
            initial_equity=100000,
            alert_callback=alert_callback,
            max_drawdown_alert=0.05,  # 5% threshold
            daily_loss_alert=0.10,  # 10% threshold (won't trigger)
        )

        # 6% drawdown should trigger drawdown alert (but not daily loss)
        monitor.update(equity=94000, balance=94000, unrealized_pnl=0, daily_pnl=-6000, positions_count=0)

        assert len(alerts) == 1
        assert alerts[0][0] == "MAX_DRAWDOWN"
