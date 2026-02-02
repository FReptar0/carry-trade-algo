"""Tests for the trading runner.

Uses mocked broker and components to test the runner's logic
without requiring an actual OANDA connection.
"""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.broker.orders import OrderSide, OrderStatus, Order, Fill
from src.engine.runner import RunnerConfig, TradingRunner
from src.validation.protocol import ProtocolStatus

UTC = ZoneInfo("UTC")


@pytest.fixture
def mock_broker():
    """Create a mock OandaBroker."""
    broker = MagicMock()
    broker.get_account_state.return_value = {
        "balance": 100000.0,
        "equity": 100000.0,
        "unrealized_pnl": 0.0,
        "margin_used": 0.0,
        "margin_available": 100000.0,
        "open_positions": 0,
    }
    broker.get_all_positions.return_value = []
    broker.fetch_candles.return_value = None
    return broker


@pytest.fixture
def runner_config(tmp_path):
    """Create a test runner config with temp paths."""
    return RunnerConfig(
        pairs=["USD/JPY"],
        check_interval_minutes=60,
        candle_lookback=300,
        position_size_units=10000,
        initial_equity=100000.0,
        log_dir=str(tmp_path / "logs"),
        db_path=str(tmp_path / "test.db"),
        events_file=str(tmp_path / "events.json"),
    )


@pytest.fixture
def runner(mock_broker, runner_config):
    """Create a TradingRunner with mocked dependencies."""
    r = TradingRunner(mock_broker, runner_config)
    return r


class TestRunnerConfig:
    """Tests for RunnerConfig defaults."""

    def test_default_pairs(self):
        config = RunnerConfig()
        assert "USD/JPY" in config.pairs
        assert "AUD/JPY" in config.pairs

    def test_default_interval(self):
        config = RunnerConfig()
        assert config.check_interval_minutes == 60

    def test_default_lookback(self):
        config = RunnerConfig()
        assert config.candle_lookback == 300


class TestRunnerInit:
    """Tests for runner initialization."""

    def test_creates_protocol(self, runner):
        assert runner.protocol is not None
        assert runner.protocol.status == ProtocolStatus.RUNNING

    def test_creates_strategy(self, runner):
        assert runner.strategy is not None

    def test_creates_monitor(self, runner):
        assert runner.monitor is not None
        assert runner.monitor.initial_equity == 100000.0

    def test_creates_circuit_breaker(self, runner):
        assert runner.circuit_breaker is not None

    def test_has_sync_positions_method(self, runner):
        """Verify _sync_positions exists and is callable (called from start)."""
        assert hasattr(runner, "_sync_positions")
        assert callable(runner._sync_positions)


class TestTickLogic:
    """Tests for the _tick method."""

    @patch("src.engine.runner.ForexMarketHours.is_market_open")
    def test_skips_when_market_closed(self, mock_market, runner):
        mock_market.return_value = False
        runner._tick()
        # Broker should not be called for candles when market is closed
        runner.broker.fetch_candles.assert_not_called()

    @patch("src.engine.runner.ForexMarketHours.is_market_open")
    def test_processes_pairs_when_open(self, mock_market, runner):
        mock_market.return_value = True
        runner._tick()
        # Account state should be checked
        runner.broker.get_account_state.assert_called()

    @patch("src.engine.runner.ForexMarketHours.is_market_open")
    def test_handles_broker_failure(self, mock_market, runner):
        mock_market.return_value = True
        runner.broker.get_account_state.return_value = None
        # Should not raise
        runner._tick()


class TestProcessPair:
    """Tests for pair processing logic."""

    def test_skips_insufficient_data(self, runner):
        # Only 100 bars (need 220+)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
                "open": [155.0] * 100,
                "high": [155.5] * 100,
                "low": [154.5] * 100,
                "close": [155.2] * 100,
                "volume": [1000] * 100,
            }
        )
        runner.broker.fetch_candles.return_value = df
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        # Should not raise
        runner._process_pair("USD/JPY", now, in_blackout=False)
        # No orders should be placed
        runner.broker.submit_market_order.assert_not_called()

    def test_skips_entry_during_blackout(self, runner):
        runner.broker.fetch_candles.return_value = None
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._process_pair("USD/JPY", now, in_blackout=True)
        runner.broker.submit_market_order.assert_not_called()

    def test_skips_entry_when_halted(self, runner):
        runner.circuit_breaker.trading_halted = True
        runner.circuit_breaker.halt_timestamp = datetime.now()
        runner.broker.fetch_candles.return_value = None
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._process_pair("USD/JPY", now, in_blackout=False)
        runner.broker.submit_market_order.assert_not_called()


class TestPositionManagement:
    """Tests for opening and closing positions."""

    def test_open_position(self, runner):
        fill = Fill(
            fill_id="123",
            order_id="456",
            timestamp=datetime.now(),
            price=155.500,
            quantity=10000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="USD/JPY",
            side=OrderSide.BUY,
            order_type=MagicMock(),
            quantity=10000,
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.submit_market_order.return_value = order

        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._open_position("USD/JPY", "Test entry", now)

        assert "USD/JPY" in runner._strategy_positions
        assert runner._trades_opened_today == 1

    def test_close_position(self, runner):
        # Set up an existing position
        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.500,
            "entry_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            "units": 10000,
        }

        fill = Fill(
            fill_id="789",
            order_id="012",
            timestamp=datetime.now(),
            price=156.000,
            quantity=10000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=MagicMock(),
            quantity=10000,
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = order

        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)
        runner._close_position("USD/JPY", "Test exit", now)

        assert "USD/JPY" not in runner._strategy_positions
        assert runner._trades_closed_today == 1


class TestDayManagement:
    """Tests for day transitions and recording."""

    def test_new_day_resets_counters(self, runner):
        runner._trades_opened_today = 5
        runner._trades_closed_today = 3
        runner._cb_triggered_today = True

        # Force new day
        runner._today = date(2026, 1, 31)
        runner._check_new_day(datetime(2026, 2, 1, 0, 0, tzinfo=UTC))

        assert runner._trades_opened_today == 0
        assert runner._trades_closed_today == 0
        assert runner._cb_triggered_today is False

    def test_same_day_no_reset(self, runner):
        runner._today = date(2026, 2, 1)
        runner._trades_opened_today = 5

        runner._check_new_day(datetime(2026, 2, 1, 12, 0, tzinfo=UTC))
        assert runner._trades_opened_today == 5


class TestProtocolRestore:
    """Tests for protocol state restoration."""

    def test_creates_new_protocol_if_none_saved(self, runner):
        assert runner.protocol.status == ProtocolStatus.RUNNING
        assert len(runner.protocol.days) == 0

    def test_restores_saved_protocol(self, mock_broker, runner_config):
        # Create and save a protocol
        runner1 = TradingRunner(mock_broker, runner_config)
        runner1.protocol.record_day(
            MagicMock(
                date=date(2026, 2, 1),
                starting_equity=100000,
                ending_equity=100500,
                daily_pnl=500,
                daily_return=0.005,
                trades_opened=1,
                trades_closed=0,
                max_drawdown_today=0.01,
                regime="LIVE",
                circuit_breaker_triggered=False,
                notes="",
                is_profitable=True,
            )
        )
        runner1.store.save_protocol_state(runner1.protocol)
        runner1.store.save_daily_result(
            MagicMock(
                date=date(2026, 2, 1),
                starting_equity=100000,
                ending_equity=100500,
                daily_pnl=500,
                daily_return=0.005,
                trades_opened=1,
                trades_closed=0,
                max_drawdown_today=0.01,
                regime="LIVE",
                circuit_breaker_triggered=False,
                notes="",
            )
        )

        # Create new runner — should restore
        runner2 = TradingRunner(mock_broker, runner_config)
        assert len(runner2.protocol.days) == 1


class TestCloseAllPositions:
    """Tests for emergency position closing."""

    def test_closes_all_tracked_positions(self, runner):
        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 155.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
            },
            "AUD/JPY": {
                "entry_price": 98.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
            },
        }

        fill = Fill(
            fill_id="x",
            order_id="y",
            timestamp=datetime.now(),
            price=155.0,
            quantity=10000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=MagicMock(),
            quantity=10000,
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = order

        runner._close_all_positions("abort")
        assert len(runner._strategy_positions) == 0
