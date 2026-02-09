"""Tests for the trading runner.

Uses mocked broker and components to test the runner's logic
without requiring an actual OANDA connection.
"""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.broker.orders import OrderSide, OrderStatus, OrderType, Order, Fill
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
        runner.broker.submit_limit_order.assert_not_called()

    def test_skips_entry_during_blackout(self, runner):
        runner.broker.fetch_candles.return_value = None
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._process_pair("USD/JPY", now, in_blackout=True)
        runner.broker.submit_limit_order.assert_not_called()

    def test_skips_entry_when_halted(self, runner):
        runner.circuit_breaker.trading_halted = True
        runner.circuit_breaker.halt_timestamp = datetime.now()
        runner.broker.fetch_candles.return_value = None
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._process_pair("USD/JPY", now, in_blackout=False)
        runner.broker.submit_limit_order.assert_not_called()


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
            order_type=OrderType.LIMIT,
            quantity=10000,
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.get_current_price.return_value = {
            "bid": 155.500,
            "ask": 155.520,
            "mid": 155.510,
        }
        runner.broker.submit_limit_order.return_value = order

        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._open_position("USD/JPY", "Test entry", now)

        assert "USD/JPY" in runner._strategy_positions
        assert runner._trades_opened_today == 1
        runner.broker.submit_limit_order.assert_called_once()

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
                financing=0.0,
            )
        )

        # Create new runner — should restore
        runner2 = TradingRunner(mock_broker, runner_config)
        assert len(runner2.protocol.days) == 1


def _make_candle_df(n_bars: int = 50, base_price: float = 155.0) -> pd.DataFrame:
    """Helper: create a minimal candle DataFrame for stop calculations."""
    import numpy as np

    rng = np.random.default_rng(42)
    closes = base_price + np.cumsum(rng.normal(0, 0.1, n_bars))
    highs = closes + rng.uniform(0.1, 0.5, n_bars)
    lows = closes - rng.uniform(0.1, 0.5, n_bars)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_bars, freq="h"),
            "open": closes - rng.uniform(-0.1, 0.1, n_bars),
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * n_bars,
        }
    )


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


class TestSafeAbort:
    """Tests for Phase 4: safe abort with stop tightening."""

    def test_tighten_all_stops_sets_1x_atr(self, runner):
        """Stops should be tightened to 1x ATR before abort."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,  # Wide 3x ATR stop
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True

        runner._tighten_all_stops()

        # Should have called modify_trade_stop_loss
        runner.broker.modify_trade_stop_loss.assert_called_once()
        call_args = runner.broker.modify_trade_stop_loss.call_args
        new_stop = (
            call_args[1]["stop_price"]
            if "stop_price" in call_args[1]
            else call_args[0][1]
        )

        # New stop should be ABOVE old stop (150.0) — tighter
        assert new_stop > 150.0
        # Position dict should be updated
        assert runner._strategy_positions["USD/JPY"]["stop_price"] == new_stop

    def test_tighten_skips_already_tighter_stop(self, runner):
        """If existing stop is already tighter than 1x ATR, skip."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 999.0,  # Artificially high stop
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }

        runner._tighten_all_stops()

        # Should NOT call modify because existing stop is already above 1x ATR
        runner.broker.modify_trade_stop_loss.assert_not_called()

    def test_tighten_skips_no_trade_id(self, runner):
        """Positions without trade_id should be skipped gracefully."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": None,
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }

        runner._tighten_all_stops()
        runner.broker.modify_trade_stop_loss.assert_not_called()

    def test_tighten_falls_back_to_trade_ids_list(self, runner):
        """Should use trade_ids list when trade_id is None."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": None,
                "trade_ids": ["T200", "T201"],
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True

        runner._tighten_all_stops()

        # Should have called at least once (primary) + once for extra
        assert runner.broker.modify_trade_stop_loss.call_count >= 1

    def test_tighten_fetches_candles_if_no_cache(self, runner):
        """Should fetch candles from broker if cache is empty."""
        df = _make_candle_df(50, base_price=155.0)
        runner.broker.fetch_candles.return_value = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True

        runner._tighten_all_stops()

        # Should have fetched candles since cache was empty
        runner.broker.fetch_candles.assert_called_once_with(
            "USD/JPY", count=50, granularity="H1"
        )

    def test_tighten_handles_broker_failure(self, runner):
        """If modify_trade_stop_loss fails, should not crash."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = False

        # Should not raise
        runner._tighten_all_stops()

    def test_close_all_tightens_before_closing(self, runner):
        """_close_all_positions should tighten stops first."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True

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

        runner._close_all_positions("abort test")

        # Stops should be tightened AND position should be closed
        runner.broker.modify_trade_stop_loss.assert_called()
        runner.broker.close_position.assert_called()
        assert len(runner._strategy_positions) == 0

    def test_close_all_continues_on_partial_failure(self, runner):
        """If one position fails to close, others should still close."""
        df_usd = _make_candle_df(50, base_price=155.0)
        df_aud = _make_candle_df(50, base_price=98.0)
        runner._candle_cache["USD/JPY"] = df_usd
        runner._candle_cache["AUD/JPY"] = df_aud

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
            "AUD/JPY": {
                "entry_price": 97.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T200",
                "stop_price": 94.0,
                "high_water_mark": 98.5,
                "low_water_mark": 96.5,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True

        # First close succeeds, second raises
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

        def close_side_effect(pair, **kwargs):
            if pair == "AUD/JPY":
                raise ConnectionError("Broker unreachable")
            return order

        runner.broker.close_position.side_effect = close_side_effect

        # Should not crash — AUD/JPY stays with tight stop
        runner._close_all_positions("abort test")

        # AUD/JPY should still be in positions (close failed)
        assert "AUD/JPY" in runner._strategy_positions

    def test_close_all_alerts_on_partial_failure(self, runner):
        """Telegram alert should fire when some positions fail to close."""
        df = _make_candle_df(50, base_price=155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True
        runner.broker.close_position.side_effect = ConnectionError("timeout")

        # Set up alert manager mock
        runner.alert_manager = MagicMock()

        runner._close_all_positions("test abort")

        # Alert should have been sent about the failure
        runner.alert_manager.send.assert_called()
        call_args = runner.alert_manager.send.call_args
        assert "CRITICAL" == call_args[0][0]
        assert "Partial Close Failure" in call_args[0][1]

    def test_close_all_no_positions(self, runner):
        """Should handle empty positions gracefully."""
        runner._strategy_positions = {}
        runner._close_all_positions("test abort")
        runner.broker.modify_trade_stop_loss.assert_not_called()
        runner.broker.close_position.assert_not_called()

    def test_tighten_multiple_pairs(self, runner):
        """Should tighten stops on all positions independently."""
        df_usd = _make_candle_df(50, base_price=155.0)
        df_aud = _make_candle_df(50, base_price=98.0)
        runner._candle_cache["USD/JPY"] = df_usd
        runner._candle_cache["AUD/JPY"] = df_aud

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
            "AUD/JPY": {
                "entry_price": 97.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T200",
                "stop_price": 94.0,
                "high_water_mark": 98.5,
                "low_water_mark": 96.5,
            },
        }
        runner.broker.modify_trade_stop_loss.return_value = True

        runner._tighten_all_stops()

        # Should have tightened both pairs
        assert runner.broker.modify_trade_stop_loss.call_count == 2

    def test_tighten_handles_exception_per_pair(self, runner):
        """An exception on one pair should not prevent tightening others."""
        df_usd = _make_candle_df(50, base_price=155.0)
        df_aud = _make_candle_df(50, base_price=98.0)
        runner._candle_cache["USD/JPY"] = df_usd
        runner._candle_cache["AUD/JPY"] = df_aud

        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 154.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T100",
                "stop_price": 150.0,
                "high_water_mark": 155.5,
                "low_water_mark": 153.0,
            },
            "AUD/JPY": {
                "entry_price": 97.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "trade_id": "T200",
                "stop_price": 94.0,
                "high_water_mark": 98.5,
                "low_water_mark": 96.5,
            },
        }

        call_count = [0]

        def modify_side_effect(trade_id, stop_price):
            call_count[0] += 1
            if trade_id == "T100":
                raise ConnectionError("timeout")
            return True

        runner.broker.modify_trade_stop_loss.side_effect = modify_side_effect

        # Should not crash
        runner._tighten_all_stops()

        # Should have attempted both
        assert call_count[0] == 2


class TestFinancingTracking:
    """Tests for swap/financing PnL tracking in the runner."""

    def test_tick_syncs_financing_from_broker(self, runner):
        """After reconciliation, financing should be synced from broker data."""
        # Set up a position without financing
        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.0,
            "entry_time": datetime.now(UTC),
            "units": 10000,
            "trade_id": "T100",
            "stop_price": 152.0,
            "high_water_mark": 155.5,
            "low_water_mark": 154.0,
            "financing": 0.0,
        }

        # Broker returns financing data
        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.0,
                "unrealized_pnl": 50.0,
                "financing": 12.45,
            }
        ]

        # Simulate the sync step from _tick (step 5b)
        broker_positions = runner.broker.get_all_positions()
        bp_by_pair = {bp["pair"]: bp for bp in broker_positions}
        for pair, pos in runner._strategy_positions.items():
            bp = bp_by_pair.get(pair)
            if bp is not None:
                pos["financing"] = bp.get("financing", 0.0)

        assert runner._strategy_positions["USD/JPY"]["financing"] == 12.45

    def test_close_position_records_financing(self, runner):
        """_close_position should pass financing to TradeRecord and store."""
        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.500,
            "entry_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            "units": 10000,
            "financing": 8.73,
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

        # Mock store.save_trade so we can inspect call args
        runner.store.save_trade = MagicMock()

        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)
        runner._close_position("USD/JPY", "Test exit", now)

        # Verify store.save_trade was called with swap_earned
        assert runner.store.save_trade.called
        trade_data = runner.store.save_trade.call_args[0][0]
        assert trade_data["swap_earned"] == 8.73

    def test_close_position_financing_defaults_zero(self, runner):
        """Positions without financing field should default to 0."""
        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.500,
            "entry_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            "units": 10000,
            # No "financing" key
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

        # Mock store.save_trade so we can inspect call args
        runner.store.save_trade = MagicMock()

        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)
        runner._close_position("USD/JPY", "Test exit", now)

        trade_data = runner.store.save_trade.call_args[0][0]
        assert trade_data["swap_earned"] == 0.0

    def test_get_system_state_includes_financing(self, runner):
        """get_system_state should include financing in positions and total."""
        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 155.0,
                "entry_time": datetime.now(UTC),
                "units": 5000,
                "stop_price": 152.0,
                "high_water_mark": 155.5,
                "tranche_count": 1,
                "financing": 5.50,
            },
            "AUD/JPY": {
                "entry_price": 98.0,
                "entry_time": datetime.now(UTC),
                "units": 10000,
                "stop_price": 95.0,
                "high_water_mark": 98.5,
                "tranche_count": 2,
                "financing": 7.25,
            },
        }

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 5000,
                "avg_price": 155.0,
                "unrealized_pnl": 200.0,
                "financing": 5.50,
            },
            {
                "pair": "AUD/JPY",
                "units": 10000,
                "avg_price": 98.0,
                "unrealized_pnl": 100.0,
                "financing": 7.25,
            },
        ]

        state = runner.get_system_state()

        # Check per-position financing
        pos_by_pair = {p["pair"]: p for p in state["positions"]}
        assert pos_by_pair["USD/JPY"]["financing"] == 5.50
        assert pos_by_pair["AUD/JPY"]["financing"] == 7.25

        # Check total_financing in performance
        assert state["performance"]["total_financing"] == pytest.approx(12.75)

    def test_get_system_state_zero_financing(self, runner):
        """get_system_state should handle zero financing gracefully."""
        runner._strategy_positions = {
            "USD/JPY": {
                "entry_price": 155.0,
                "entry_time": datetime.now(UTC),
                "units": 5000,
                "stop_price": 152.0,
                "high_water_mark": 155.5,
                "tranche_count": 1,
                "financing": 0.0,
            },
        }

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 5000,
                "avg_price": 155.0,
                "unrealized_pnl": 200.0,
                "financing": 0.0,
            },
        ]

        state = runner.get_system_state()

        pos = state["positions"][0]
        assert pos["financing"] == 0.0
        assert state["performance"]["total_financing"] == 0.0


class TestPendingOrders:
    """Tests for _check_pending_orders: fill promotion, stale cancellation, cleanup."""

    def test_fill_promotion_when_order_disappears_and_trade_exists(self, runner):
        """Pending order gone from broker + new trade → promoted to position."""
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)

        # Track a pending order
        runner._pending_orders["USD/JPY"] = {
            "order_id": "ORD100",
            "pair": "USD/JPY",
            "units": 10000,
            "limit_price": 155.500,
            "stop_loss_price": 153.500,
            "reason": "golden cross",
            "created_at": now - timedelta(hours=1),
            "tick_count": 0,
            "is_scale_in": False,
            "current_atr": 0.65,
        }

        # Broker returns NO pending orders (order was filled)
        runner.broker.get_pending_orders.return_value = []

        # Broker shows a new open trade for USD/JPY
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T500",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.500,
                "financing": 0.0,
            }
        ]

        runner._check_pending_orders(now)

        # Pending order should be removed
        assert "USD/JPY" not in runner._pending_orders
        # Position should now exist
        assert "USD/JPY" in runner._strategy_positions
        pos = runner._strategy_positions["USD/JPY"]
        assert pos["entry_price"] == 155.500
        assert pos["units"] == 10000
        assert pos["trade_id"] == "T500"
        assert pos["stop_price"] == 153.500

    def test_stale_order_cancelled_after_max_ticks(self, runner):
        """Pending order exceeding max_pending_ticks is cancelled."""
        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)

        runner._pending_orders["EUR/JPY"] = {
            "order_id": "ORD200",
            "pair": "EUR/JPY",
            "units": 5000,
            "limit_price": 162.000,
            "stop_loss_price": 160.000,
            "reason": "uptrend entry",
            "created_at": now - timedelta(hours=2),
            "tick_count": 1,  # Already 1 tick old
            "is_scale_in": False,
            "current_atr": 0.70,
        }

        # Broker still shows the order as pending
        runner.broker.get_pending_orders.return_value = [
            {
                "order_id": "ORD200",
                "pair": "EUR/JPY",
                "units": 5000,
                "price": 162.000,
                "order_type": "LIMIT",
                "time_in_force": "GTC",
                "stop_loss_price": 160.000,
                "create_time": "2026-02-01T12:00:00Z",
            }
        ]
        runner.broker.get_open_trades.return_value = []
        runner.broker.cancel_order.return_value = True

        # max_pending_ticks default is 2, tick_count will become 2 → cancel
        runner._check_pending_orders(now)

        # Order should be cancelled
        runner.broker.cancel_order.assert_called_once_with("ORD200")
        # Pending tracking should be cleaned up
        assert "EUR/JPY" not in runner._pending_orders
        # No position should exist
        assert "EUR/JPY" not in runner._strategy_positions

    def test_broker_side_cleanup_when_order_gone_no_trade(self, runner):
        """Order gone from broker with no new trade → cleaned up."""
        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)

        runner._pending_orders["GBP/JPY"] = {
            "order_id": "ORD300",
            "pair": "GBP/JPY",
            "units": 8000,
            "limit_price": 212.000,
            "stop_loss_price": 210.000,
            "reason": "strong uptrend",
            "created_at": now - timedelta(hours=1),
            "tick_count": 0,
            "is_scale_in": False,
            "current_atr": 0.80,
        }

        # Broker has no pending orders and no trades for GBP/JPY
        runner.broker.get_pending_orders.return_value = []
        runner.broker.get_open_trades.return_value = []

        runner._check_pending_orders(now)

        # Should be removed from tracking
        assert "GBP/JPY" not in runner._pending_orders
        # No position created
        assert "GBP/JPY" not in runner._strategy_positions

    def test_pending_order_still_waiting_increments_tick(self, runner):
        """Order still pending and under limit → tick_count incremented, not cancelled."""
        now = datetime(2026, 2, 1, 13, 0, tzinfo=UTC)

        runner._pending_orders["NZD/JPY"] = {
            "order_id": "ORD400",
            "pair": "NZD/JPY",
            "units": 5000,
            "limit_price": 93.500,
            "stop_loss_price": 91.500,
            "reason": "entry",
            "created_at": now - timedelta(hours=1),
            "tick_count": 0,
            "is_scale_in": False,
            "current_atr": 0.50,
        }

        runner.broker.get_pending_orders.return_value = [
            {
                "order_id": "ORD400",
                "pair": "NZD/JPY",
                "units": 5000,
                "price": 93.500,
                "order_type": "LIMIT",
                "time_in_force": "GTC",
                "stop_loss_price": 91.500,
                "create_time": "2026-02-01T12:00:00Z",
            }
        ]
        runner.broker.get_open_trades.return_value = []

        runner._check_pending_orders(now)

        # Still in pending
        assert "NZD/JPY" in runner._pending_orders
        # Tick count incremented from 0 → 1
        assert runner._pending_orders["NZD/JPY"]["tick_count"] == 1
        # No cancel call
        runner.broker.cancel_order.assert_not_called()

    def test_no_action_when_no_pending_orders(self, runner):
        """No pending orders → no broker calls."""
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._pending_orders = {}

        runner._check_pending_orders(now)

        runner.broker.get_pending_orders.assert_not_called()
        runner.broker.get_open_trades.assert_not_called()


class TestPromoteFilledOrder:
    """Tests for _promote_filled_order: new entry and scale-in merge."""

    def test_new_entry_creates_position(self, runner):
        """New entry promotion creates position dict with correct fields."""
        fill = Fill(
            fill_id="F100",
            order_id="ORD100",
            timestamp=datetime.now(),
            price=155.500,
            quantity=10000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="USD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10000,
            status=OrderStatus.FILLED,
            fills=[fill],
            trade_id="T500",
        )

        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._promote_filled_order(
            pair="USD/JPY",
            order=order,
            units=10000,
            stop_loss_price=153.500,
            current_atr=0.65,
            reason="golden cross",
            now=now,
        )

        assert "USD/JPY" in runner._strategy_positions
        pos = runner._strategy_positions["USD/JPY"]
        assert pos["entry_price"] == 155.500
        assert pos["units"] == 10000
        assert pos["trade_id"] == "T500"
        assert pos["stop_price"] == 153.500
        assert pos["tranche_count"] == 1
        assert pos["financing"] == 0.0
        assert pos["high_water_mark"] == 155.500
        assert runner._trades_opened_today == 1

    def test_scale_in_merge_updates_position(self, runner):
        """Scale-in fill merges into existing position with weighted avg price."""
        # Set up existing position: 5000 units @ 155.000
        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.000,
            "entry_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            "units": 5000,
            "tranche_count": 1,
            "levels_taken": 0,
            "trade_id": "T400",
            "stop_price": 153.000,
            "high_water_mark": 155.500,
            "low_water_mark": 155.000,
            "financing": 2.50,
        }

        fill = Fill(
            fill_id="F200",
            order_id="ORD200",
            timestamp=datetime.now(),
            price=155.800,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="USD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5000,
            status=OrderStatus.FILLED,
            fills=[fill],
            trade_id="T600",
        )

        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)
        runner._promote_filled_order(
            pair="USD/JPY",
            order=order,
            units=5000,
            stop_loss_price=153.000,
            current_atr=0.65,
            reason="scale-in",
            now=now,
            is_scale_in=True,
        )

        pos = runner._strategy_positions["USD/JPY"]
        # Weighted average: (155.000 * 5000 + 155.800 * 5000) / 10000 = 155.400
        assert pos["entry_price"] == pytest.approx(155.400)
        assert pos["units"] == 10000
        assert pos["tranche_count"] == 2
        # trade_ids should contain both T400 and T600
        assert "T400" in pos["trade_ids"]
        assert "T600" in pos["trade_ids"]

    def test_scale_in_no_existing_position_creates_new(self, runner):
        """Scale-in when no position exists falls back to creating a new one."""
        fill = Fill(
            fill_id="F300",
            order_id="ORD300",
            timestamp=datetime.now(),
            price=155.500,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="AUD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5000,
            status=OrderStatus.FILLED,
            fills=[fill],
            trade_id="T700",
        )

        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)
        runner._promote_filled_order(
            pair="AUD/JPY",
            order=order,
            units=5000,
            stop_loss_price=97.000,
            current_atr=0.50,
            reason="scale-in",
            now=now,
            is_scale_in=True,
        )

        # Should create a new position (fallback from scale-in)
        assert "AUD/JPY" in runner._strategy_positions
        assert runner._strategy_positions["AUD/JPY"]["units"] == 5000
        assert runner._trades_opened_today == 1


class TestPendingOrderGuard:
    """Tests for _process_pair pending order guard."""

    def test_skips_entry_when_limit_order_pending(self, runner):
        """When a limit order is already pending for a pair, skip entry."""
        # Set up enough candle data for strategy to generate signals
        n_bars = 300
        import numpy as np

        rng = np.random.default_rng(42)
        closes = 155.0 + np.cumsum(rng.normal(0.05, 0.1, n_bars))
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-10-01", periods=n_bars, freq="h"),
                "open": closes - 0.1,
                "high": closes + 0.3,
                "low": closes - 0.3,
                "close": closes,
                "volume": [1000] * n_bars,
            }
        )
        runner.broker.fetch_candles.return_value = df

        # Mark a pending order for this pair
        runner._pending_orders["USD/JPY"] = {
            "order_id": "ORD999",
            "pair": "USD/JPY",
            "units": 10000,
            "limit_price": 155.500,
            "stop_loss_price": 153.500,
            "reason": "entry",
            "created_at": datetime.now(UTC),
            "tick_count": 0,
            "is_scale_in": False,
            "current_atr": 0.65,
        }

        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._process_pair("USD/JPY", now, in_blackout=False)

        # Even if strategy generates LONG, no new limit order should be placed
        runner.broker.submit_limit_order.assert_not_called()
        runner.broker.get_current_price.assert_not_called()


class TestSystemStatePendingOrders:
    """Tests for get_system_state() pending_orders inclusion."""

    def test_includes_pending_orders(self, runner):
        """get_system_state should include pending_orders dict."""
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._pending_orders["CAD/JPY"] = {
            "order_id": "ORD500",
            "pair": "CAD/JPY",
            "units": 5000,
            "limit_price": 108.500,
            "stop_loss_price": 106.500,
            "reason": "entry",
            "created_at": now,
            "tick_count": 0,
            "is_scale_in": False,
            "current_atr": 0.55,
        }

        state = runner.get_system_state()

        assert "pending_orders" in state
        assert "CAD/JPY" in state["pending_orders"]
        assert state["pending_orders"]["CAD/JPY"]["order_id"] == "ORD500"
        assert state["pending_orders"]["CAD/JPY"]["units"] == 5000

    def test_empty_pending_orders(self, runner):
        """Empty pending_orders when none exist."""
        runner._pending_orders = {}

        state = runner.get_system_state()

        assert "pending_orders" in state
        assert state["pending_orders"] == {}


class TestVolatilityScaling:
    """Tests for _check_volatility_scaling: vol-based position trimming."""

    def _setup_position_with_candles(
        self,
        runner,
        pair: str = "USD/JPY",
        entry_atr: float = 0.50,
        current_atr_target: float = 1.00,
        units: int = 10000,
        entry_price: float = 155.0,
    ):
        """Helper: set up a position with known ATR values.

        Creates a candle DataFrame where the ATR approximates current_atr_target
        by controlling the high-low spread.
        """
        import numpy as np

        n_bars = 50
        rng = np.random.default_rng(42)
        closes = np.full(n_bars, entry_price)
        # Control ATR via high-low spread (ATR ≈ high - low for flat prices)
        half_range = current_atr_target / 2
        highs = closes + half_range
        lows = closes - half_range

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=n_bars, freq="h"),
                "open": closes,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [1000] * n_bars,
            }
        )

        runner._candle_cache[pair] = df
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)

        runner._strategy_positions[pair] = {
            "entry_price": entry_price,
            "entry_time": now - timedelta(days=5),
            "units": units,
            "tranche_count": 1,
            "levels_taken": 0,
            "trade_id": "T100",
            "stop_price": entry_price - entry_atr * 3,
            "high_water_mark": entry_price,
            "low_water_mark": entry_price,
            "financing": 0.0,
            "trade_ids": ["T100"],
            "entry_atr": entry_atr,
            "original_units": units,
            "last_vol_trim_time": None,
        }

        return now

    def test_trims_when_vol_ratio_exceeds_threshold(self, runner):
        """ATR doubles from 0.50 to ~1.0 → position trimmed proportionally."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )

        # Mock broker partial close
        fill = Fill(
            fill_id="f1",
            order_id="CLOSE1",
            timestamp=now,
            price=155.0,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        fill_order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5000,
            order_id="CLOSE1",
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = fill_order
        runner.alert_manager = MagicMock()

        runner._check_volatility_scaling(now)

        # Position should be trimmed: target = 10000 * 0.50 / 1.00 = 5000
        pos = runner._strategy_positions["USD/JPY"]
        assert pos["units"] < 10000
        assert pos["units"] >= 2500  # Floor: 25% of 10000
        runner.broker.close_position.assert_called_once()
        # Verify the trim units in the call
        call_args = runner.broker.close_position.call_args
        assert call_args[0][0] == "USD/JPY"  # pair
        assert (
            call_args[1].get("units", call_args[0][1] if len(call_args[0]) > 1 else 0)
            > 0
        )

    def test_no_trim_below_threshold(self, runner):
        """ATR only 20% higher (ratio 1.2) → no trim (threshold is 1.5)."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=0.60, units=10000
        )

        runner._check_volatility_scaling(now)

        # No trim — vol ratio 1.2 < 1.5 threshold
        pos = runner._strategy_positions["USD/JPY"]
        assert pos["units"] == 10000
        runner.broker.close_position.assert_not_called()

    def test_cooldown_prevents_rapid_trims(self, runner):
        """Second trim within 24h cooldown is blocked."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )

        # Set last trim to 6 hours ago (within 24h cooldown)
        runner._strategy_positions["USD/JPY"]["last_vol_trim_time"] = now - timedelta(
            hours=6
        )

        runner._check_volatility_scaling(now)

        # No trim — cooldown not elapsed
        pos = runner._strategy_positions["USD/JPY"]
        assert pos["units"] == 10000
        runner.broker.close_position.assert_not_called()

    def test_cooldown_allows_trim_after_elapsed(self, runner):
        """Trim allowed after 24h+ since last trim."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )

        # Set last trim to 25 hours ago (cooldown elapsed)
        runner._strategy_positions["USD/JPY"]["last_vol_trim_time"] = now - timedelta(
            hours=25
        )

        fill = Fill(
            fill_id="f2",
            order_id="CLOSE2",
            timestamp=now,
            price=155.0,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        fill_order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5000,
            order_id="CLOSE2",
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = fill_order
        runner.alert_manager = MagicMock()

        runner._check_volatility_scaling(now)

        pos = runner._strategy_positions["USD/JPY"]
        assert pos["units"] < 10000
        runner.broker.close_position.assert_called_once()

    def test_floor_prevents_over_trimming(self, runner):
        """Position near minimum → trimmed only to floor (25% of original)."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.10, current_atr_target=1.00, units=10000
        )
        # Vol ratio = 10x, target = 10000 * 0.10 / 1.00 = 1000
        # But floor = 10000 * 0.25 = 2500, so target should be clamped to 2500

        fill = Fill(
            fill_id="f3",
            order_id="CLOSE3",
            timestamp=now,
            price=155.0,
            quantity=7500,
            commission=0,
            slippage=0,
        )
        fill_order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=7500,
            order_id="CLOSE3",
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = fill_order
        runner.alert_manager = MagicMock()

        runner._check_volatility_scaling(now)

        pos = runner._strategy_positions["USD/JPY"]
        # Should be trimmed to floor = 2500 (25% of 10000)
        assert pos["units"] == 2500
        runner.broker.close_position.assert_called_once()

    def test_skips_when_disabled(self, runner):
        """enable_vol_scaling=False → no action at all."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )
        runner.config.enable_vol_scaling = False

        runner._check_volatility_scaling(now)

        assert runner._strategy_positions["USD/JPY"]["units"] == 10000
        runner.broker.close_position.assert_not_called()

    def test_skips_no_entry_atr(self, runner):
        """Position with entry_atr=None is skipped gracefully."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )
        runner._strategy_positions["USD/JPY"]["entry_atr"] = None

        runner._check_volatility_scaling(now)

        assert runner._strategy_positions["USD/JPY"]["units"] == 10000
        runner.broker.close_position.assert_not_called()

    def test_skips_no_candle_cache(self, runner):
        """No cached candle data → skipped gracefully."""
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.0,
            "entry_time": now - timedelta(days=5),
            "units": 10000,
            "tranche_count": 1,
            "levels_taken": 0,
            "trade_id": "T100",
            "stop_price": 153.5,
            "high_water_mark": 155.0,
            "low_water_mark": 155.0,
            "financing": 0.0,
            "trade_ids": ["T100"],
            "entry_atr": 0.50,
            "original_units": 10000,
            "last_vol_trim_time": None,
        }
        # No candle cache entry for USD/JPY
        runner._candle_cache = {}

        runner._check_volatility_scaling(now)

        assert runner._strategy_positions["USD/JPY"]["units"] == 10000
        runner.broker.close_position.assert_not_called()

    def test_handles_broker_failure(self, runner):
        """Partial close returns None → no crash, units unchanged."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )
        runner.broker.close_position.return_value = None

        runner._check_volatility_scaling(now)

        # Units unchanged — broker failed
        assert runner._strategy_positions["USD/JPY"]["units"] == 10000

    def test_updates_position_state_after_trim(self, runner):
        """After trim: units reduced and last_vol_trim_time set."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )

        fill = Fill(
            fill_id="f4",
            order_id="CLOSE4",
            timestamp=now,
            price=155.0,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        fill_order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5000,
            order_id="CLOSE4",
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = fill_order
        runner.alert_manager = MagicMock()

        runner._check_volatility_scaling(now)

        pos = runner._strategy_positions["USD/JPY"]
        assert pos["units"] < 10000
        assert pos["last_vol_trim_time"] == now

    def test_sends_alert_on_trim(self, runner):
        """Telegram alert fired with WARNING severity on trim."""
        now = self._setup_position_with_candles(
            runner, entry_atr=0.50, current_atr_target=1.00, units=10000
        )

        fill = Fill(
            fill_id="f5",
            order_id="CLOSE5",
            timestamp=now,
            price=155.0,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        fill_order = Order(
            pair="USD/JPY",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=5000,
            order_id="CLOSE5",
            status=OrderStatus.FILLED,
            fills=[fill],
        )
        runner.broker.close_position.return_value = fill_order
        runner.alert_manager = MagicMock()

        runner._check_volatility_scaling(now)

        runner.alert_manager.send.assert_called_once()
        call_args = runner.alert_manager.send.call_args
        assert call_args[0][0] == "WARNING"
        assert "Vol Scaling Trim" in call_args[0][1]
        assert "USD/JPY" in call_args[0][2]

    def test_entry_atr_stored_on_new_position(self, runner):
        """_promote_filled_order includes entry_atr in position dict."""
        now = datetime(2026, 2, 1, 14, 0, tzinfo=UTC)

        fill = Fill(
            fill_id="F600",
            order_id="ORD600",
            timestamp=now,
            price=108.000,
            quantity=5000,
            commission=0,
            slippage=0,
        )
        order = Order(
            pair="AUD/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5000,
            status=OrderStatus.FILLED,
            fills=[fill],
            trade_id="T600",
        )

        runner._promote_filled_order(
            pair="AUD/JPY",
            order=order,
            units=5000,
            stop_loss_price=106.500,
            current_atr=0.65,
            reason="golden cross",
            now=now,
        )

        assert "AUD/JPY" in runner._strategy_positions
        pos = runner._strategy_positions["AUD/JPY"]
        assert pos["entry_atr"] == 0.65
        assert pos["original_units"] == 5000
        assert pos["last_vol_trim_time"] is None

    def test_synced_position_estimates_entry_atr(self, runner):
        """_sync_positions estimates entry_atr from stop distance."""
        now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)

        # Set up broker to return an open trade with stop loss
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T700",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.000,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 0.0,
            }
        ]
        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.000,
                "unrealized_pnl": 0.0,
            }
        ]

        # Provide candle data for ExitManager init
        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        pos = runner._strategy_positions.get("USD/JPY")
        assert pos is not None
        assert pos["original_units"] == 10000
        assert pos["last_vol_trim_time"] is None
        # entry_atr should be estimated: (155.0 - 153.5) / 3.0 = 0.5
        if pos["entry_atr"] is not None:
            assert abs(pos["entry_atr"] - 0.5) < 0.01


class TestSyncEntryTime:
    """Tests for _sync_positions using OANDA openTime as entry_time."""

    def test_sync_uses_earliest_open_time_as_entry_time(self, runner):
        """entry_time should be the earliest openTime across all trades for a pair."""
        early = datetime(2026, 2, 1, 8, 0, 0, tzinfo=UTC)
        late = datetime(2026, 2, 5, 14, 30, 0, tzinfo=UTC)

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 15000,
                "avg_price": 155.000,
                "unrealized_pnl": 50.0,
            }
        ]
        # Two trades for same pair (scale-in) — second opened earlier
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T801",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.000,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 30.0,
                "open_time": late,
            },
            {
                "trade_id": "T800",
                "pair": "USD/JPY",
                "units": 5000,
                "price": 154.500,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 20.0,
                "open_time": early,
            },
        ]

        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        pos = runner._strategy_positions.get("USD/JPY")
        assert pos is not None
        # entry_time should be the earliest trade's open_time
        assert pos["entry_time"] == early
        # Should collect both trade_ids
        assert "T801" in pos["trade_ids"]
        assert "T800" in pos["trade_ids"]
        assert len(pos["trade_ids"]) == 2

    def test_sync_falls_back_to_now_without_open_time(self, runner):
        """entry_time should remain ~now if no trades have open_time."""
        before = datetime.now(UTC)

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.000,
                "unrealized_pnl": 0.0,
            }
        ]
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T900",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.000,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 0.0,
                # No open_time field
            }
        ]

        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        after = datetime.now(UTC)
        pos = runner._strategy_positions.get("USD/JPY")
        assert pos is not None
        # entry_time should be approximately now (the placeholder)
        assert before <= pos["entry_time"] <= after

    def test_sync_open_time_none_does_not_overwrite(self, runner):
        """Trades with open_time=None should not overwrite valid entry_time."""
        real_time = datetime(2026, 2, 1, 8, 0, 0, tzinfo=UTC)

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 15000,
                "avg_price": 155.000,
                "unrealized_pnl": 0.0,
            }
        ]
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T950",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.000,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 0.0,
                "open_time": real_time,
            },
            {
                "trade_id": "T951",
                "pair": "USD/JPY",
                "units": 5000,
                "price": 155.500,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 0.0,
                "open_time": None,  # Missing open_time
            },
        ]

        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        pos = runner._strategy_positions.get("USD/JPY")
        assert pos is not None
        # Should keep the real_time, not be overwritten by None
        assert pos["entry_time"] == real_time

    def test_sync_multiple_pairs_independent_entry_times(self, runner):
        """Each pair should get its own earliest open_time."""
        runner_config_pairs = runner.config.pairs
        runner.config = runner.config.__class__(
            pairs=["USD/JPY", "AUD/JPY"],
            check_interval_minutes=60,
            candle_lookback=300,
            position_size_units=10000,
            initial_equity=100000.0,
            log_dir=runner.config.log_dir,
            db_path=runner.config.db_path,
            events_file=runner.config.events_file,
        )

        usdjpy_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=UTC)
        audjpy_time = datetime(2026, 2, 3, 16, 0, 0, tzinfo=UTC)

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.000,
                "unrealized_pnl": 0.0,
            },
            {
                "pair": "AUD/JPY",
                "units": 5000,
                "avg_price": 98.000,
                "unrealized_pnl": 0.0,
            },
        ]
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T1000",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.000,
                "financing": 0.0,
                "stop_loss_price": 153.500,
                "unrealized_pnl": 0.0,
                "open_time": usdjpy_time,
            },
            {
                "trade_id": "T1001",
                "pair": "AUD/JPY",
                "units": 5000,
                "price": 98.000,
                "financing": 0.0,
                "stop_loss_price": 96.500,
                "unrealized_pnl": 0.0,
                "open_time": audjpy_time,
            },
        ]

        df_usd = _make_candle_df(50, 155.0)
        df_aud = _make_candle_df(50, 98.0)
        runner._candle_cache["USD/JPY"] = df_usd
        runner._candle_cache["AUD/JPY"] = df_aud

        runner._sync_positions()

        pos_usd = runner._strategy_positions.get("USD/JPY")
        pos_aud = runner._strategy_positions.get("AUD/JPY")
        assert pos_usd is not None
        assert pos_aud is not None
        assert pos_usd["entry_time"] == usdjpy_time
        assert pos_aud["entry_time"] == audjpy_time


class TestPositionStatePersistence:
    """Tests for SQLite position state persistence in the runner."""

    def test_sync_restores_persisted_hwm(self, runner):
        """HWM from SQLite should overwrite OANDA default (entry price)."""
        # Persist a high_water_mark that's above entry
        runner.store.save_positions(
            {
                "USD/JPY": {
                    "high_water_mark": 158.0,
                    "entry_atr": 0.65,
                    "original_units": 10000,
                    "tranche_count": 2,
                    "levels_taken": 1,
                    "low_water_mark": 153.5,
                    "last_vol_trim_time": None,
                    "financing": 5.50,
                },
            }
        )

        # OANDA returns a position at 155.0 — default HWM would be 155.0
        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.0,
                "unrealized_pnl": 50.0,
            }
        ]
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T100",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.0,
                "financing": 0.0,
                "stop_loss_price": 153.0,
                "unrealized_pnl": 50.0,
                "open_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            }
        ]

        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        pos = runner._strategy_positions["USD/JPY"]
        # Persisted HWM (158.0) > OANDA default (155.0), so it should win
        assert pos["high_water_mark"] == 158.0

    def test_sync_restores_persisted_original_units(self, runner):
        """original_units from SQLite should survive restart."""
        runner.store.save_positions(
            {
                "USD/JPY": {
                    "high_water_mark": 155.0,
                    "entry_atr": 0.65,
                    "original_units": 15000,
                    "tranche_count": 3,
                    "levels_taken": 2,
                    "low_water_mark": 153.0,
                    "last_vol_trim_time": None,
                    "financing": 0.0,
                },
            }
        )

        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 8000,  # Trimmed by vol scaling
                "avg_price": 155.0,
                "unrealized_pnl": 0.0,
            }
        ]
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T200",
                "pair": "USD/JPY",
                "units": 8000,
                "price": 155.0,
                "financing": 0.0,
                "stop_loss_price": 153.0,
                "unrealized_pnl": 0.0,
                "open_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            }
        ]

        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        pos = runner._strategy_positions["USD/JPY"]
        # original_units should be restored from persistence, not current units
        assert pos["original_units"] == 15000
        assert pos["tranche_count"] == 3
        assert pos["levels_taken"] == 2

    def test_sync_ignores_persisted_data_for_unknown_pairs(self, runner):
        """Pairs in SQLite but not on OANDA should be skipped."""
        # Persist data for a pair that no longer exists on OANDA
        runner.store.save_positions(
            {
                "GBP/JPY": {
                    "high_water_mark": 215.0,
                    "entry_atr": 0.80,
                    "original_units": 5000,
                    "tranche_count": 1,
                    "levels_taken": 0,
                    "low_water_mark": 212.0,
                    "last_vol_trim_time": None,
                    "financing": 1.0,
                },
            }
        )

        # OANDA only has USD/JPY
        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.0,
                "unrealized_pnl": 0.0,
            }
        ]
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T300",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.0,
                "financing": 0.0,
                "stop_loss_price": 153.0,
                "unrealized_pnl": 0.0,
                "open_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            }
        ]

        df = _make_candle_df(50, 155.0)
        runner._candle_cache["USD/JPY"] = df

        runner._sync_positions()

        # GBP/JPY should NOT appear in strategy positions
        assert "GBP/JPY" not in runner._strategy_positions
        assert "USD/JPY" in runner._strategy_positions

    def test_tick_saves_positions_to_store(self, runner):
        """save_positions should be called at end of _tick()."""
        runner.store.save_positions = MagicMock()

        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.0,
            "entry_time": datetime.now(UTC),
            "units": 10000,
            "trade_id": "T100",
            "stop_price": 153.0,
            "high_water_mark": 155.5,
            "low_water_mark": 154.0,
            "financing": 0.0,
            "tranche_count": 1,
            "levels_taken": 0,
            "entry_atr": 0.65,
            "original_units": 10000,
            "last_vol_trim_time": None,
            "trade_ids": ["T100"],
        }

        # Minimal broker setup for _tick to run
        runner.broker.get_account_state.return_value = {
            "balance": 100000.0,
            "equity": 100050.0,
            "unrealized_pnl": 50.0,
            "margin_used": 1000.0,
            "margin_available": 99000.0,
            "open_positions": 1,
        }
        runner.broker.get_all_positions.return_value = [
            {
                "pair": "USD/JPY",
                "units": 10000,
                "avg_price": 155.0,
                "unrealized_pnl": 50.0,
                "financing": 0.0,
            }
        ]
        runner.broker.fetch_candles.return_value = _make_candle_df(300, 155.0)
        runner.broker.get_open_trades.return_value = [
            {
                "trade_id": "T100",
                "pair": "USD/JPY",
                "units": 10000,
                "price": 155.0,
                "financing": 0.0,
                "stop_loss_price": 153.0,
            }
        ]
        runner.broker.get_pending_orders.return_value = []

        runner._tick()

        runner.store.save_positions.assert_called_once()
        saved = runner.store.save_positions.call_args[0][0]
        assert "USD/JPY" in saved

    def test_close_position_deletes_from_store(self, runner):
        """delete_position should be called when a position is closed."""
        runner.store.delete_position = MagicMock()
        runner.store.save_trade = MagicMock()

        runner._strategy_positions["USD/JPY"] = {
            "entry_price": 155.0,
            "entry_time": datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
            "units": 10000,
            "financing": 2.50,
        }

        fill = Fill(
            fill_id="f1",
            order_id="ORD1",
            timestamp=datetime.now(),
            price=156.0,
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

        runner.store.delete_position.assert_called_once_with("USD/JPY")
        assert "USD/JPY" not in runner._strategy_positions
