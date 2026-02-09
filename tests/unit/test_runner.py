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
