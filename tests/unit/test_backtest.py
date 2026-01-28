"""Tests for backtest engine, portfolio, and metrics."""

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.portfolio import Portfolio
from src.config.settings import DEFAULT_PAIRS
from src.data.generator import SyntheticDataGenerator
from src.strategy.carry_trade import CarryTradeStrategy


def _get_data(pair_idx: int = 0, days: int = 365) -> pd.DataFrame:
    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_pair(DEFAULT_PAIRS[pair_idx], days=days)


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(initial_capital=10_000)
        assert p.cash == 10_000
        assert p.position is None
        assert len(p.trades) == 0

    def test_open_position(self):
        p = Portfolio(initial_capital=10_000, position_size_pct=0.10)
        ts = pd.Timestamp("2024-01-01")
        result = p.open_position("USD/JPY", 148.0, ts, "long")
        assert result is True
        assert p.position is not None
        assert p.cash < 10_000  # Used some capital

    def test_cannot_open_twice(self):
        p = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01")
        p.open_position("USD/JPY", 148.0, ts)
        result = p.open_position("AUD/JPY", 98.0, ts)
        assert result is False

    def test_close_position_records_trade(self):
        p = Portfolio(initial_capital=10_000, position_size_pct=0.10)
        ts1 = pd.Timestamp("2024-01-01")
        ts2 = pd.Timestamp("2024-01-10")
        p.open_position("USD/JPY", 148.0, ts1)
        trade = p.close_position(150.0, ts2, "take profit")
        assert trade is not None
        assert trade.pnl > 0  # Price went up on a long
        assert len(p.trades) == 1
        assert p.position is None

    def test_accrue_swap(self):
        p = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01")
        p.open_position("USD/JPY", 148.0, ts)
        p.accrue_swap(0.0137)
        assert p.position.swap_accumulated > 0

    def test_equity_history(self):
        p = Portfolio(initial_capital=10_000)
        ts = pd.Timestamp("2024-01-01")
        p.record_equity(ts, 148.0)
        df = p.get_equity_df()
        assert len(df) == 1
        assert df["equity"].iloc[0] == 10_000

    def test_trades_df_empty(self):
        p = Portfolio()
        df = p.get_trades_df()
        assert len(df) == 0


class TestBacktestEngine:
    def test_runs_without_error(self):
        data = _get_data(pair_idx=0, days=200)
        strategy = CarryTradeStrategy()
        engine = BacktestEngine(strategy)
        result = engine.run(data, "USD/JPY")
        assert result is not None
        assert result.pair == "USD/JPY"

    def test_equity_df_not_empty(self):
        data = _get_data(pair_idx=0, days=200)
        engine = BacktestEngine(CarryTradeStrategy())
        result = engine.run(data, "USD/JPY")
        assert len(result.equity_df) > 0

    def test_metrics_computed(self):
        data = _get_data(pair_idx=1, days=365)  # AUD/JPY for more trades
        engine = BacktestEngine(CarryTradeStrategy())
        result = engine.run(data, "AUD/JPY")
        m = result.metrics
        assert m.total_trades >= 0
        assert -1.0 <= m.total_return <= 10.0
        assert 0 <= m.max_drawdown <= 1.0

    def test_summary_string(self):
        data = _get_data(pair_idx=0, days=200)
        engine = BacktestEngine(CarryTradeStrategy())
        result = engine.run(data, "USD/JPY")
        s = result.summary()
        assert "USD/JPY" in s
        assert "Total Return" in s

    def test_no_trades_on_negative_carry(self):
        data = _get_data(pair_idx=2, days=200)  # EUR/USD
        engine = BacktestEngine(CarryTradeStrategy())
        result = engine.run(data, "EUR/USD")
        assert result.metrics.total_trades == 0

    def test_execution_time(self):
        """1 year backtest should complete in under 5 seconds."""
        import time
        data = _get_data(pair_idx=1, days=365)
        engine = BacktestEngine(CarryTradeStrategy())
        start = time.time()
        engine.run(data, "AUD/JPY")
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Backtest took {elapsed:.1f}s, limit is 5s"
