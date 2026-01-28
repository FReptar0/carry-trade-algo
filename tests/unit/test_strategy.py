"""Tests for carry trade strategy."""

import pandas as pd

from src.config.settings import StrategyConfig
from src.data.generator import SyntheticDataGenerator
from src.strategy.base import Signal
from src.strategy.carry_trade import CarryTradeStrategy


def _generate_data(days: int = 100, seed: int = 42) -> pd.DataFrame:
    from src.config.settings import DEFAULT_PAIRS
    gen = SyntheticDataGenerator(seed=seed)
    return gen.generate_pair(DEFAULT_PAIRS[0], days=days)  # USD/JPY


class TestCarryTradeStrategy:
    def test_generates_signals(self):
        data = _generate_data(days=200)
        strategy = CarryTradeStrategy()
        signals = strategy.generate_signals(data)
        assert isinstance(signals, list)

    def test_signals_are_long_or_close(self):
        data = _generate_data(days=200)
        strategy = CarryTradeStrategy()
        signals = strategy.generate_signals(data)
        for s in signals:
            assert s.signal in (Signal.LONG, Signal.CLOSE)

    def test_signals_alternate_long_close(self):
        """Can't have two LONGs in a row without a CLOSE between them."""
        data = _generate_data(days=365)
        strategy = CarryTradeStrategy()
        signals = strategy.generate_signals(data)
        if len(signals) < 2:
            return
        for i in range(1, len(signals)):
            if signals[i].signal == Signal.LONG:
                assert signals[i - 1].signal == Signal.CLOSE

    def test_no_signals_with_negative_carry(self):
        """EUR/USD has negative carry -- should produce no LONG signals."""
        from src.config.settings import DEFAULT_PAIRS
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate_pair(DEFAULT_PAIRS[2], days=200)  # EUR/USD
        strategy = CarryTradeStrategy()
        signals = strategy.generate_signals(data)
        longs = [s for s in signals if s.signal == Signal.LONG]
        assert len(longs) == 0

    def test_validate_params_default(self):
        strategy = CarryTradeStrategy()
        assert strategy.validate_params()

    def test_validate_params_bad_stop_loss(self):
        config = StrategyConfig(stop_loss_pct=0.10, take_profit_pct=0.05)
        strategy = CarryTradeStrategy(config)
        assert not strategy.validate_params()  # stop > take profit

    def test_signal_has_reason(self):
        data = _generate_data(days=365)
        strategy = CarryTradeStrategy()
        signals = strategy.generate_signals(data)
        for s in signals:
            assert s.reason != ""

    def test_high_threshold_fewer_signals(self):
        data = _generate_data(days=365)
        low_thresh = CarryTradeStrategy(StrategyConfig(min_swap_threshold=0.001))
        high_thresh = CarryTradeStrategy(StrategyConfig(min_swap_threshold=0.02))
        s_low = low_thresh.generate_signals(data)
        s_high = high_thresh.generate_signals(data)
        assert len(s_low) >= len(s_high)
