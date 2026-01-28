"""Tests for configuration settings."""

from src.config.settings import (
    BacktestConfig,
    DataConfig,
    DEFAULT_PAIRS,
    PairConfig,
    RiskConfig,
    StrategyConfig,
)


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.synthetic_data_path == "data/synthetic"
    assert cfg.lookback_days == 365


def test_strategy_config_defaults():
    cfg = StrategyConfig()
    assert cfg.name == "carry_trade_v1"
    assert len(cfg.pairs) == 3
    assert cfg.min_swap_threshold == 0.005
    assert cfg.stop_loss_pct == 0.02


def test_risk_config_defaults():
    cfg = RiskConfig()
    assert cfg.initial_capital == 10_000.0
    assert cfg.max_drawdown_pct == 0.20
    assert cfg.max_positions == 3


def test_backtest_config_defaults():
    cfg = BacktestConfig()
    assert cfg.slippage_pips == 1.0
    assert cfg.commission_per_lot == 5.0


def test_pair_config_creation():
    pair = PairConfig(
        name="TEST/USD",
        price_min=1.0,
        price_max=2.0,
        volatility=0.01,
        trend="sideways",
        interest_rate_base=0.05,
        interest_rate_quote=0.01,
    )
    assert pair.name == "TEST/USD"
    assert pair.interest_rate_base > pair.interest_rate_quote


def test_default_pairs_has_three():
    assert len(DEFAULT_PAIRS) == 3
    names = [p.name for p in DEFAULT_PAIRS]
    assert "USD/JPY" in names
    assert "AUD/JPY" in names
    assert "EUR/USD" in names
