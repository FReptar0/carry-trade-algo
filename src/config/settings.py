"""Centralized configuration for the carry trade algorithm."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data paths and parameters."""

    synthetic_data_path: str = "data/synthetic"
    cache_path: str = "data/cache"
    lookback_days: int = 365


@dataclass
class PairConfig:
    """Configuration for a single forex pair."""

    name: str
    price_min: float
    price_max: float
    volatility: float
    trend: str  # "sideways", "uptrend", "downtrend"
    interest_rate_base: float
    interest_rate_quote: float


@dataclass
class StrategyConfig:
    """Configuration for carry trade strategy parameters."""

    name: str = "carry_trade_v1"
    pairs: list[str] = field(default_factory=lambda: ["USD/JPY", "AUD/JPY", "EUR/USD"])
    min_swap_threshold: float = 0.005
    trend_filter_period: int = 50
    rsi_threshold: int = 70
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.06
    position_size_pct: float = 0.10


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    initial_capital: float = 10_000.0
    max_position_size: float = 0.10
    stop_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.20
    max_daily_loss_pct: float = 0.03
    max_weekly_loss_pct: float = 0.07
    max_positions: int = 3
    max_exposure_per_pair: float = 0.15


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""

    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    commission_per_lot: float = 5.0
    slippage_pips: float = 1.0


# Default pair configurations for synthetic data generation
DEFAULT_PAIRS: list[PairConfig] = [
    PairConfig(
        name="USD/JPY",
        price_min=145.0,
        price_max=152.0,
        volatility=0.008,
        trend="sideways",
        interest_rate_base=0.0525,
        interest_rate_quote=0.0025,
    ),
    PairConfig(
        name="AUD/JPY",
        price_min=95.0,
        price_max=102.0,
        volatility=0.012,
        trend="uptrend",
        interest_rate_base=0.0435,
        interest_rate_quote=0.0025,
    ),
    PairConfig(
        name="EUR/USD",
        price_min=1.06,
        price_max=1.12,
        volatility=0.007,
        trend="sideways",
        interest_rate_base=0.045,
        interest_rate_quote=0.0525,
    ),
]
