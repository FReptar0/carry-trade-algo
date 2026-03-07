"""Validation module for paper trading, backtest comparison, and robustness testing.

This module provides tools to validate trading strategies before
real money deployment by comparing backtest results to paper trading
and running statistical robustness tests.
"""

from src.validation.validator import (
    PaperTradingValidator,
    ValidationMetrics,
    ValidationCriteria,
)
from src.validation.protocol import (
    TradingProtocol,
    ProtocolDay,
    ProtocolCheckpoint,
    ProtocolStatus,
)
from src.validation.robustness import (
    DataSource,
    ValidationSuiteResult,
    load_from_backtest,
    load_from_multi_pair_backtest,
    load_from_live_db,
    run_validation_suite,
)

__all__ = [
    "PaperTradingValidator",
    "ValidationMetrics",
    "ValidationCriteria",
    "TradingProtocol",
    "ProtocolDay",
    "ProtocolCheckpoint",
    "ProtocolStatus",
    "DataSource",
    "ValidationSuiteResult",
    "load_from_backtest",
    "load_from_multi_pair_backtest",
    "load_from_live_db",
    "run_validation_suite",
]
