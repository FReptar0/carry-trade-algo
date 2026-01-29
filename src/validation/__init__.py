"""Validation module for paper trading and backtest comparison.

This module provides tools to validate trading strategies before
real money deployment by comparing backtest results to paper trading.
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

__all__ = [
    "PaperTradingValidator",
    "ValidationMetrics",
    "ValidationCriteria",
    "TradingProtocol",
    "ProtocolDay",
    "ProtocolCheckpoint",
    "ProtocolStatus",
]
