"""Base strategy class that all trading strategies must follow.

Think of this as a contract: any strategy you build must implement
these methods. This makes it easy to swap strategies in and out
of the backtesting engine without changing other code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Signal(Enum):
    """What the strategy wants to do at each bar.

    LONG  = buy (bet the price goes UP)
    SHORT = sell (bet the price goes DOWN)
    CLOSE = exit any open position
    HOLD  = do nothing, keep current state
    """

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """A trading decision at a specific point in time.

    Attributes:
        timestamp: When this signal was generated.
        signal: What to do (LONG, SHORT, CLOSE, HOLD).
        strength: How confident the strategy is (0.0 to 1.0).
            Used later for position sizing -- stronger signal = bigger bet.
        reason: Human-readable explanation for logging.
    """

    timestamp: pd.Timestamp
    signal: Signal
    strength: float = 1.0
    reason: str = ""


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    To create a new strategy:
    1. Inherit from this class
    2. Implement generate_signals()
    3. Optionally override validate_params()

    The backtesting engine calls generate_signals() on your data,
    gets back a list of TradeSignals, and executes them.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Analyze price data and return trading signals.

        Args:
            data: DataFrame with columns: timestamp, open, high, low,
                close, volume, swap_long, swap_short.

        Returns:
            List of TradeSignal objects, one per bar where a decision is made.
            Bars with no action can be omitted or returned as HOLD.
        """
        ...

    def validate_params(self) -> bool:
        """Check that strategy parameters are within valid ranges.

        Returns:
            True if all parameters are valid.
        """
        return True
