"""Shadow mode evaluator for the contextual bandit.

During shadow mode, the bandit logs what it would have done but
doesn't block signals. After a configurable number of trades,
if the bandit's simulated decisions beat random, it activates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShadowRecord:
    """A single shadow mode observation."""

    bandit_action: str  # "TAKE" or "SKIP"
    confidence: float
    actual_pnl: float


class ShadowEvaluator:
    """Evaluates bandit decisions in shadow mode before activation.

    Args:
        min_trades: Minimum trades before considering activation.
        min_advantage: Minimum advantage over random to activate.

    Example:
        >>> shadow = ShadowEvaluator(min_trades=50)
        >>> shadow.record(features, "TAKE", pnl=100)
        >>> should_go_live, stats = shadow.should_activate()
    """

    def __init__(
        self,
        min_trades: int = 50,
        min_advantage: float = 0.1,
    ) -> None:
        self.min_trades = min_trades
        self.min_advantage = min_advantage
        self._records: list[ShadowRecord] = []

    def record(
        self,
        features: np.ndarray,
        bandit_decision: str,
        actual_pnl: float,
    ) -> None:
        """Record a shadow observation.

        Args:
            features: Context features (not stored, for future use).
            bandit_decision: What the bandit would have chosen.
            actual_pnl: Actual P&L of the trade.
        """
        self._records.append(
            ShadowRecord(
                bandit_action=bandit_decision,
                confidence=0.0,
                actual_pnl=actual_pnl,
            )
        )

    def should_activate(self) -> tuple[bool, dict]:
        """Check if the bandit should go live.

        Returns:
            Tuple of (should_activate, stats_dict).
        """
        n = len(self._records)
        if n < self.min_trades:
            return False, {
                "reason": "insufficient_trades",
                "trades": n,
                "required": self.min_trades,
            }

        # Calculate simulated performance
        take_pnls = [
            r.actual_pnl
            for r in self._records
            if r.bandit_action == "TAKE"
        ]
        skip_pnls = [
            r.actual_pnl
            for r in self._records
            if r.bandit_action == "SKIP"
        ]
        all_pnls = [r.actual_pnl for r in self._records]

        # Bandit's simulated return: take good trades, skip bad ones
        bandit_pnl = sum(take_pnls)
        random_pnl = sum(all_pnls)

        # Advantage: bandit should do better than taking all signals
        if random_pnl != 0:
            advantage = (bandit_pnl - random_pnl) / abs(random_pnl)
        elif bandit_pnl > 0:
            advantage = 1.0
        else:
            advantage = 0.0

        # Accuracy: did bandit correctly identify winners vs losers?
        correct = 0
        for r in self._records:
            if r.bandit_action == "TAKE" and r.actual_pnl > 0:
                correct += 1
            elif r.bandit_action == "SKIP" and r.actual_pnl <= 0:
                correct += 1
        accuracy = correct / n if n > 0 else 0.0

        stats = {
            "trades": n,
            "bandit_pnl": bandit_pnl,
            "random_pnl": random_pnl,
            "advantage": advantage,
            "accuracy": accuracy,
            "take_count": len(take_pnls),
            "skip_count": len(skip_pnls),
            "avg_take_pnl": np.mean(take_pnls) if take_pnls else 0.0,
            "avg_skip_pnl": np.mean(skip_pnls) if skip_pnls else 0.0,
        }

        should_activate = advantage >= self.min_advantage
        if should_activate:
            logger.info(
                "Shadow evaluation: ACTIVATE (advantage=%.2f, "
                "accuracy=%.2f)",
                advantage,
                accuracy,
            )
        else:
            logger.info(
                "Shadow evaluation: STAY SHADOW (advantage=%.2f < "
                "%.2f required)",
                advantage,
                self.min_advantage,
            )

        return should_activate, stats

    @property
    def trade_count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        """Clear all shadow records."""
        self._records.clear()
