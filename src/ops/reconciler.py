"""Position reconciler: syncs internal state with broker positions.

Compares the runner's internal `_strategy_positions` dict with
OANDA's actual positions on every tick. On mismatch, logs warnings
and updates internal state to match broker (broker is source of truth).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReconcileResult:
    """Result of a position reconciliation check.

    Attributes:
        matched: Pairs that match between internal and broker.
        orphaned_broker: Pairs on broker but not tracked internally.
        orphaned_internal: Pairs tracked internally but not on broker.
        mismatched_units: Pairs where unit counts differ.
        is_clean: True if no discrepancies found.
    """

    matched: list[str] = field(default_factory=list)
    orphaned_broker: list[str] = field(default_factory=list)
    orphaned_internal: list[str] = field(default_factory=list)
    mismatched_units: list[dict] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            not self.orphaned_broker
            and not self.orphaned_internal
            and not self.mismatched_units
        )


class Reconciler:
    """Compares internal position tracking with broker state.

    Args:
        alert_manager: Optional AlertManager for mismatch alerts.

    Example:
        >>> rec = Reconciler()
        >>> result = rec.reconcile(internal_positions, broker_positions)
        >>> if not result.is_clean:
        ...     print("Mismatch detected!")
    """

    def __init__(
        self, alert_manager: Optional[object] = None
    ) -> None:
        self.alert_manager = alert_manager

    def reconcile(
        self,
        internal: dict[str, dict],
        broker_positions: list[dict],
        managed_pairs: Optional[list[str]] = None,
    ) -> ReconcileResult:
        """Compare internal positions with broker positions.

        Args:
            internal: Dict of {pair: {entry_price, units, ...}}.
            broker_positions: List of dicts from broker.get_all_positions().
            managed_pairs: Optional list of pairs this system manages.
                Only reconcile these pairs.

        Returns:
            ReconcileResult describing any discrepancies.
        """
        result = ReconcileResult()
        broker_by_pair: dict[str, dict] = {}

        for pos in broker_positions:
            pair = pos["pair"]
            if managed_pairs and pair not in managed_pairs:
                continue
            broker_by_pair[pair] = pos

        internal_pairs = set(internal.keys())
        broker_pairs = set(broker_by_pair.keys())

        # Matched pairs
        for pair in internal_pairs & broker_pairs:
            internal_units = internal[pair].get("units", 0)
            broker_units = broker_by_pair[pair].get("units", 0)

            if internal_units != broker_units:
                result.mismatched_units.append(
                    {
                        "pair": pair,
                        "internal_units": internal_units,
                        "broker_units": broker_units,
                    }
                )
                logger.warning(
                    "Unit mismatch for %s: internal=%d broker=%d",
                    pair,
                    internal_units,
                    broker_units,
                )
            else:
                result.matched.append(pair)

        # Orphaned on broker (not tracked internally)
        for pair in broker_pairs - internal_pairs:
            result.orphaned_broker.append(pair)
            logger.warning(
                "Orphaned broker position: %s (%d units) — "
                "not tracked internally",
                pair,
                broker_by_pair[pair].get("units", 0),
            )

        # Orphaned internally (not on broker)
        for pair in internal_pairs - broker_pairs:
            result.orphaned_internal.append(pair)
            logger.warning(
                "Orphaned internal position: %s — "
                "not found on broker",
                pair,
            )

        if not result.is_clean and self.alert_manager is not None:
            self._send_alert(result)

        return result

    def apply_fixes(
        self,
        internal: dict[str, dict],
        result: ReconcileResult,
        broker_positions: list[dict],
    ) -> dict[str, dict]:
        """Apply reconciliation fixes (broker is truth).

        Args:
            internal: Mutable internal positions dict.
            result: ReconcileResult from reconcile().
            broker_positions: Broker positions for data.

        Returns:
            Updated internal positions dict.
        """
        broker_by_pair = {p["pair"]: p for p in broker_positions}

        # Remove orphaned internal positions
        for pair in result.orphaned_internal:
            logger.info("Removing orphaned internal position: %s", pair)
            internal.pop(pair, None)

        # Add orphaned broker positions
        for pair in result.orphaned_broker:
            pos = broker_by_pair.get(pair)
            if pos:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                internal[pair] = {
                    "entry_price": pos["avg_price"],
                    "entry_time": datetime.now(ZoneInfo("UTC")),
                    "units": pos["units"],
                }
                logger.info(
                    "Added orphaned broker position: %s (%d units)",
                    pair,
                    pos["units"],
                )

        # Fix unit mismatches (broker wins)
        for mismatch in result.mismatched_units:
            pair = mismatch["pair"]
            broker_units = mismatch["broker_units"]
            if pair in internal:
                internal[pair]["units"] = broker_units
                logger.info(
                    "Updated units for %s: %d -> %d",
                    pair,
                    mismatch["internal_units"],
                    broker_units,
                )

        return internal

    def _send_alert(self, result: ReconcileResult) -> None:
        """Send alert about reconciliation mismatches."""
        parts = []
        if result.orphaned_broker:
            parts.append(
                f"Orphaned on broker: {result.orphaned_broker}"
            )
        if result.orphaned_internal:
            parts.append(
                f"Orphaned internal: {result.orphaned_internal}"
            )
        if result.mismatched_units:
            pairs = [m["pair"] for m in result.mismatched_units]
            parts.append(f"Unit mismatches: {pairs}")

        try:
            self.alert_manager.send(
                "HIGH",
                "Position Reconciliation Mismatch",
                "\n".join(parts),
            )
        except Exception as e:
            logger.error("Failed to send reconciliation alert: %s", e)
