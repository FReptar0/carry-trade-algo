"""Scale-in/scale-out position management.

Instead of all-in/all-out, positions are built and reduced in
tranches. Adds to winners and takes partial profits at ATR-based
levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScaleLevel:
    """A profit level for partial exit."""

    atr_multiple: float  # Take profit at this many ATRs
    exit_fraction: float  # Fraction of remaining position to exit


class ScaleManager:
    """Manages position scaling in tranches.

    Args:
        max_tranches: Maximum number of position additions.
        tranche_pct: Fraction of full size per tranche.
        scale_in_atr: ATR multiples of favorable move to add.
        profit_levels: ATR levels for partial profit taking.

    Example:
        >>> mgr = ScaleManager()
        >>> if mgr.should_add(position, current_price, atr):
        ...     add_units(mgr.tranche_units(total_units))
        >>> if mgr.should_reduce(position, current_price, atr):
        ...     reduce_units(mgr.tranche_units(total_units))
    """

    DEFAULT_PROFIT_LEVELS = [
        ScaleLevel(atr_multiple=2.0, exit_fraction=0.33),
        ScaleLevel(atr_multiple=4.0, exit_fraction=0.33),
    ]

    def __init__(
        self,
        max_tranches: int = 3,
        tranche_pct: float = 0.33,
        scale_in_atr: float = 1.0,
        profit_levels: list[ScaleLevel] | None = None,
    ) -> None:
        self.max_tranches = max_tranches
        self.tranche_pct = tranche_pct
        self.scale_in_atr = scale_in_atr
        self.profit_levels = profit_levels or self.DEFAULT_PROFIT_LEVELS

    def should_add(
        self,
        position: dict,
        current_price: float,
        atr: float,
    ) -> bool:
        """Check if we should add to an existing position.

        Adds if price has moved favorably by scale_in_atr * ATR
        and we haven't exceeded max tranches.

        Args:
            position: Position dict with entry_price, units,
                and optional tranche_count.
            current_price: Current market price.
            atr: Current ATR value.

        Returns:
            True if should add a tranche.
        """
        entry_price = position.get("entry_price", 0)
        tranche_count = position.get("tranche_count", 1)

        if tranche_count >= self.max_tranches:
            return False

        if atr <= 0 or entry_price <= 0:
            return False

        # Price must have moved favorably by scale_in_atr * ATR
        favorable_move = current_price - entry_price
        threshold = self.scale_in_atr * atr

        if favorable_move >= threshold:
            logger.debug(
                "Scale-in: move=%.4f >= threshold=%.4f, "
                "tranche %d/%d",
                favorable_move,
                threshold,
                tranche_count + 1,
                self.max_tranches,
            )
            return True

        return False

    def should_reduce(
        self,
        position: dict,
        current_price: float,
        atr: float,
    ) -> bool:
        """Check if we should take partial profits.

        Args:
            position: Position dict with entry_price, units,
                and optional levels_taken.
            current_price: Current market price.
            atr: Current ATR value.

        Returns:
            True if should reduce position.
        """
        entry_price = position.get("entry_price", 0)
        levels_taken = position.get("levels_taken", 0)

        if levels_taken >= len(self.profit_levels):
            return False

        if atr <= 0 or entry_price <= 0:
            return False

        next_level = self.profit_levels[levels_taken]
        target_move = next_level.atr_multiple * atr

        actual_move = current_price - entry_price
        if actual_move >= target_move:
            logger.debug(
                "Scale-out: profit=%.4f >= level %.1f ATR "
                "(%.4f), exit %.0f%%",
                actual_move,
                next_level.atr_multiple,
                target_move,
                next_level.exit_fraction * 100,
            )
            return True

        return False

    def tranche_units(self, total_units: int) -> int:
        """Calculate units for one tranche.

        Args:
            total_units: Full target position size.

        Returns:
            Units for one tranche.
        """
        return max(1, int(total_units * self.tranche_pct))

    def reduction_units(
        self, current_units: int, level_index: int
    ) -> int:
        """Calculate units to reduce at a profit level.

        Args:
            current_units: Current position size.
            level_index: Which profit level (0, 1, ...).

        Returns:
            Units to exit.
        """
        if level_index >= len(self.profit_levels):
            return 0

        fraction = self.profit_levels[level_index].exit_fraction
        return max(1, int(current_units * fraction))
