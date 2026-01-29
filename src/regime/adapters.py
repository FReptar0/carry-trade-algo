"""Regime-based parameter adaptation for trading strategies.

This module provides the bridge between regime detection and strategy execution.
When the market regime changes, the strategy should adapt its parameters:

TREND_LOW_VOL (Best regime):
- Tighter stops (trend is clear, vol is low)
- Larger positions (high confidence)
- Relaxed entry (don't over-filter)

TREND_HIGH_VOL (Risky but profitable):
- Wider stops (need room for volatility)
- Smaller positions (manage risk)
- Moderate entry requirements

RANGE_LOW_VOL (Collect carry):
- Medium stops
- Very small positions (just hold for swap)
- Very strict entry (almost no new trades)

RANGE_HIGH_VOL (Avoid):
- Do not trade
- Close existing positions on weakness
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.regime.detector import CompositeRegime, RegimeState, TrendRegime


@dataclass
class RegimeAdaptedParams:
    """Strategy parameters adjusted for current regime.

    These multipliers are applied to base strategy parameters:
    - stop_loss_mult: Multiplier for ATR-based stop distance
    - position_size_mult: Multiplier for position size
    - entry_threshold: Strictness of entry (higher = fewer trades)
    - should_trade: Whether to allow new entries
    - should_close_on_weakness: Whether to close on any weakness signal
    """

    stop_loss_mult: float
    position_size_mult: float
    entry_threshold: float  # 0.0 = relaxed, 1.0 = strict
    should_trade: bool
    should_close_on_weakness: bool = False

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.stop_loss_mult <= 0:
            raise ValueError("stop_loss_mult must be positive")
        if self.position_size_mult < 0 or self.position_size_mult > 2.0:
            raise ValueError("position_size_mult must be between 0 and 2.0")
        if self.entry_threshold < 0 or self.entry_threshold > 1.0:
            raise ValueError("entry_threshold must be between 0 and 1.0")


class RegimeAdapter:
    """Adapts strategy parameters based on market regime.

    Example:
        adapter = RegimeAdapter()
        params = adapter.adapt(regime.composite_regime)

        if not params.should_trade:
            return None  # Skip entry

        # Apply multipliers
        adjusted_stop = base_stop * params.stop_loss_mult
        adjusted_size = base_size * params.position_size_mult
    """

    # Default regime -> parameter mappings
    # These are starting points; tune based on backtest results
    DEFAULT_PARAMS: dict[CompositeRegime, RegimeAdaptedParams] = {
        CompositeRegime.TREND_LOW_VOL: RegimeAdaptedParams(
            stop_loss_mult=0.8,      # Tighter stops (trend is clear)
            position_size_mult=1.2,  # Larger size (high confidence)
            entry_threshold=0.5,     # Relaxed entry
            should_trade=True,
            should_close_on_weakness=False,
        ),
        CompositeRegime.TREND_HIGH_VOL: RegimeAdaptedParams(
            stop_loss_mult=1.5,      # Wider stops for volatility
            position_size_mult=0.7,  # Smaller size (manage risk)
            entry_threshold=0.7,     # Moderate entry requirements
            should_trade=True,
            should_close_on_weakness=False,
        ),
        CompositeRegime.RANGE_LOW_VOL: RegimeAdaptedParams(
            stop_loss_mult=1.0,      # Normal stops
            position_size_mult=0.3,  # Very small (just for carry)
            entry_threshold=0.95,    # Very strict (almost no trades)
            should_trade=False,      # Prefer to avoid new entries
            should_close_on_weakness=False,
        ),
        CompositeRegime.RANGE_HIGH_VOL: RegimeAdaptedParams(
            stop_loss_mult=2.0,      # Very wide if forced to trade
            position_size_mult=0.0,  # Zero size = no trading
            entry_threshold=1.0,     # Never enters
            should_trade=False,
            should_close_on_weakness=True,  # Exit on any weakness
        ),
    }

    def __init__(
        self,
        custom_params: Optional[dict[CompositeRegime, RegimeAdaptedParams]] = None,
    ) -> None:
        """Initialize adapter with optional custom parameters.

        Args:
            custom_params: Override default regime parameters.
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if custom_params:
            self.params.update(custom_params)

    def adapt(self, regime: CompositeRegime) -> RegimeAdaptedParams:
        """Get adapted parameters for given regime.

        Args:
            regime: Current composite regime.

        Returns:
            RegimeAdaptedParams for this regime.
        """
        return self.params[regime]

    def adapt_from_state(self, state: RegimeState) -> RegimeAdaptedParams:
        """Get adapted parameters from full regime state.

        This method allows for more nuanced adaptation based on
        confidence levels and specific indicator values.

        Args:
            state: Full regime state with confidence and raw values.

        Returns:
            RegimeAdaptedParams, potentially adjusted for confidence.
        """
        base_params = self.adapt(state.composite_regime)

        # If confidence is low, be more conservative
        if state.confidence < 0.3:
            return RegimeAdaptedParams(
                stop_loss_mult=base_params.stop_loss_mult * 1.2,  # Wider stops
                position_size_mult=base_params.position_size_mult * 0.5,  # Smaller
                entry_threshold=min(base_params.entry_threshold + 0.2, 1.0),  # Stricter
                should_trade=base_params.should_trade,
                should_close_on_weakness=base_params.should_close_on_weakness,
            )

        return base_params

    def should_enter_long(
        self, state: RegimeState, signal_strength: float = 0.5
    ) -> bool:
        """Check if long entry is allowed given regime and signal strength.

        Args:
            state: Current regime state.
            signal_strength: Entry signal strength (0-1).

        Returns:
            True if entry is allowed.
        """
        params = self.adapt_from_state(state)

        # Check if trading is allowed
        if not params.should_trade:
            return False

        # Check if trend direction supports long
        if state.trend_direction == "down":
            return False

        # Check if signal strength meets threshold
        if signal_strength < params.entry_threshold:
            return False

        return True

    def should_exit_early(self, state: RegimeState, current_pnl_pct: float) -> bool:
        """Check if position should exit early due to regime change.

        Args:
            state: Current regime state.
            current_pnl_pct: Current position P&L as percentage.

        Returns:
            True if should exit early.
        """
        params = self.adapt_from_state(state)

        # Definitely exit if regime says so and position is weak
        if params.should_close_on_weakness and current_pnl_pct < 0.01:
            return True

        # Exit winning positions if regime turns bad
        if state.composite_regime == CompositeRegime.RANGE_HIGH_VOL:
            if current_pnl_pct > 0.02:  # Lock in 2%+ gains
                return True

        return False

    def get_adjusted_stop(
        self, base_atr_mult: float, state: RegimeState
    ) -> float:
        """Calculate adjusted stop loss multiplier.

        Args:
            base_atr_mult: Base ATR multiplier (e.g., 2.0).
            state: Current regime state.

        Returns:
            Adjusted ATR multiplier for stops.
        """
        params = self.adapt_from_state(state)
        return base_atr_mult * params.stop_loss_mult

    def get_adjusted_size(
        self, base_size_pct: float, state: RegimeState
    ) -> float:
        """Calculate adjusted position size.

        Args:
            base_size_pct: Base position size as percentage (e.g., 0.10).
            state: Current regime state.

        Returns:
            Adjusted position size percentage.
        """
        params = self.adapt_from_state(state)
        return base_size_pct * params.position_size_mult


def get_regime_description(regime: CompositeRegime) -> str:
    """Get human-readable description of a regime.

    Useful for logging and debugging.
    """
    descriptions = {
        CompositeRegime.TREND_LOW_VOL: (
            "TREND + LOW VOL: Best regime for carry trades. "
            "Clear trend direction with predictable price action. "
            "Trade with confidence."
        ),
        CompositeRegime.TREND_HIGH_VOL: (
            "TREND + HIGH VOL: Profitable but risky. "
            "Strong trend but large swings. "
            "Use wider stops, reduce size."
        ),
        CompositeRegime.RANGE_LOW_VOL: (
            "RANGE + LOW VOL: Sideways, calm market. "
            "Hold existing positions for swap income. "
            "Avoid new entries."
        ),
        CompositeRegime.RANGE_HIGH_VOL: (
            "RANGE + HIGH VOL: Worst regime. "
            "No trend + big swings = whipsaw. "
            "Do not trade."
        ),
    }
    return descriptions.get(regime, "Unknown regime")
