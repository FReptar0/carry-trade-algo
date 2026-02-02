"""Adaptive regime adapter backed by ParamStore.

Drop-in replacement for RegimeAdapter that reads optimized parameters
from the ParamStore instead of hardcoded defaults. Falls back to
hardcoded defaults if no stored parameters exist.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.adaptive.param_store import ParamStore
from src.regime.adapters import RegimeAdapter, RegimeAdaptedParams
from src.regime.detector import CompositeRegime

logger = logging.getLogger(__name__)


class AdaptiveRegimeAdapter(RegimeAdapter):
    """Regime adapter that reads from ParamStore first.

    Args:
        param_store: ParamStore with optimized parameters.

    Example:
        >>> adapter = AdaptiveRegimeAdapter(param_store)
        >>> params = adapter.adapt(CompositeRegime.TREND_LOW_VOL)
    """

    def __init__(self, param_store: ParamStore) -> None:
        super().__init__()
        self.param_store = param_store

    def adapt(self, regime: CompositeRegime) -> RegimeAdaptedParams:
        """Get adapted parameters, preferring stored over defaults.

        Args:
            regime: Current composite regime.

        Returns:
            RegimeAdaptedParams from store or hardcoded defaults.
        """
        regime_str = regime.value if hasattr(regime, "value") else str(regime)

        stored = self.param_store.get_active_params(regime_str)
        if stored is not None:
            logger.debug(
                "Using stored params for %s (id=%d, sharpe=%.3f)",
                regime_str,
                stored.id,
                stored.sharpe or 0.0,
            )
            return stored.params

        logger.debug(
            "No stored params for %s, using defaults", regime_str
        )
        return super().adapt(regime)
