"""Cross-pair correlation monitor for portfolio risk management.

Tracks rolling correlation between open position pairs and provides
a correlation factor to reduce position sizes when pairs are highly
correlated (concentrated risk).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationMonitor:
    """Monitors rolling correlation between currency pairs.

    Args:
        lookback: Number of periods for rolling correlation.

    Example:
        >>> monitor = CorrelationMonitor(lookback=60)
        >>> monitor.update({"USD/JPY": df1, "AUD/JPY": df2})
        >>> factor = monitor.portfolio_correlation_factor(
        ...     ["USD/JPY", "AUD/JPY"]
        ... )
    """

    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback
        self._returns: dict[str, pd.Series] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def update(
        self, candles: dict[str, pd.DataFrame]
    ) -> None:
        """Update correlation estimates with latest candle data.

        Args:
            candles: Dict of {pair: DataFrame with 'close' column}.
        """
        for pair, df in candles.items():
            if df is not None and "close" in df.columns and len(df) > 1:
                returns = df["close"].pct_change().dropna()
                self._returns[pair] = returns.tail(self.lookback)

        if len(self._returns) < 2:
            self._correlation_matrix = None
            return

        # Build return matrix
        combined = pd.DataFrame(self._returns)
        if len(combined) < 10:
            self._correlation_matrix = None
            return

        self._correlation_matrix = combined.corr()
        logger.debug(
            "Correlation matrix updated: %d pairs, %d periods",
            len(self._returns),
            len(combined),
        )

    def get_correlation(
        self, pair_a: str, pair_b: str
    ) -> float:
        """Get correlation between two pairs.

        Args:
            pair_a: First pair.
            pair_b: Second pair.

        Returns:
            Correlation coefficient (-1 to 1), or 0.0 if unknown.
        """
        if self._correlation_matrix is None:
            return 0.0

        if (
            pair_a not in self._correlation_matrix.columns
            or pair_b not in self._correlation_matrix.columns
        ):
            return 0.0

        corr = self._correlation_matrix.loc[pair_a, pair_b]
        return float(corr) if np.isfinite(corr) else 0.0

    def portfolio_correlation_factor(
        self, open_pairs: list[str]
    ) -> float:
        """Calculate a sizing factor based on portfolio correlation.

        Returns a value between 0.5 and 1.0:
        - 1.0: positions are uncorrelated (full size OK)
        - 0.5: positions are highly correlated (reduce size)

        Args:
            open_pairs: List of currently open position pairs.

        Returns:
            Correlation adjustment factor.
        """
        if len(open_pairs) < 2:
            return 1.0

        if self._correlation_matrix is None:
            # Default assumption: JPY crosses are correlated
            jpy_pairs = [p for p in open_pairs if "JPY" in p]
            if len(jpy_pairs) >= 2:
                return 0.7  # Conservative default for JPY pairs
            return 1.0

        # Average pairwise correlation
        total_corr = 0.0
        n_pairs = 0
        for i, pa in enumerate(open_pairs):
            for pb in open_pairs[i + 1 :]:
                corr = abs(self.get_correlation(pa, pb))
                total_corr += corr
                n_pairs += 1

        if n_pairs == 0:
            return 1.0

        avg_corr = total_corr / n_pairs

        # Map: 0 correlation -> 1.0 factor, 1.0 correlation -> 0.5
        factor = 1.0 - (avg_corr * 0.5)
        return max(0.5, min(factor, 1.0))
