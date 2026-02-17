"""Dynamic position sizer using Kelly criterion, volatility, and correlation.

Replaces fixed position_size_units with adaptive sizing that considers:
1. Kelly criterion (optimal bet size based on win rate and payoff)
2. Volatility adjustment (scale inversely with realized vol)
3. Regime multiplier from the adapter
4. Correlation factor (reduce when positions are correlated)
5. Drawdown scalar (quadratic decay as drawdown increases)
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

DEFAULT_MIN_UNITS = 1000
DEFAULT_MAX_UNITS = 25000


class DynamicSizer:
    """Calculates adaptive position sizes.

    Args:
        base_units: Default position size in units.
        max_portfolio_risk_pct: Max portfolio risk per trade.
        target_vol: Target annualized volatility for sizing.
        min_units: Floor for position size.
        max_units: Ceiling for position size.
        max_drawdown_pct: Maximum drawdown before circuit breaker (0-1).
        drawdown_floor: Minimum scaling factor at max drawdown.

    Example:
        >>> sizer = DynamicSizer(base_units=10000)
        >>> units = sizer.calculate_size(
        ...     pair="USD/JPY", equity=100000, atr=0.5,
        ...     price=150.0, win_rate=0.55, avg_win=200,
        ...     avg_loss=100, regime_mult=1.0,
        ...     correlation_factor=1.0,
        ... )
    """

    def __init__(
        self,
        base_units: int = 10000,
        max_portfolio_risk_pct: float = 0.02,
        target_vol: float = 0.10,
        min_units: int = DEFAULT_MIN_UNITS,
        max_units: int = DEFAULT_MAX_UNITS,
        max_drawdown_pct: float = 0.20,
        drawdown_floor: float = 0.25,
    ) -> None:
        self.base_units = base_units
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.target_vol = target_vol
        self.min_units = min_units
        self.max_units = max_units
        self.max_drawdown_pct = max_drawdown_pct
        self.drawdown_floor = drawdown_floor

    def drawdown_scalar(self, drawdown_pct: float) -> float:
        """Quadratic decay factor based on current drawdown.

        Smoothly reduces position size as drawdown increases, reaching
        the floor at max_drawdown_pct. Quadratic decay is gentle for
        small drawdowns but drops fast near the limit.

        Args:
            drawdown_pct: Current drawdown as a fraction (0-1).

        Returns:
            Scaling factor between drawdown_floor and 1.0.
        """
        clamped = min(max(drawdown_pct, 0.0), self.max_drawdown_pct)
        return max(
            self.drawdown_floor,
            1.0 - (clamped / self.max_drawdown_pct) ** 2,
        )

    def calculate_size(
        self,
        pair: str,
        equity: float,
        atr: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        regime_mult: float = 1.0,
        correlation_factor: float = 1.0,
        drawdown_pct: float = 0.0,
    ) -> int:
        """Calculate position size in units.

        Args:
            pair: Currency pair name.
            equity: Current account equity.
            atr: Current ATR value.
            price: Current price.
            win_rate: Historical win rate (0-1).
            avg_win: Average winning trade P&L.
            avg_loss: Average losing trade P&L (positive number).
            regime_mult: Position size multiplier from regime adapter.
            correlation_factor: 0.5-1.0, lower = more correlated.
            drawdown_pct: Current portfolio drawdown (0-1).

        Returns:
            Position size in units (integer).
        """
        # 1. Kelly fraction
        kelly = self._kelly_fraction(win_rate, avg_win, avg_loss)
        half_kelly = kelly * 0.5  # Conservative

        # 2. Volatility adjustment
        if price > 0 and atr > 0:
            realized_vol = (atr / price) * math.sqrt(252)
            vol_scale = (
                self.target_vol / realized_vol
                if realized_vol > 0
                else 1.0
            )
            vol_scale = min(max(vol_scale, 0.5), 2.0)
        else:
            vol_scale = 1.0

        # 3. Risk-based sizing: equity * risk_pct / (atr * units_per_lot)
        if atr > 0 and price > 0:
            risk_budget = equity * self.max_portfolio_risk_pct
            risk_units = int(risk_budget / atr)
        else:
            risk_units = self.base_units

        # 4. Combine: min(kelly-based, risk-based) * adjustments
        kelly_units = int(equity * half_kelly / price) if price > 0 else self.base_units
        raw_units = min(kelly_units, risk_units)

        # 5. Apply multipliers (including drawdown scalar)
        dd_factor = self.drawdown_scalar(drawdown_pct)
        adjusted = int(
            raw_units * vol_scale * regime_mult * correlation_factor * dd_factor
        )

        # 6. Clamp
        result = max(self.min_units, min(adjusted, self.max_units))

        logger.debug(
            "DynamicSizer %s: kelly=%.3f vol_scale=%.2f "
            "regime=%.2f corr=%.2f dd=%.2f -> %d units",
            pair,
            half_kelly,
            vol_scale,
            regime_mult,
            correlation_factor,
            dd_factor,
            result,
        )
        return result

    @staticmethod
    def _kelly_fraction(
        win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate Kelly criterion fraction.

        f = W - (1-W) / R
        where W = win rate, R = avg_win / avg_loss

        Returns:
            Kelly fraction (clamped to [0, 0.25]).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0

        payoff_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / payoff_ratio

        # Clamp: never risk more than 25% even if Kelly says so
        return max(0.0, min(kelly, 0.25))
