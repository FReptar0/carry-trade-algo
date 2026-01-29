"""Risk metrics: VaR, CVaR, MAE, MFE.

This module calculates statistical risk measures used by professional traders
and risk managers.

VALUE AT RISK (VaR)
-------------------
"What's the maximum I expect to lose on 95% of days?"

VaR 95% = -1.65 × σ × Portfolio Value

If your VaR 95% is $10,000, it means:
- 95% of days, you'll lose less than $10,000
- 5% of days, you could lose MORE than $10,000

VaR methods:
1. HISTORICAL: Use actual past returns distribution
2. PARAMETRIC: Assume normal distribution (faster, less accurate)
3. MONTE CARLO: Simulate many scenarios (most accurate, slowest)

CONDITIONAL VAR (CVaR / Expected Shortfall)
--------------------------------------------
"When I DO exceed VaR, how bad does it get?"

CVaR is the average of losses that exceed VaR.
It captures tail risk that VaR misses.

Example:
- VaR 95% = $10,000 loss
- CVaR 95% = $15,000 loss
This means when you have a bad day (worst 5%), you lose $15k on average.

MAE / MFE (Trade-Level Risk)
----------------------------
Maximum Adverse Excursion: How far did a trade go against you?
Maximum Favorable Excursion: How far did a trade go in your favor?

Why this matters:
- If MAE often exceeds your stop loss → stop is too tight
- If MFE is much larger than final profit → you're exiting too early
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class VaRResult:
    """Result of Value at Risk calculation."""
    confidence_level: float  # e.g., 0.95 for 95%
    var: float  # VaR value (as positive loss)
    cvar: float  # Conditional VaR (expected shortfall)
    method: str  # "historical", "parametric", or "monte_carlo"
    portfolio_value: float
    var_pct: float  # VaR as percentage of portfolio
    cvar_pct: float  # CVaR as percentage of portfolio
    horizon_days: int = 1  # Time horizon


@dataclass
class TradeExcursion:
    """MAE/MFE for a single trade."""
    entry_price: float
    exit_price: float
    max_favorable: float  # Best price during trade
    max_adverse: float  # Worst price during trade
    is_long: bool

    @property
    def mae(self) -> float:
        """Maximum adverse excursion as price difference."""
        if self.is_long:
            return self.entry_price - self.max_adverse
        else:
            return self.max_adverse - self.entry_price

    @property
    def mfe(self) -> float:
        """Maximum favorable excursion as price difference."""
        if self.is_long:
            return self.max_favorable - self.entry_price
        else:
            return self.entry_price - self.max_favorable

    @property
    def mae_pct(self) -> float:
        """MAE as percentage of entry price."""
        return self.mae / self.entry_price

    @property
    def mfe_pct(self) -> float:
        """MFE as percentage of entry price."""
        return self.mfe / self.entry_price

    @property
    def realized_pct(self) -> float:
        """Realized profit/loss as percentage."""
        if self.is_long:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price


@dataclass
class ExcursionAnalysis:
    """Aggregate MAE/MFE analysis across trades."""
    trades: list[TradeExcursion] = field(default_factory=list)

    @property
    def avg_mae_pct(self) -> float:
        """Average MAE across all trades."""
        if not self.trades:
            return 0.0
        return np.mean([t.mae_pct for t in self.trades])

    @property
    def avg_mfe_pct(self) -> float:
        """Average MFE across all trades."""
        if not self.trades:
            return 0.0
        return np.mean([t.mfe_pct for t in self.trades])

    @property
    def mfe_mae_ratio(self) -> float:
        """Ratio of MFE to MAE (higher is better)."""
        mae = self.avg_mae_pct
        if mae == 0:
            return float("inf") if self.avg_mfe_pct > 0 else 0.0
        return self.avg_mfe_pct / mae

    @property
    def capture_ratio(self) -> float:
        """How much of MFE was captured in final exit."""
        if not self.trades or self.avg_mfe_pct == 0:
            return 0.0
        avg_realized = np.mean([t.realized_pct for t in self.trades])
        return avg_realized / self.avg_mfe_pct


class RiskAnalyzer:
    """Calculate risk metrics for a portfolio or returns series.

    Example:
        >>> analyzer = RiskAnalyzer()
        >>> returns = df['close'].pct_change().dropna()
        >>> var_result = analyzer.historical_var(returns, confidence=0.95, portfolio_value=100000)
        >>> print(f"VaR 95%: ${var_result.var:,.0f}")
        VaR 95%: $2,500
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

    def historical_var(
        self,
        returns: pd.Series | np.ndarray,
        confidence: float = 0.95,
        portfolio_value: float = 100000,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Calculate VaR using historical simulation.

        Uses actual past returns - no distribution assumptions.
        Most accurate for the past, but past may not predict future.

        Args:
            returns: Series of daily returns.
            confidence: Confidence level (e.g., 0.95 for 95%).
            portfolio_value: Current portfolio value.
            horizon_days: Time horizon in days.

        Returns:
            VaRResult with VaR and CVaR values.
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return VaRResult(
                confidence_level=confidence,
                var=0.0,
                cvar=0.0,
                method="historical",
                portfolio_value=portfolio_value,
                var_pct=0.0,
                cvar_pct=0.0,
                horizon_days=horizon_days,
            )

        # Scale returns for horizon
        if horizon_days > 1:
            returns = returns * np.sqrt(horizon_days)

        # VaR is the percentile of the loss distribution
        var_pct = np.percentile(returns, (1 - confidence) * 100)

        # CVaR is the mean of returns below VaR
        cvar_pct = returns[returns <= var_pct].mean() if len(returns[returns <= var_pct]) > 0 else var_pct

        # Convert to dollar values (as positive losses)
        var_dollar = abs(var_pct) * portfolio_value
        cvar_dollar = abs(cvar_pct) * portfolio_value

        return VaRResult(
            confidence_level=confidence,
            var=var_dollar,
            cvar=cvar_dollar,
            method="historical",
            portfolio_value=portfolio_value,
            var_pct=abs(var_pct),
            cvar_pct=abs(cvar_pct),
            horizon_days=horizon_days,
        )

    def parametric_var(
        self,
        returns: pd.Series | np.ndarray,
        confidence: float = 0.95,
        portfolio_value: float = 100000,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Calculate VaR assuming normal distribution.

        Fast but assumes returns are normally distributed (they're not).
        Underestimates tail risk in real markets.

        Args:
            returns: Series of daily returns.
            confidence: Confidence level.
            portfolio_value: Current portfolio value.
            horizon_days: Time horizon.

        Returns:
            VaRResult with VaR and CVaR values.
        """
        from scipy import stats

        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return VaRResult(
                confidence_level=confidence,
                var=0.0,
                cvar=0.0,
                method="parametric",
                portfolio_value=portfolio_value,
                var_pct=0.0,
                cvar_pct=0.0,
                horizon_days=horizon_days,
            )

        mean = returns.mean()
        std = returns.std()

        # Scale for horizon
        mean_h = mean * horizon_days
        std_h = std * np.sqrt(horizon_days)

        # VaR from normal distribution
        z_score = stats.norm.ppf(1 - confidence)
        var_pct = -(mean_h + z_score * std_h)

        # CVaR for normal distribution
        # E[X | X < VaR] = μ - σ × φ(z) / (1-Φ(z))
        pdf_z = stats.norm.pdf(z_score)
        cdf_z = 1 - confidence
        cvar_pct = -(mean_h - std_h * pdf_z / cdf_z)

        var_dollar = var_pct * portfolio_value
        cvar_dollar = cvar_pct * portfolio_value

        return VaRResult(
            confidence_level=confidence,
            var=var_dollar,
            cvar=cvar_dollar,
            method="parametric",
            portfolio_value=portfolio_value,
            var_pct=var_pct,
            cvar_pct=cvar_pct,
            horizon_days=horizon_days,
        )

    def monte_carlo_var(
        self,
        returns: pd.Series | np.ndarray,
        confidence: float = 0.95,
        portfolio_value: float = 100000,
        horizon_days: int = 1,
        n_simulations: int = 10000,
    ) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation.

        Simulates many possible future paths by resampling historical returns.
        More robust than parametric, can capture fat tails.

        Args:
            returns: Series of daily returns.
            confidence: Confidence level.
            portfolio_value: Current portfolio value.
            horizon_days: Time horizon.
            n_simulations: Number of simulations to run.

        Returns:
            VaRResult with VaR and CVaR values.
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return VaRResult(
                confidence_level=confidence,
                var=0.0,
                cvar=0.0,
                method="monte_carlo",
                portfolio_value=portfolio_value,
                var_pct=0.0,
                cvar_pct=0.0,
                horizon_days=horizon_days,
            )

        # Simulate paths by resampling returns
        simulated_returns = self.rng.choice(returns, size=(n_simulations, horizon_days), replace=True)

        # Calculate cumulative return for each path
        path_returns = np.prod(1 + simulated_returns, axis=1) - 1

        # VaR and CVaR from simulated distribution
        var_pct = np.percentile(path_returns, (1 - confidence) * 100)
        cvar_pct = path_returns[path_returns <= var_pct].mean() if len(path_returns[path_returns <= var_pct]) > 0 else var_pct

        var_dollar = abs(var_pct) * portfolio_value
        cvar_dollar = abs(cvar_pct) * portfolio_value

        return VaRResult(
            confidence_level=confidence,
            var=var_dollar,
            cvar=cvar_dollar,
            method="monte_carlo",
            portfolio_value=portfolio_value,
            var_pct=abs(var_pct),
            cvar_pct=abs(cvar_pct),
            horizon_days=horizon_days,
        )

    def calculate_excursions(
        self,
        trades: list[dict],
        prices_during_trade: dict[int, pd.DataFrame],
    ) -> ExcursionAnalysis:
        """Calculate MAE/MFE for a list of trades.

        Args:
            trades: List of trade dicts with entry_price, exit_price, is_long.
            prices_during_trade: Dict mapping trade index to price data during trade.

        Returns:
            ExcursionAnalysis with all trade excursions.
        """
        excursions = []

        for i, trade in enumerate(trades):
            if i not in prices_during_trade:
                continue

            prices = prices_during_trade[i]
            entry = trade["entry_price"]
            exit_price = trade["exit_price"]
            is_long = trade.get("is_long", True)

            max_fav = prices["high"].max() if is_long else prices["low"].min()
            max_adv = prices["low"].min() if is_long else prices["high"].max()

            excursions.append(TradeExcursion(
                entry_price=entry,
                exit_price=exit_price,
                max_favorable=max_fav,
                max_adverse=max_adv,
                is_long=is_long,
            ))

        return ExcursionAnalysis(trades=excursions)

    def risk_summary(
        self,
        returns: pd.Series | np.ndarray,
        portfolio_value: float = 100000,
    ) -> dict:
        """Generate comprehensive risk summary.

        Args:
            returns: Series of daily returns.
            portfolio_value: Current portfolio value.

        Returns:
            Dict with all risk metrics.
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        var_95 = self.historical_var(returns, 0.95, portfolio_value)
        var_99 = self.historical_var(returns, 0.99, portfolio_value)

        return {
            "var_95_pct": var_95.var_pct,
            "var_95_dollar": var_95.var,
            "cvar_95_pct": var_95.cvar_pct,
            "cvar_95_dollar": var_95.cvar,
            "var_99_pct": var_99.var_pct,
            "var_99_dollar": var_99.var,
            "cvar_99_pct": var_99.cvar_pct,
            "cvar_99_dollar": var_99.cvar,
            "daily_volatility": returns.std() if len(returns) > 0 else 0,
            "annualized_volatility": returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            "worst_day": returns.min() if len(returns) > 0 else 0,
            "best_day": returns.max() if len(returns) > 0 else 0,
            "skewness": pd.Series(returns).skew() if len(returns) > 0 else 0,
            "kurtosis": pd.Series(returns).kurtosis() if len(returns) > 0 else 0,
            "negative_days_pct": (returns < 0).mean() if len(returns) > 0 else 0,
        }
