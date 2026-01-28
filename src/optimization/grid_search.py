"""Grid search parameter optimizer.

In Phase 2, our strategy lost money because the parameters were wrong
(stop loss too tight for volatile pairs). Grid search fixes this by
trying EVERY combination of parameters and finding the best one.

How it works (simple version):
    1. You give it a list of values to try for each parameter
       e.g., stop_loss = [2%, 3%, 4%, 5%], SMA period = [20, 50, 100]
    2. It runs a backtest for EVERY combination (4 x 3 = 12 runs)
    3. It ranks them by a metric you choose (usually Sharpe ratio)
    4. You get back the best combination

Warning: grid search can find parameters that ONLY work on past data
(overfitting). That's why we also do walk-forward analysis.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.config.settings import RiskConfig, StrategyConfig
from src.strategy.carry_trade import CarryTradeStrategy


@dataclass
class OptimizationResult:
    """One row in the optimization results table."""

    params: dict
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float


class GridSearchOptimizer:
    """Tries all parameter combinations and finds the best one.

    Args:
        data: Price data to backtest on.
        pair_name: Forex pair name.
        risk_config: Risk settings (shared across all runs).
        rank_by: Which metric to optimize for. Default "sharpe".

    Example:
        >>> opt = GridSearchOptimizer(data, "USD/JPY")
        >>> results = opt.run({"stop_loss_pct": [0.02, 0.03, 0.05]})
        >>> print(results[0].params)  # best combination
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pair_name: str,
        risk_config: RiskConfig | None = None,
        rank_by: str = "sharpe",
    ) -> None:
        self.data = data
        self.pair_name = pair_name
        self.risk_config = risk_config or RiskConfig()
        self.rank_by = rank_by

    def run(self, param_grid: dict[str, list]) -> list[OptimizationResult]:
        """Run grid search over all parameter combinations.

        Args:
            param_grid: Dict mapping parameter names to lists of values.
                e.g. {"stop_loss_pct": [0.02, 0.03], "trend_filter_period": [20, 50]}

        Returns:
            List of OptimizationResult sorted by rank_by metric (best first).
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results: list[OptimizationResult] = []

        for combo in combos:
            params = dict(zip(keys, combo))
            config = StrategyConfig(**params)
            strategy = CarryTradeStrategy(config)
            engine = BacktestEngine(strategy, config, self.risk_config)

            bt_result = engine.run(self.data, self.pair_name)
            m = bt_result.metrics

            results.append(OptimizationResult(
                params=params,
                sharpe=m.sharpe_ratio,
                total_return=m.total_return,
                max_drawdown=m.max_drawdown,
                win_rate=m.win_rate,
                total_trades=m.total_trades,
                profit_factor=m.profit_factor,
            ))

        # Sort by chosen metric (higher is better for sharpe/return, lower for drawdown)
        reverse = self.rank_by != "max_drawdown"
        results.sort(key=lambda r: getattr(r, self.rank_by), reverse=reverse)

        return results

    def results_to_df(self, results: list[OptimizationResult]) -> pd.DataFrame:
        """Convert results list to a DataFrame for easy viewing."""
        rows = []
        for r in results:
            row = {**r.params}
            row["sharpe"] = r.sharpe
            row["total_return"] = r.total_return
            row["max_drawdown"] = r.max_drawdown
            row["win_rate"] = r.win_rate
            row["total_trades"] = r.total_trades
            row["profit_factor"] = r.profit_factor
            rows.append(row)
        return pd.DataFrame(rows)
