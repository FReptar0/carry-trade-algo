"""Walk-forward analysis -- the antidote to overfitting.

Grid search finds the "best" parameters on past data, but those
parameters might only work on THAT specific data (overfitting).

Walk-forward analysis tests whether optimal parameters STAY good
on data they've never seen:

    1. Split data into windows:
       [--- Train 6 months ---][- Test 1 month -]
                   [--- Train 6 months ---][- Test 1 month -]
                               [--- Train 6 months ---][- Test 1 month -]

    2. For each window:
       a. Find best parameters on the training data (grid search)
       b. Test those parameters on the NEXT month (out-of-sample)

    3. If the strategy works on the test windows too, it's more likely
       to work in real trading.

If the best parameters keep changing wildly between windows,
that's a red flag -- the strategy is probably overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.config.settings import StrategyConfig
from src.optimization.grid_search import GridSearchOptimizer
from src.strategy.carry_trade import CarryTradeStrategy


@dataclass
class WalkForwardWindow:
    """Results for one train/test window."""

    window_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_trades: int


class WalkForwardAnalyzer:
    """Runs walk-forward analysis to validate parameter stability.

    Args:
        data: Full price data.
        pair_name: Forex pair name.
        param_grid: Parameter grid to optimize over.
        train_days: Training window size in calendar days (default 180 = 6 months).
        test_days: Testing window size in calendar days (default 30 = 1 month).
        step_days: How far to slide the window each step (default 30 = 1 month).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pair_name: str,
        param_grid: dict[str, list],
        train_days: int = 180,
        test_days: int = 30,
        step_days: int = 30,
    ) -> None:
        self.data = data
        self.pair_name = pair_name
        self.param_grid = param_grid
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def run(self) -> list[WalkForwardWindow]:
        """Execute walk-forward analysis across all windows.

        Returns:
            List of WalkForwardWindow results for each train/test split.
        """
        timestamps = self.data["timestamp"]
        start = timestamps.iloc[0]
        end = timestamps.iloc[-1]

        results: list[WalkForwardWindow] = []
        window_num = 0
        current = start

        while True:
            train_start = current
            train_end = train_start + pd.Timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_days)

            if test_end > end:
                break

            # Split data
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            test_mask = (timestamps >= test_start) & (timestamps < test_end)
            train_data = self.data[train_mask].reset_index(drop=True)
            test_data = self.data[test_mask].reset_index(drop=True)

            if len(train_data) < 100 or len(test_data) < 20:
                current += pd.Timedelta(days=self.step_days)
                continue

            # Optimize on training data
            optimizer = GridSearchOptimizer(train_data, self.pair_name)
            opt_results = optimizer.run(self.param_grid)

            if not opt_results:
                current += pd.Timedelta(days=self.step_days)
                continue

            best = opt_results[0]

            # Test best params on out-of-sample data
            config = StrategyConfig(**best.params)
            strategy = CarryTradeStrategy(config)
            engine = BacktestEngine(strategy, config)
            test_result = engine.run(test_data, self.pair_name)

            results.append(WalkForwardWindow(
                window_num=window_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best.params,
                train_sharpe=best.sharpe,
                test_sharpe=test_result.metrics.sharpe_ratio,
                test_return=test_result.metrics.total_return,
                test_trades=test_result.metrics.total_trades,
            ))

            window_num += 1
            current += pd.Timedelta(days=self.step_days)

        return results

    def results_to_df(self, results: list[WalkForwardWindow]) -> pd.DataFrame:
        """Convert walk-forward results to DataFrame."""
        rows = []
        for w in results:
            row = {
                "window": w.window_num,
                "train_period": f"{w.train_start.date()} → {w.train_end.date()}",
                "test_period": f"{w.test_start.date()} → {w.test_end.date()}",
                "train_sharpe": w.train_sharpe,
                "test_sharpe": w.test_sharpe,
                "test_return": w.test_return,
                "test_trades": w.test_trades,
            }
            # Add individual params
            for k, v in w.best_params.items():
                row[f"param_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def summary(results: list[WalkForwardWindow]) -> str:
        """Print human-readable summary of walk-forward results."""
        if not results:
            return "No walk-forward windows completed."

        test_sharpes = [w.test_sharpe for w in results]
        test_returns = [w.test_return for w in results]
        avg_sharpe = sum(test_sharpes) / len(test_sharpes)
        avg_return = sum(test_returns) / len(test_returns)
        profitable_windows = sum(1 for r in test_returns if r > 0)

        lines = [
            "=== Walk-Forward Summary ===",
            f"Windows completed:    {len(results)}",
            f"Avg test Sharpe:      {avg_sharpe:.2f}",
            f"Avg test return:      {avg_return:.2%}",
            f"Profitable windows:   {profitable_windows}/{len(results)}",
            "",
        ]

        # Check parameter stability
        param_keys = list(results[0].best_params.keys())
        for key in param_keys:
            vals = [w.best_params[key] for w in results]
            unique = set(vals)
            lines.append(f"  {key}: {len(unique)} unique values across windows")

        return "\n".join(lines)
