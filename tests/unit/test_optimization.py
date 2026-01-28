"""Tests for grid search and walk-forward optimization."""

import pandas as pd

from src.config.settings import DEFAULT_PAIRS
from src.data.generator import SyntheticDataGenerator
from src.optimization.grid_search import GridSearchOptimizer
from src.optimization.walk_forward import WalkForwardAnalyzer


def _get_data(days: int = 200) -> pd.DataFrame:
    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_pair(DEFAULT_PAIRS[1], days=days)  # AUD/JPY


SMALL_GRID = {
    "stop_loss_pct": [0.03, 0.06],
    "trend_filter_period": [20, 50],
}


class TestGridSearch:
    def test_returns_list(self):
        data = _get_data()
        opt = GridSearchOptimizer(data, "AUD/JPY")
        results = opt.run(SMALL_GRID)
        assert isinstance(results, list)
        assert len(results) == 4  # 2 x 2

    def test_sorted_by_sharpe(self):
        data = _get_data()
        opt = GridSearchOptimizer(data, "AUD/JPY")
        results = opt.run(SMALL_GRID)
        sharpes = [r.sharpe for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_results_have_params(self):
        data = _get_data()
        opt = GridSearchOptimizer(data, "AUD/JPY")
        results = opt.run(SMALL_GRID)
        for r in results:
            assert "stop_loss_pct" in r.params
            assert "trend_filter_period" in r.params

    def test_to_dataframe(self):
        data = _get_data()
        opt = GridSearchOptimizer(data, "AUD/JPY")
        results = opt.run(SMALL_GRID)
        df = opt.results_to_df(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "sharpe" in df.columns

    def test_rank_by_return(self):
        data = _get_data()
        opt = GridSearchOptimizer(data, "AUD/JPY", rank_by="total_return")
        results = opt.run(SMALL_GRID)
        returns = [r.total_return for r in results]
        assert returns == sorted(returns, reverse=True)


class TestWalkForward:
    def test_returns_list(self):
        data = _get_data(days=365)
        wf = WalkForwardAnalyzer(data, "AUD/JPY", SMALL_GRID,
                                  train_days=150, test_days=30, step_days=60)
        results = wf.run()
        assert isinstance(results, list)

    def test_windows_have_fields(self):
        data = _get_data(days=365)
        wf = WalkForwardAnalyzer(data, "AUD/JPY", SMALL_GRID,
                                  train_days=150, test_days=30, step_days=60)
        results = wf.run()
        if results:
            w = results[0]
            assert hasattr(w, "best_params")
            assert hasattr(w, "test_sharpe")
            assert hasattr(w, "train_sharpe")

    def test_to_dataframe(self):
        data = _get_data(days=365)
        wf = WalkForwardAnalyzer(data, "AUD/JPY", SMALL_GRID,
                                  train_days=150, test_days=30, step_days=60)
        results = wf.run()
        df = wf.results_to_df(results)
        assert isinstance(df, pd.DataFrame)

    def test_summary_string(self):
        data = _get_data(days=365)
        wf = WalkForwardAnalyzer(data, "AUD/JPY", SMALL_GRID,
                                  train_days=150, test_days=30, step_days=60)
        results = wf.run()
        s = WalkForwardAnalyzer.summary(results)
        assert "Walk-Forward" in s

    def test_empty_results(self):
        s = WalkForwardAnalyzer.summary([])
        assert "No walk-forward" in s
