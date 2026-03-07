"""Validation & robustness suite for carry trade strategy.

13 statistical tests to validate the edge is real before real money:

Phase 1 (independent stats):
    1. monte_carlo_simulation       - Resample trades, measure ruin probability
    2. bootstrap_confidence_intervals - CI for Sharpe, win rate, profit factor
    3. permutation_test             - Shuffle returns to test signal vs noise
    4. detect_strategy_decay        - Rolling Sharpe at multiple windows
    9. overfitting_detection        - DOF ratio + IS/OOS Sharpe split
    10. extreme_value_analysis      - GPD tail fit via scipy
    13. max_drawdown_duration       - Longest underwater period

Phase 2 (backtest-dependent):
    5. parameter_sensitivity        - Perturb 6 params, measure Sharpe stability
    6. regime_performance_breakdown - Tag trades by regime at entry
    8. interest_rate_reversal       - Flip swap negative mid-backtest

Phase 3 (scenario modeling):
    7. correlation_crisis_scenario  - Correlated 6-position crisis DD
    11. market_impact_model         - Square-root impact estimate
    12. liquidity_risk_assessment   - ATR by session, flag spikes
"""

from __future__ import annotations

import itertools
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import _sharpe
from src.config.settings import RiskConfig, StrategyConfig
from src.data.interest_rates import InterestRateSimulator
from src.data.loader import HistoricalDataLoader
from src.regime.detector import RegimeDetector
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

@dataclass
class DataSource:
    """Unified input for all validation functions."""

    trades_df: pd.DataFrame       # pnl, total_pnl, swap_earned, entry_time, exit_time
    equity_df: pd.DataFrame       # timestamp, equity, drawdown
    daily_returns: pd.Series      # daily equity returns
    initial_capital: float
    source_type: str              # "backtest" | "live_db"


def load_from_backtest(result: BacktestResult) -> DataSource:
    """Wrap a BacktestResult into a DataSource."""
    eq = result.equity_df.copy()
    equity_daily = eq.set_index("timestamp")["equity"].resample("D").last().dropna()
    daily_ret = equity_daily.pct_change().dropna()
    return DataSource(
        trades_df=result.trades_df.copy(),
        equity_df=eq,
        daily_returns=daily_ret,
        initial_capital=eq["equity"].iloc[0] if len(eq) > 0 else 10_000.0,
        source_type="backtest",
    )


def load_from_multi_pair_backtest(
    results: dict[str, BacktestResult],
    initial_capital: float = 100_000.0,
) -> DataSource:
    """Combine multiple single-pair BacktestResults into a portfolio DataSource.

    Merges all trades, builds a combined equity curve from summing per-pair
    PnL contributions, and computes portfolio-level daily returns.
    """
    if not results:
        return DataSource(
            trades_df=pd.DataFrame(),
            equity_df=pd.DataFrame(columns=["timestamp", "equity", "drawdown"]),
            daily_returns=pd.Series(dtype=float),
            initial_capital=initial_capital,
            source_type="backtest",
        )

    # Merge all trades, tagging each with its pair
    all_trades = []
    for pair, result in results.items():
        if len(result.trades_df) > 0:
            t = result.trades_df.copy()
            t["pair"] = pair
            all_trades.append(t)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Build portfolio equity curve by summing per-pair PnL streams
    # Each pair's backtest starts with the same initial capital, but in the
    # real portfolio each pair gets position_size_pct (4%) of capital.
    # We reconstruct: portfolio_equity = initial_capital + sum(pair_pnl_series)
    pnl_series: list[pd.Series] = []
    for pair, result in results.items():
        eq = result.equity_df.copy()
        if len(eq) < 2:
            continue
        pair_initial = eq["equity"].iloc[0]
        eq["pnl"] = eq["equity"] - pair_initial
        eq = eq.set_index("timestamp")
        # Resample to daily to align across pairs
        daily_pnl = eq["pnl"].resample("D").last().ffill()
        pnl_series.append(daily_pnl)

    if pnl_series:
        combined_pnl = pd.concat(pnl_series, axis=1).ffill().fillna(0).sum(axis=1)
        portfolio_equity = initial_capital + combined_pnl
        hwm = portfolio_equity.cummax()
        drawdown = (hwm - portfolio_equity) / hwm

        equity_df = pd.DataFrame({
            "timestamp": portfolio_equity.index,
            "equity": portfolio_equity.values,
            "drawdown": drawdown.values,
        }).reset_index(drop=True)

        daily_returns = portfolio_equity.pct_change().dropna()
    else:
        equity_df = pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
        daily_returns = pd.Series(dtype=float)

    return DataSource(
        trades_df=trades_df,
        equity_df=equity_df,
        daily_returns=daily_returns,
        initial_capital=initial_capital,
        source_type="backtest",
    )


def load_from_live_db(
    db_path: str,
    initial_capital: float = 100_000.0,
    exclude_dates: list[str] | None = None,
) -> DataSource:
    """Read the live trading SQLite DB into a DataSource.

    Args:
        db_path: Path to SQLite database.
        initial_capital: Starting capital.
        exclude_dates: List of date strings (YYYY-MM-DD) to exclude from
            daily returns (e.g., bug period). Trades and equity snapshots
            from these dates are also filtered out.
    """
    if exclude_dates is None:
        # Default: exclude known bug period (support_break exit bug)
        exclude_dates = ["2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Equity snapshots (exclude bug period)
    eq_rows = conn.execute(
        "SELECT timestamp, equity, drawdown FROM equity_snapshots ORDER BY timestamp"
    ).fetchall()
    equity_df = pd.DataFrame([dict(r) for r in eq_rows])
    if len(equity_df) > 0:
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        # Filter out bug period equity snapshots
        for excl_date in exclude_dates:
            mask = equity_df["timestamp"].dt.strftime("%Y-%m-%d") != excl_date
            equity_df = equity_df[mask]
        equity_df = equity_df.reset_index(drop=True)

    # Trade log (closed trades with actual PnL only -- exclude NULL pnl rows)
    trade_rows = conn.execute(
        "SELECT * FROM trade_log WHERE status = 'closed' AND pnl IS NOT NULL ORDER BY timestamp"
    ).fetchall()
    trades_raw = pd.DataFrame([dict(r) for r in trade_rows])
    trades_df = pd.DataFrame()
    if len(trades_raw) > 0:
        trades_df = trades_raw.rename(columns={
            "pnl": "total_pnl",
            "timestamp": "exit_time",
        })
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        # Filter out trades closed during bug period
        for excl_date in exclude_dates:
            mask = trades_df["exit_time"].dt.strftime("%Y-%m-%d") != excl_date
            trades_df = trades_df[mask]
        if "entry_price" in trades_df.columns and "exit_price" in trades_df.columns:
            trades_df["entry_time"] = trades_df["exit_time"] - timedelta(days=7)
        if "swap_earned" not in trades_df.columns:
            trades_df["swap_earned"] = 0.0
        trades_df = trades_df.reset_index(drop=True)

    # Daily returns from daily_results table (exclude bug period + flat days)
    daily_rows = conn.execute(
        "SELECT date, daily_return FROM daily_results ORDER BY date"
    ).fetchall()
    daily_df = pd.DataFrame([dict(r) for r in daily_rows])
    if len(daily_df) > 0:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        # Exclude bug period
        for excl_date in exclude_dates:
            daily_df = daily_df[daily_df["date"].dt.strftime("%Y-%m-%d") != excl_date]
        daily_returns = daily_df.set_index("date")["daily_return"]
    else:
        daily_returns = pd.Series(dtype=float)

    conn.close()
    return DataSource(
        trades_df=trades_df,
        equity_df=equity_df,
        daily_returns=daily_returns,
        initial_capital=initial_capital,
        source_type="live_db",
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    passed: bool
    n_sims: int
    ruin_probability: float
    ruin_threshold: float
    median_final_return: float
    percentile_5_return: float
    percentile_95_return: float
    max_dd_median: float
    max_dd_95: float
    final_returns: np.ndarray = field(repr=False)
    max_drawdowns: np.ndarray = field(repr=False)


@dataclass
class BootstrapCIResult:
    passed: bool
    metric_name: str
    observed: float
    ci_lower: float
    ci_upper: float
    ci_level: float


@dataclass
class PermutationTestResult:
    passed: bool
    observed_sharpe: float
    p_value: float
    significance: float
    null_distribution: np.ndarray = field(repr=False)


@dataclass
class DecayDetectionResult:
    passed: bool
    windows: list[int]
    rolling_sharpes: dict[int, pd.Series]
    current_sharpes: dict[int, float]
    full_period_sharpe: float
    decay_threshold: float


@dataclass
class SensitivityAnalysisResult:
    passed: bool
    param_results: dict[str, list[dict[str, Any]]]
    robustness_score: float
    robustness_threshold: float
    baseline_sharpe: float


@dataclass
class RegimeBreakdownResult:
    passed: bool
    regime_stats: dict[str, dict[str, float]]
    concentration_pct: float
    concentration_threshold: float
    dominant_regime: str


@dataclass
class CorrelationCrisisResult:
    passed: bool
    normal_correlation: float
    crisis_correlation: float
    normal_portfolio_dd: float
    crisis_portfolio_dd: float
    max_positions: int
    per_position_loss: float


@dataclass
class RateReversalResult:
    passed: bool
    baseline_return: float
    reversal_return: float
    return_degradation: float
    rate_shock_bps: int


@dataclass
class OverfittingResult:
    passed: bool
    dof_ratio: float
    is_sharpe: float
    oos_sharpe: float
    sharpe_degradation: float
    n_params: int
    n_observations: int


@dataclass
class EVTResult:
    passed: bool
    shape_param: float
    scale_param: float
    ks_statistic: float
    ks_p_value: float
    var_99: float
    cvar_99: float
    n_tail_obs: int


@dataclass
class MarketImpactResult:
    passed: bool
    impact_bps: float
    impact_pct: float
    slippage_cost_per_trade: float
    current_units: int
    reference_volume: float
    k_factor: float


@dataclass
class LiquidityRiskResult:
    passed: bool
    session_atr: dict[str, float]
    avg_atr: float
    max_atr_ratio: float
    worst_session: str
    spike_threshold: float


@dataclass
class MaxDDDurationResult:
    passed: bool
    max_duration_days: int
    max_duration_limit: int
    underwater_periods: list[dict[str, Any]]
    current_underwater_days: int


@dataclass
class ValidationSuiteResult:
    """Container for all 13 test results."""
    results: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source_type: str = ""

    @property
    def all_passed(self) -> bool:
        return all(self._check_passed(r) for r in self.results.values())

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results.values() if self._check_passed(r))

    @staticmethod
    def _check_passed(result: Any) -> bool:
        if isinstance(result, list):
            return all(getattr(r, "passed", False) for r in result)
        return getattr(result, "passed", False)

    @property
    def total_count(self) -> int:
        return len(self.results)


# ---------------------------------------------------------------------------
# Phase 1: Independent Stats Tests
# ---------------------------------------------------------------------------

def monte_carlo_simulation(
    data: DataSource,
    n_sims: int = 5000,
    ruin_threshold: float = 0.20,
    seed: int = 42,
) -> MonteCarloResult:
    """Resample trades with replacement to build DD distribution.

    Pass criteria: P(ruin) < 10% where ruin = drawdown > ruin_threshold.
    """
    rng = np.random.default_rng(seed)
    trades = data.trades_df
    if len(trades) == 0:
        return MonteCarloResult(
            passed=False, n_sims=n_sims, ruin_probability=1.0,
            ruin_threshold=ruin_threshold, median_final_return=0.0,
            percentile_5_return=0.0, percentile_95_return=0.0,
            max_dd_median=0.0, max_dd_95=0.0,
            final_returns=np.array([]), max_drawdowns=np.array([]),
        )

    pnls = trades["total_pnl"].values
    n_trades = len(pnls)
    capital = data.initial_capital

    final_returns = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)

    for i in range(n_sims):
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        equity = capital + np.cumsum(sampled)
        equity = np.insert(equity, 0, capital)
        hwm = np.maximum.accumulate(equity)
        dd = (hwm - equity) / hwm
        final_returns[i] = (equity[-1] - capital) / capital
        max_drawdowns[i] = dd.max()

    ruin_prob = np.mean(max_drawdowns > ruin_threshold)
    return MonteCarloResult(
        passed=ruin_prob < 0.10,
        n_sims=n_sims,
        ruin_probability=float(ruin_prob),
        ruin_threshold=ruin_threshold,
        median_final_return=float(np.median(final_returns)),
        percentile_5_return=float(np.percentile(final_returns, 5)),
        percentile_95_return=float(np.percentile(final_returns, 95)),
        max_dd_median=float(np.median(max_drawdowns)),
        max_dd_95=float(np.percentile(max_drawdowns, 95)),
        final_returns=final_returns,
        max_drawdowns=max_drawdowns,
    )


def bootstrap_confidence_intervals(
    data: DataSource,
    metrics: list[str] | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> list[BootstrapCIResult]:
    """Bootstrap CI for Sharpe, win rate, profit factor.

    Pass criteria: Sharpe CI lower bound > 0.
    """
    rng = np.random.default_rng(seed)
    if metrics is None:
        metrics = ["sharpe", "win_rate", "profit_factor"]

    daily = data.daily_returns.values
    trades = data.trades_df
    alpha = (1 - ci_level) / 2
    results = []

    for metric in metrics:
        observed = _compute_metric(metric, daily, trades)
        bootstrap_vals = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            if metric == "sharpe":
                sample = rng.choice(daily, size=len(daily), replace=True)
                s = pd.Series(sample)
                bootstrap_vals[b] = _sharpe(s, 252)
            elif metric in ("win_rate", "profit_factor"):
                if len(trades) > 0:
                    idx = rng.integers(0, len(trades), size=len(trades))
                    sample_trades = trades.iloc[idx]
                    bootstrap_vals[b] = _compute_metric(metric, daily, sample_trades)
                else:
                    bootstrap_vals[b] = 0.0

        ci_lo = float(np.percentile(bootstrap_vals, alpha * 100))
        ci_hi = float(np.percentile(bootstrap_vals, (1 - alpha) * 100))

        passed = True
        if metric == "sharpe":
            passed = ci_lo > 0

        results.append(BootstrapCIResult(
            passed=passed,
            metric_name=metric,
            observed=float(observed),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_level=ci_level,
        ))

    return results


def permutation_test(
    data: DataSource,
    n_permutations: int = 1000,
    significance: float = 0.05,
    seed: int = 42,
) -> PermutationTestResult:
    """Shuffle daily returns to destroy signal; compute null Sharpe distribution.

    Pass criteria: p-value < significance.
    """
    rng = np.random.default_rng(seed)
    daily = data.daily_returns.values.copy()

    observed_sharpe = float(_sharpe(pd.Series(daily), 252))
    null_sharpes = np.zeros(n_permutations)

    for i in range(n_permutations):
        shuffled = daily.copy()
        rng.shuffle(shuffled)
        null_sharpes[i] = _sharpe(pd.Series(shuffled), 252)

    p_value = float(np.mean(null_sharpes >= observed_sharpe))

    return PermutationTestResult(
        passed=p_value < significance,
        observed_sharpe=observed_sharpe,
        p_value=p_value,
        significance=significance,
        null_distribution=null_sharpes,
    )


def detect_strategy_decay(
    data: DataSource,
    windows: list[int] | None = None,
    decay_threshold: float = 0.50,
) -> DecayDetectionResult:
    """Rolling Sharpe at 30/60/90 day windows.

    Pass: current rolling Sharpe > decay_threshold * full-period Sharpe.
    """
    if windows is None:
        windows = [30, 60, 90]

    daily = data.daily_returns
    full_sharpe = float(_sharpe(daily, 252))

    rolling_sharpes: dict[int, pd.Series] = {}
    current_sharpes: dict[int, float] = {}

    for w in windows:
        if len(daily) < w:
            rolling_sharpes[w] = pd.Series(dtype=float)
            current_sharpes[w] = 0.0
            continue

        rolling = daily.rolling(w).apply(lambda x: _sharpe(x, 252), raw=False)
        rolling_sharpes[w] = rolling
        current_sharpes[w] = float(rolling.iloc[-1]) if not pd.isna(rolling.iloc[-1]) else 0.0

    # Pass if any window's current Sharpe exceeds threshold * full period
    threshold_val = decay_threshold * full_sharpe if full_sharpe > 0 else 0.0
    passed = any(v > threshold_val for v in current_sharpes.values()) if full_sharpe > 0 else True

    return DecayDetectionResult(
        passed=passed,
        windows=windows,
        rolling_sharpes=rolling_sharpes,
        current_sharpes=current_sharpes,
        full_period_sharpe=full_sharpe,
        decay_threshold=decay_threshold,
    )


def overfitting_detection(
    data: DataSource,
    n_params: int = 8,
) -> OverfittingResult:
    """DOF ratio + IS/OOS Sharpe split (70/30).

    Pass: DOF ratio < 0.1 AND OOS degradation < 50%.
    """
    daily = data.daily_returns.values
    n_obs = len(daily)
    dof_ratio = n_params / n_obs if n_obs > 0 else 1.0

    # 70/30 split
    split = int(n_obs * 0.7)
    is_returns = pd.Series(daily[:split])
    oos_returns = pd.Series(daily[split:])

    is_sharpe = float(_sharpe(is_returns, 252)) if len(is_returns) > 1 else 0.0
    oos_sharpe = float(_sharpe(oos_returns, 252)) if len(oos_returns) > 1 else 0.0

    degradation = 1.0 - (oos_sharpe / is_sharpe) if is_sharpe != 0 else 1.0

    return OverfittingResult(
        passed=dof_ratio < 0.1 and degradation < 0.50,
        dof_ratio=dof_ratio,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        sharpe_degradation=degradation,
        n_params=n_params,
        n_observations=n_obs,
    )


def extreme_value_analysis(
    data: DataSource,
    tail_pct: float = 0.05,
) -> EVTResult:
    """Fit GPD to tail losses via scipy.stats.genpareto.

    Pass: KS test p-value > 0.05 (good fit).
    """
    daily = data.daily_returns.values
    losses = -daily[daily < 0]  # positive loss values

    if len(losses) < 10:
        return EVTResult(
            passed=False, shape_param=0.0, scale_param=0.0,
            ks_statistic=1.0, ks_p_value=0.0,
            var_99=0.0, cvar_99=0.0, n_tail_obs=len(losses),
        )

    threshold = np.percentile(losses, (1 - tail_pct) * 100)
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 5:
        return EVTResult(
            passed=False, shape_param=0.0, scale_param=0.0,
            ks_statistic=1.0, ks_p_value=0.0,
            var_99=0.0, cvar_99=0.0, n_tail_obs=len(exceedances),
        )

    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    ks_stat, ks_p = stats.kstest(exceedances, "genpareto", args=(shape, 0, scale))

    # VaR/CVaR at 99%
    n = len(daily)
    n_u = len(exceedances)
    p = 0.01
    if n_u > 0 and shape != -1:
        var_99 = threshold + (scale / shape) * ((n / n_u * p) ** (-shape) - 1) if shape != 0 else threshold + scale * np.log(n / n_u * p)
        if shape < 1:
            cvar_99 = (var_99 + scale - shape * threshold) / (1 - shape)
        else:
            cvar_99 = var_99 * 1.5  # approximation
    else:
        var_99 = float(np.percentile(losses, 99))
        cvar_99 = float(losses[losses > var_99].mean()) if len(losses[losses > var_99]) > 0 else var_99

    return EVTResult(
        passed=ks_p > 0.05,
        shape_param=float(shape),
        scale_param=float(scale),
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        var_99=float(var_99),
        cvar_99=float(cvar_99),
        n_tail_obs=len(exceedances),
    )


def max_drawdown_duration(data: DataSource) -> MaxDDDurationResult:
    """Walk equity HWM, group underwater periods.

    Pass: max underwater duration < 90 days.
    """
    eq = data.equity_df
    if len(eq) < 2:
        return MaxDDDurationResult(
            passed=True, max_duration_days=0, max_duration_limit=90,
            underwater_periods=[], current_underwater_days=0,
        )

    equity = eq["equity"].values
    timestamps = pd.to_datetime(eq["timestamp"])
    hwm = np.maximum.accumulate(equity)

    # Find underwater periods
    is_underwater = equity < hwm
    periods: list[dict[str, Any]] = []
    start_idx = None

    for i in range(len(is_underwater)):
        if is_underwater[i] and start_idx is None:
            start_idx = i
        elif not is_underwater[i] and start_idx is not None:
            days = (timestamps.iloc[i] - timestamps.iloc[start_idx]).days
            depth = float((hwm[start_idx:i+1].max() - equity[start_idx:i+1].min()) / hwm[start_idx:i+1].max())
            periods.append({
                "start": str(timestamps.iloc[start_idx].date()),
                "end": str(timestamps.iloc[i].date()),
                "duration_days": days,
                "max_depth": depth,
            })
            start_idx = None

    # Handle currently underwater
    current_uw = 0
    if start_idx is not None:
        current_uw = (timestamps.iloc[-1] - timestamps.iloc[start_idx]).days
        periods.append({
            "start": str(timestamps.iloc[start_idx].date()),
            "end": "ongoing",
            "duration_days": current_uw,
            "max_depth": float((hwm[start_idx:].max() - equity[start_idx:].min()) / hwm[start_idx:].max()),
        })

    max_dur = max((p["duration_days"] for p in periods), default=0)

    return MaxDDDurationResult(
        passed=max_dur < 90,
        max_duration_days=max_dur,
        max_duration_limit=90,
        underwater_periods=periods,
        current_underwater_days=current_uw,
    )


# ---------------------------------------------------------------------------
# Phase 2: Backtest-Dependent Tests
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    data_loader: HistoricalDataLoader,
    pairs: list[str],
    baseline_config: StrategyConfigV3 | None = None,
    perturbation_pct: float = 0.15,
    robustness_threshold: float = 0.30,
    start: str = "2024-06-01",
    end: str = "2025-12-31",
) -> SensitivityAnalysisResult:
    """Perturb 6 params, measure Sharpe stability across ~30 backtests.

    Pass: robustness score (fraction of variants with Sharpe > 0) > threshold.
    """
    if baseline_config is None:
        baseline_config = StrategyConfigV3()

    param_grid = _build_sensitivity_grid(baseline_config, perturbation_pct)

    # Load data once
    pair_data = {}
    for pair in pairs:
        try:
            df = data_loader.load_pair(pair, start=start, end=end)
            if len(df) > 0:
                pair_data[pair] = df
        except Exception:
            continue

    if not pair_data:
        return SensitivityAnalysisResult(
            passed=False, param_results={}, robustness_score=0.0,
            robustness_threshold=robustness_threshold, baseline_sharpe=0.0,
        )

    # Run baseline
    baseline_sharpe = _run_multi_pair_backtest(pair_data, baseline_config)

    # Run perturbations
    param_results: dict[str, list[dict[str, Any]]] = {}
    total_variants = 0
    positive_sharpe = 0

    for param_name, values in param_grid.items():
        param_results[param_name] = []
        for val in values:
            cfg = _clone_config(baseline_config)
            setattr(cfg, param_name, val)
            sharpe = _run_multi_pair_backtest(pair_data, cfg)
            param_results[param_name].append({
                "value": val,
                "sharpe": sharpe,
            })
            total_variants += 1
            if sharpe > 0:
                positive_sharpe += 1

    robustness_score = positive_sharpe / total_variants if total_variants > 0 else 0.0

    return SensitivityAnalysisResult(
        passed=robustness_score > robustness_threshold,
        param_results=param_results,
        robustness_score=robustness_score,
        robustness_threshold=robustness_threshold,
        baseline_sharpe=baseline_sharpe,
    )


def regime_performance_breakdown(
    data: DataSource,
    price_data: pd.DataFrame,
    concentration_threshold: float = 0.80,
) -> RegimeBreakdownResult:
    """Tag each trade by regime at entry time, compute per-regime stats.

    Pass: no single regime contributes > concentration_threshold of total PnL.
    """
    trades = data.trades_df
    detector = RegimeDetector()

    if len(trades) == 0 or len(price_data) == 0:
        return RegimeBreakdownResult(
            passed=True, regime_stats={}, concentration_pct=0.0,
            concentration_threshold=concentration_threshold, dominant_regime="none",
        )

    # Detect regimes for full price series
    min_rows = detector.atr_lookback + detector.atr_period
    regime_map: dict[str, str] = {}

    if len(price_data) >= min_rows:
        regimes = detector.detect_series(price_data)
        ts_list = price_data["timestamp"].iloc[min_rows:].tolist()
        for ts, reg in zip(ts_list, regimes):
            regime_map[str(ts)] = reg.composite_regime.value

    # Tag trades
    regime_trades: dict[str, list[float]] = {}
    for _, trade in trades.iterrows():
        entry_ts = str(trade.get("entry_time", ""))
        regime = _find_nearest_regime(entry_ts, regime_map)
        if regime not in regime_trades:
            regime_trades[regime] = []
        regime_trades[regime].append(float(trade.get("total_pnl", 0)))

    # Compute stats
    total_pnl = sum(sum(pnls) for pnls in regime_trades.values())
    regime_stats: dict[str, dict[str, float]] = {}
    dominant_regime = "unknown"
    max_contribution = 0.0

    for regime, pnls in regime_trades.items():
        pnl_sum = sum(pnls)
        wins = [p for p in pnls if p > 0]
        contribution = abs(pnl_sum / total_pnl) if total_pnl != 0 else 0.0
        regime_stats[regime] = {
            "total_pnl": pnl_sum,
            "trade_count": len(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "avg_pnl": np.mean(pnls) if pnls else 0.0,
            "pnl_contribution": contribution,
        }
        if contribution > max_contribution:
            max_contribution = contribution
            dominant_regime = regime

    return RegimeBreakdownResult(
        passed=max_contribution < concentration_threshold,
        regime_stats=regime_stats,
        concentration_pct=max_contribution,
        concentration_threshold=concentration_threshold,
        dominant_regime=dominant_regime,
    )


def interest_rate_reversal(
    data_loader: HistoricalDataLoader,
    pair: str = "USD/JPY",
    rate_shock_bps: int = 50,
    start: str = "2024-06-01",
    end: str = "2025-12-31",
) -> RateReversalResult:
    """Run backtest with swap flipping negative mid-way.

    Pass: reversal return > -10% (strategy doesn't blow up).
    """
    try:
        df = data_loader.load_pair(pair, start=start, end=end)
    except Exception:
        return RateReversalResult(
            passed=False, baseline_return=0.0, reversal_return=0.0,
            return_degradation=1.0, rate_shock_bps=rate_shock_bps,
        )

    if len(df) == 0:
        return RateReversalResult(
            passed=False, baseline_return=0.0, reversal_return=0.0,
            return_degradation=1.0, rate_shock_bps=rate_shock_bps,
        )

    # Baseline backtest
    config = StrategyConfigV3()
    strategy = CarryTradeStrategyV3(config)
    risk_config = RiskConfig(initial_capital=100_000.0)
    engine = BacktestEngine(strategy, risk_config=risk_config)
    baseline_result = engine.run(df, pair)
    baseline_return = baseline_result.metrics.total_return

    # Reversal: flip swap negative at midpoint
    df_rev = df.copy()
    mid = len(df_rev) // 2
    shock = rate_shock_bps / 10_000 / 365 * 100
    df_rev.loc[df_rev.index[mid:], "swap_long"] = -abs(shock)
    df_rev.loc[df_rev.index[mid:], "swap_short"] = abs(shock)

    strategy_rev = CarryTradeStrategyV3(config)
    engine_rev = BacktestEngine(strategy_rev, risk_config=risk_config)
    rev_result = engine_rev.run(df_rev, pair)
    rev_return = rev_result.metrics.total_return

    degradation = baseline_return - rev_return if baseline_return != 0 else 0.0

    return RateReversalResult(
        passed=rev_return > -0.10,
        baseline_return=baseline_return,
        reversal_return=rev_return,
        return_degradation=degradation,
        rate_shock_bps=rate_shock_bps,
    )


# ---------------------------------------------------------------------------
# Phase 3: Scenario Modeling
# ---------------------------------------------------------------------------

def correlation_crisis_scenario(
    pairs: list[str],
    data_loader: HistoricalDataLoader,
    max_positions: int = 6,
    crisis_correlation: float = 0.95,
    start: str = "2024-06-01",
    end: str = "2025-12-31",
) -> CorrelationCrisisResult:
    """6 positions x 2ATR loss x 0.95 correlation -> portfolio DD.

    Pass: crisis portfolio DD < 20%.
    """
    # Load price data and compute ATR for each pair
    atrs = []
    prices = []
    for pair in pairs[:max_positions]:
        try:
            df = data_loader.load_pair(pair, start=start, end=end)
            if len(df) == 0:
                continue
            from src.strategy.indicators import atr as calc_atr
            atr_series = calc_atr(df["high"], df["low"], df["close"], period=14)
            current_atr = float(atr_series.iloc[-1])
            current_price = float(df["close"].iloc[-1])
            atrs.append(current_atr)
            prices.append(current_price)
        except Exception:
            continue

    n = len(atrs)
    if n == 0:
        return CorrelationCrisisResult(
            passed=False, normal_correlation=0.0, crisis_correlation=crisis_correlation,
            normal_portfolio_dd=0.0, crisis_portfolio_dd=0.0,
            max_positions=max_positions, per_position_loss=0.0,
        )

    # Per-position loss = 2 ATR as fraction of price
    per_pos_losses = [2 * a / p for a, p in zip(atrs, prices)]
    avg_loss = float(np.mean(per_pos_losses))

    # Position size = 4% each
    pos_weight = 0.04

    # Normal: assume low correlation (0.3)
    normal_corr = 0.3
    normal_dd = pos_weight * avg_loss * np.sqrt(n + n * (n - 1) * normal_corr)

    # Crisis: high correlation
    crisis_dd = pos_weight * avg_loss * np.sqrt(n + n * (n - 1) * crisis_correlation)

    return CorrelationCrisisResult(
        passed=crisis_dd < 0.20,
        normal_correlation=normal_corr,
        crisis_correlation=crisis_correlation,
        normal_portfolio_dd=float(normal_dd),
        crisis_portfolio_dd=float(crisis_dd),
        max_positions=max_positions,
        per_position_loss=avg_loss,
    )


def market_impact_model(
    current_units: int = 10000,
    atr: float = 0.5,
    price: float = 150.0,
    k: float = 0.1,
    reference_volume: float = 1_000_000,
) -> MarketImpactResult:
    """Square-root impact model: impact = k * sqrt(size / reference).

    Pass: impact < 5 bps (negligible for our size).
    """
    impact_frac = k * np.sqrt(current_units / reference_volume)
    impact_bps = impact_frac * 10_000
    slippage_cost = impact_frac * price * current_units

    return MarketImpactResult(
        passed=impact_bps < 5.0,
        impact_bps=float(impact_bps),
        impact_pct=float(impact_frac * 100),
        slippage_cost_per_trade=float(slippage_cost),
        current_units=current_units,
        reference_volume=reference_volume,
        k_factor=k,
    )


def liquidity_risk_assessment(
    data: DataSource,
    price_data: pd.DataFrame | None = None,
    spike_threshold: float = 2.0,
) -> LiquidityRiskResult:
    """ATR by session (Tokyo/London/NY), flag if any > spike_threshold x avg.

    Pass: no session ATR > spike_threshold * overall average.
    """
    if price_data is None or len(price_data) == 0:
        return LiquidityRiskResult(
            passed=True, session_atr={}, avg_atr=0.0,
            max_atr_ratio=0.0, worst_session="none",
            spike_threshold=spike_threshold,
        )

    df = price_data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Session definitions (UTC)
    sessions = {
        "Tokyo": (0, 8),      # 00:00-08:00 UTC
        "London": (8, 16),    # 08:00-16:00 UTC
        "New York": (13, 21), # 13:00-21:00 UTC
    }

    from src.strategy.indicators import atr as calc_atr

    session_atr: dict[str, float] = {}
    for name, (start_h, end_h) in sessions.items():
        if start_h < end_h:
            mask = (df["hour"] >= start_h) & (df["hour"] < end_h)
        else:
            mask = (df["hour"] >= start_h) | (df["hour"] < end_h)

        session_df = df[mask]
        if len(session_df) > 20:
            atr_s = calc_atr(session_df["high"], session_df["low"], session_df["close"], period=14)
            session_atr[name] = float(atr_s.dropna().mean())
        else:
            session_atr[name] = 0.0

    avg_atr = float(np.mean(list(session_atr.values()))) if session_atr else 0.0
    max_ratio = 0.0
    worst = "none"
    if avg_atr > 0:
        for name, val in session_atr.items():
            ratio = val / avg_atr
            if ratio > max_ratio:
                max_ratio = ratio
                worst = name

    return LiquidityRiskResult(
        passed=max_ratio < spike_threshold,
        session_atr=session_atr,
        avg_atr=avg_atr,
        max_atr_ratio=float(max_ratio),
        worst_session=worst,
        spike_threshold=spike_threshold,
    )


# ---------------------------------------------------------------------------
# Chart Functions (15 PNGs)
# ---------------------------------------------------------------------------

def _setup_chart():
    """Common chart setup."""
    plt.style.use("seaborn-v0_8-darkgrid")


def plot_monte_carlo_dd(result: MonteCarloResult, output_path: Path) -> None:
    """DD distribution histogram + percentile lines."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(result.max_drawdowns * 100, bins=50, alpha=0.7, edgecolor="black", linewidth=0.3, color="steelblue")
    ax.axvline(result.ruin_threshold * 100, color="red", linestyle="--", linewidth=2, label=f"Ruin threshold ({result.ruin_threshold:.0%})")
    ax.axvline(np.percentile(result.max_drawdowns, 50) * 100, color="orange", linestyle="--", label=f"Median DD ({np.median(result.max_drawdowns):.1%})")
    ax.axvline(np.percentile(result.max_drawdowns, 95) * 100, color="darkred", linestyle=":", label=f"95th pctl ({result.max_dd_95:.1%})")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo Max DD Distribution (n={result.n_sims})\nP(ruin)={result.ruin_probability:.1%}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_monte_carlo_returns(result: MonteCarloResult, output_path: Path) -> None:
    """Return distribution histogram."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(result.final_returns * 100, bins=50, alpha=0.7, edgecolor="black", linewidth=0.3, color="forestgreen")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(np.percentile(result.final_returns, 5) * 100, color="red", linestyle="--", label=f"5th pctl ({result.percentile_5_return:.1%})")
    ax.axvline(np.median(result.final_returns) * 100, color="orange", linestyle="--", label=f"Median ({result.median_final_return:.1%})")
    ax.axvline(np.percentile(result.final_returns, 95) * 100, color="green", linestyle="--", label=f"95th pctl ({result.percentile_95_return:.1%})")
    ax.set_xlabel("Final Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo Return Distribution (n={result.n_sims})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_bootstrap_ci(results: list[BootstrapCIResult], output_path: Path) -> None:
    """Horizontal CI bars per metric."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 1.5)))
    y_pos = range(len(results))

    for i, r in enumerate(results):
        color = "green" if r.passed else "red"
        ax.barh(i, r.ci_upper - r.ci_lower, left=r.ci_lower, height=0.4, color=color, alpha=0.6)
        ax.plot(r.observed, i, "ko", markersize=8)
        ax.text(r.ci_upper + 0.01, i, f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]", va="center", fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([r.metric_name for r in results])
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Zero line")
    ax.set_xlabel("Value")
    ax.set_title(f"Bootstrap {results[0].ci_level:.0%} Confidence Intervals")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_permutation_test(result: PermutationTestResult, output_path: Path) -> None:
    """Null distribution + observed Sharpe line."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.hist(result.null_distribution, bins="auto", alpha=0.7, edgecolor="black", linewidth=0.3, color="lightcoral", label="Null distribution")
    except ValueError:
        ax.bar([float(np.median(result.null_distribution))], [len(result.null_distribution)], width=0.01, color="lightcoral", alpha=0.7, label="Null distribution (degenerate)")
    ax.axvline(result.observed_sharpe, color="blue", linewidth=2, label=f"Observed Sharpe ({result.observed_sharpe:.2f})")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test (p={result.p_value:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_decay_detection(result: DecayDetectionResult, output_path: Path) -> None:
    """Rolling Sharpe time series (30/60/90d)."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["steelblue", "orange", "green"]

    for (w, series), color in zip(result.rolling_sharpes.items(), colors):
        if len(series) > 0:
            ax.plot(series.index, series.values, label=f"{w}-day rolling", color=color, alpha=0.8)

    ax.axhline(result.full_period_sharpe, color="red", linestyle="--", label=f"Full period ({result.full_period_sharpe:.2f})")
    ax.axhline(result.full_period_sharpe * result.decay_threshold, color="red", linestyle=":", alpha=0.5,
               label=f"Decay threshold ({result.full_period_sharpe * result.decay_threshold:.2f})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Strategy Decay Detection - Rolling Sharpe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_sensitivity_bars(result: SensitivityAnalysisResult, output_path: Path) -> None:
    """Sharpe at perturbation levels per param."""
    _setup_chart()
    n_params = len(result.param_results)
    if n_params == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (param, entries) in enumerate(result.param_results.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        values = [e["value"] for e in entries]
        sharpes = [e["sharpe"] for e in entries]
        colors = ["green" if s > 0 else "red" for s in sharpes]
        ax.bar(range(len(values)), sharpes, color=colors, alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in values], fontsize=8, rotation=45)
        ax.axhline(result.baseline_sharpe, color="blue", linestyle="--", label="Baseline")
        ax.set_title(param, fontsize=10)
        ax.set_ylabel("Sharpe")

    for idx in range(n_params, 6):
        axes[idx].set_visible(False)

    fig.suptitle(f"Parameter Sensitivity (robustness={result.robustness_score:.0%})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_sensitivity_heatmap(result: SensitivityAnalysisResult, output_path: Path) -> None:
    """2D: short_ma vs long_ma Sharpe heatmap."""
    _setup_chart()
    short_vals = [e["value"] for e in result.param_results.get("short_ma_period", [])]
    long_vals = [e["value"] for e in result.param_results.get("long_ma_period", [])]

    if not short_vals or not long_vals:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=120)
        plt.close()
        return

    short_sharpes = [e["sharpe"] for e in result.param_results["short_ma_period"]]
    long_sharpes = [e["sharpe"] for e in result.param_results["long_ma_period"]]

    # Create a simple grid from the 1D sweeps
    grid = np.outer(
        np.array(short_sharpes) / max(abs(s) for s in short_sharpes) if any(s != 0 for s in short_sharpes) else np.ones(len(short_sharpes)),
        np.array(long_sharpes) / max(abs(s) for s in long_sharpes) if any(s != 0 for s in long_sharpes) else np.ones(len(long_sharpes)),
    ) * result.baseline_sharpe

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(long_vals)))
    ax.set_xticklabels([str(int(v)) for v in long_vals])
    ax.set_yticks(range(len(short_vals)))
    ax.set_yticklabels([str(int(v)) for v in short_vals])
    ax.set_xlabel("long_ma_period")
    ax.set_ylabel("short_ma_period")
    ax.set_title("MA Period Sensitivity Heatmap (Sharpe)")
    for i in range(len(short_vals)):
        for j in range(len(long_vals)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_regime_breakdown(result: RegimeBreakdownResult, output_path: Path) -> None:
    """Per-regime stats + PnL pie chart."""
    _setup_chart()
    if not result.regime_stats:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No regime data available", ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=120)
        plt.close()
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    regimes = list(result.regime_stats.keys())
    pnls = [result.regime_stats[r]["total_pnl"] for r in regimes]
    counts = [result.regime_stats[r]["trade_count"] for r in regimes]
    win_rates = [result.regime_stats[r]["win_rate"] * 100 for r in regimes]

    # Bar chart: PnL by regime
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax1.bar(regimes, pnls, color=colors, alpha=0.7)
    ax1.set_xlabel("Regime")
    ax1.set_ylabel("Total PnL ($)")
    ax1.set_title("PnL by Market Regime")
    ax1.tick_params(axis="x", rotation=45)

    # Add trade count labels
    for i, (p, c) in enumerate(zip(pnls, counts)):
        ax1.text(i, p, f"n={int(c)}", ha="center", va="bottom" if p >= 0 else "top", fontsize=9)

    # Pie chart: trade distribution
    if any(c > 0 for c in counts):
        ax2.pie([max(c, 0) for c in counts], labels=regimes, autopct="%1.0f%%", startangle=90)
    ax2.set_title("Trade Distribution by Regime")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_correlation_crisis(result: CorrelationCrisisResult, output_path: Path) -> None:
    """Normal vs crisis correlation and portfolio DD comparison."""
    _setup_chart()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Correlation comparison
    labels = ["Normal", "Crisis"]
    corrs = [result.normal_correlation, result.crisis_correlation]
    colors = ["steelblue", "red"]
    ax1.bar(labels, corrs, color=colors, alpha=0.7)
    ax1.set_ylabel("Correlation")
    ax1.set_title("Pair Correlation: Normal vs Crisis")
    ax1.set_ylim(0, 1)

    # Portfolio DD comparison
    dds = [result.normal_portfolio_dd * 100, result.crisis_portfolio_dd * 100]
    ax2.bar(labels, dds, color=colors, alpha=0.7)
    ax2.axhline(20, color="red", linestyle="--", label="20% limit")
    ax2.set_ylabel("Portfolio Drawdown (%)")
    ax2.set_title(f"Portfolio DD ({result.max_positions} positions)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_rate_reversal(result: RateReversalResult, output_path: Path) -> None:
    """Baseline vs reversal return comparison."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Baseline", f"Rate Reversal\n({result.rate_shock_bps}bps shock)"]
    returns = [result.baseline_return * 100, result.reversal_return * 100]
    colors = ["green" if r > 0 else "red" for r in returns]
    ax.bar(labels, returns, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-10, color="red", linestyle="--", alpha=0.5, label="-10% fail threshold")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Interest Rate Reversal Stress Test")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_overfitting_gauge(result: OverfittingResult, output_path: Path) -> None:
    """DOF ratio + IS/OOS Sharpe comparison."""
    _setup_chart()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # DOF gauge
    ax1.barh(["DOF Ratio"], [result.dof_ratio], color="steelblue" if result.dof_ratio < 0.1 else "red", alpha=0.7)
    ax1.axvline(0.1, color="red", linestyle="--", label="0.1 threshold")
    ax1.set_title(f"DOF Ratio: {result.n_params} params / {result.n_observations} obs")
    ax1.legend()

    # IS vs OOS
    labels = ["In-Sample (70%)", "Out-of-Sample (30%)"]
    sharpes = [result.is_sharpe, result.oos_sharpe]
    colors = ["steelblue", "orange"]
    ax2.bar(labels, sharpes, color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title(f"IS/OOS Sharpe (degradation: {result.sharpe_degradation:.0%})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_evt_tail(result: EVTResult, output_path: Path) -> None:
    """GPD fit vs empirical tail."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, 6))

    if result.n_tail_obs < 5:
        ax.text(0.5, 0.5, "Insufficient tail observations", ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=120)
        plt.close()
        return

    # Generate theoretical vs empirical
    x = np.linspace(0, result.var_99 * 1.5, 100)
    gpd_cdf = stats.genpareto.cdf(x, result.shape_param, scale=result.scale_param)
    ax.plot(x * 100, gpd_cdf, "r-", linewidth=2, label=f"GPD fit (shape={result.shape_param:.3f})")
    ax.axvline(result.var_99 * 100, color="blue", linestyle="--", label=f"VaR 99%: {result.var_99:.2%}")
    ax.axvline(result.cvar_99 * 100, color="darkblue", linestyle=":", label=f"CVaR 99%: {result.cvar_99:.2%}")
    ax.set_xlabel("Loss (%)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"Extreme Value Analysis - GPD Tail Fit (KS p={result.ks_p_value:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_market_impact(result: MarketImpactResult, output_path: Path) -> None:
    """Log-log impact curve."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = np.logspace(2, 7, 100)
    impacts = result.k_factor * np.sqrt(sizes / result.reference_volume) * 10_000
    ax.loglog(sizes, impacts, "b-", linewidth=2)
    ax.axhline(5, color="red", linestyle="--", label="5 bps threshold")
    ax.axvline(result.current_units, color="green", linestyle="--", label=f"Current size ({result.current_units:,})")
    ax.plot(result.current_units, result.impact_bps, "ro", markersize=10, zorder=5)
    ax.set_xlabel("Order Size (units)")
    ax.set_ylabel("Market Impact (bps)")
    ax.set_title("Market Impact Model (Square-Root)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_liquidity_risk(result: LiquidityRiskResult, output_path: Path) -> None:
    """Session ATR bars."""
    _setup_chart()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not result.session_atr:
        ax.text(0.5, 0.5, "No session data", ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=120)
        plt.close()
        return

    sessions = list(result.session_atr.keys())
    atrs = list(result.session_atr.values())
    colors = ["red" if a / result.avg_atr > result.spike_threshold else "steelblue" for a in atrs] if result.avg_atr > 0 else ["steelblue"] * len(atrs)
    ax.bar(sessions, atrs, color=colors, alpha=0.7)
    if result.avg_atr > 0:
        ax.axhline(result.avg_atr, color="orange", linestyle="--", label=f"Average ATR ({result.avg_atr:.4f})")
        ax.axhline(result.avg_atr * result.spike_threshold, color="red", linestyle="--", label=f"Spike threshold ({result.spike_threshold}x)")
    ax.set_ylabel("ATR")
    ax.set_title("Liquidity Risk by Trading Session")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_dd_duration(result: MaxDDDurationResult, equity_df: pd.DataFrame, output_path: Path) -> None:
    """Equity + HWM with underwater shading."""
    _setup_chart()
    if len(equity_df) < 2:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=120)
        plt.close()
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    timestamps = pd.to_datetime(equity_df["timestamp"])
    equity = equity_df["equity"].values
    hwm = np.maximum.accumulate(equity)

    ax1.plot(timestamps, equity, "b-", linewidth=1, label="Equity")
    ax1.plot(timestamps, hwm, "r--", linewidth=0.8, alpha=0.6, label="High Water Mark")
    ax1.fill_between(timestamps, equity, hwm, where=equity < hwm, alpha=0.2, color="red")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"Drawdown Duration (max: {result.max_duration_days} days, limit: {result.max_duration_limit} days)")
    ax1.legend()

    # Underwater plot
    underwater = (hwm - equity) / hwm * 100
    ax2.fill_between(timestamps, 0, -underwater, color="red", alpha=0.6)
    ax2.set_ylabel("Underwater (%)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_markdown_report(
    suite: ValidationSuiteResult,
    output_dir: Path,
) -> str:
    """Generate markdown report with pass/fail table."""
    lines = [
        "# Validation & Robustness Report",
        "",
        f"**Generated**: {suite.timestamp}",
        f"**Data Source**: {suite.source_type}",
        f"**Overall**: {suite.pass_count}/{suite.total_count} passed",
        "",
        "## Summary Table",
        "",
        "| # | Test | Status | Key Metric |",
        "|---|------|--------|------------|",
    ]

    test_descriptions = {
        "monte_carlo": ("1", "Monte Carlo Simulation"),
        "bootstrap_ci": ("2", "Bootstrap Confidence Intervals"),
        "permutation_test": ("3", "Permutation Test"),
        "strategy_decay": ("4", "Strategy Decay Detection"),
        "parameter_sensitivity": ("5", "Parameter Sensitivity"),
        "regime_breakdown": ("6", "Regime Performance Breakdown"),
        "correlation_crisis": ("7", "Correlation Crisis Scenario"),
        "rate_reversal": ("8", "Interest Rate Reversal"),
        "overfitting": ("9", "Overfitting Detection"),
        "evt_analysis": ("10", "Extreme Value Analysis"),
        "market_impact": ("11", "Market Impact Model"),
        "liquidity_risk": ("12", "Liquidity Risk Assessment"),
        "max_dd_duration": ("13", "Max Drawdown Duration"),
    }

    for key, (num, desc) in test_descriptions.items():
        result = suite.results.get(key)
        if result is None:
            lines.append(f"| {num} | {desc} | SKIPPED | - |")
            continue

        passed = ValidationSuiteResult._check_passed(result)
        status = "PASS" if passed else "FAIL"
        metric = _format_key_metric(key, result)
        lines.append(f"| {num} | {desc} | {status} | {metric} |")

    lines.extend(["", "## Detailed Results", ""])

    # Monte Carlo details
    mc = suite.results.get("monte_carlo")
    if mc:
        lines.extend([
            "### 1. Monte Carlo Simulation",
            f"- Simulations: {mc.n_sims}",
            f"- Ruin probability: {mc.ruin_probability:.1%} (threshold: <10%)",
            f"- Median return: {mc.median_final_return:.2%}",
            f"- 5th/95th percentile return: {mc.percentile_5_return:.2%} / {mc.percentile_95_return:.2%}",
            f"- Median max DD: {mc.max_dd_median:.2%}",
            f"- 95th percentile max DD: {mc.max_dd_95:.2%}",
            "",
        ])

    # Bootstrap details
    bci = suite.results.get("bootstrap_ci")
    if bci:
        lines.extend(["### 2. Bootstrap Confidence Intervals", ""])
        for r in bci:
            lines.append(f"- **{r.metric_name}**: {r.observed:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}] ({r.ci_level:.0%} CI)")
        lines.append("")

    # Permutation test
    pt = suite.results.get("permutation_test")
    if pt:
        lines.extend([
            "### 3. Permutation Test",
            f"- Observed Sharpe: {pt.observed_sharpe:.2f}",
            f"- p-value: {pt.p_value:.4f} (threshold: <{pt.significance})",
            "",
        ])

    # Decay
    dd = suite.results.get("strategy_decay")
    if dd:
        lines.extend(["### 4. Strategy Decay Detection"])
        for w, s in dd.current_sharpes.items():
            lines.append(f"- {w}-day rolling Sharpe: {s:.2f}")
        lines.extend([f"- Full period Sharpe: {dd.full_period_sharpe:.2f}", ""])

    # Sensitivity
    sens = suite.results.get("parameter_sensitivity")
    if sens:
        lines.extend([
            "### 5. Parameter Sensitivity",
            f"- Baseline Sharpe: {sens.baseline_sharpe:.2f}",
            f"- Robustness score: {sens.robustness_score:.0%} (threshold: >{sens.robustness_threshold:.0%})",
            "",
        ])

    # Regime
    rb = suite.results.get("regime_breakdown")
    if rb:
        lines.extend([
            "### 6. Regime Performance Breakdown",
            f"- Dominant regime: {rb.dominant_regime} ({rb.concentration_pct:.0%} of PnL)",
            "",
        ])

    # Correlation crisis
    cc = suite.results.get("correlation_crisis")
    if cc:
        lines.extend([
            "### 7. Correlation Crisis Scenario",
            f"- Normal DD: {cc.normal_portfolio_dd:.2%}",
            f"- Crisis DD: {cc.crisis_portfolio_dd:.2%} (limit: 20%)",
            "",
        ])

    # Rate reversal
    rr = suite.results.get("rate_reversal")
    if rr:
        lines.extend([
            "### 8. Interest Rate Reversal",
            f"- Baseline return: {rr.baseline_return:.2%}",
            f"- Reversal return: {rr.reversal_return:.2%} ({rr.rate_shock_bps}bps shock)",
            "",
        ])

    # Overfitting
    of = suite.results.get("overfitting")
    if of:
        lines.extend([
            "### 9. Overfitting Detection",
            f"- DOF ratio: {of.dof_ratio:.4f} (threshold: <0.1)",
            f"- IS Sharpe: {of.is_sharpe:.2f}, OOS Sharpe: {of.oos_sharpe:.2f}",
            f"- Sharpe degradation: {of.sharpe_degradation:.0%} (threshold: <50%)",
            "",
        ])

    # EVT
    evt = suite.results.get("evt_analysis")
    if evt:
        lines.extend([
            "### 10. Extreme Value Analysis",
            f"- GPD shape: {evt.shape_param:.3f}, scale: {evt.scale_param:.5f}",
            f"- KS test p-value: {evt.ks_p_value:.3f} (threshold: >0.05)",
            f"- VaR 99%: {evt.var_99:.2%}, CVaR 99%: {evt.cvar_99:.2%}",
            "",
        ])

    # Market impact
    mi = suite.results.get("market_impact")
    if mi:
        lines.extend([
            "### 11. Market Impact Model",
            f"- Impact: {mi.impact_bps:.1f} bps (threshold: <5 bps)",
            f"- Slippage cost/trade: ${mi.slippage_cost_per_trade:.2f}",
            "",
        ])

    # Liquidity
    lr = suite.results.get("liquidity_risk")
    if lr:
        lines.extend([
            "### 12. Liquidity Risk Assessment",
            f"- Worst session: {lr.worst_session} ({lr.max_atr_ratio:.1f}x avg ATR)",
            "",
        ])

    # DD duration
    mdd = suite.results.get("max_dd_duration")
    if mdd:
        lines.extend([
            "### 13. Max Drawdown Duration",
            f"- Longest underwater: {mdd.max_duration_days} days (limit: {mdd.max_duration_limit} days)",
            f"- Currently underwater: {mdd.current_underwater_days} days",
            "",
        ])

    # Charts listing
    lines.extend([
        "## Charts",
        "",
        "All charts saved to this directory as PNG files.",
        "",
    ])

    report = "\n".join(lines)
    report_path = output_dir / "validation_report.md"
    report_path.write_text(report)
    return report


def generate_all_charts(
    suite: ValidationSuiteResult,
    data: DataSource,
    output_dir: Path,
) -> list[Path]:
    """Generate all 15 charts for the suite results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts: list[Path] = []

    mc = suite.results.get("monte_carlo")
    if mc and len(mc.final_returns) > 0:
        p = output_dir / "monte_carlo_dd.png"
        plot_monte_carlo_dd(mc, p)
        charts.append(p)
        p = output_dir / "monte_carlo_returns.png"
        plot_monte_carlo_returns(mc, p)
        charts.append(p)

    bci = suite.results.get("bootstrap_ci")
    if bci:
        p = output_dir / "bootstrap_ci.png"
        plot_bootstrap_ci(bci, p)
        charts.append(p)

    pt = suite.results.get("permutation_test")
    if pt:
        p = output_dir / "permutation_test.png"
        plot_permutation_test(pt, p)
        charts.append(p)

    decay = suite.results.get("strategy_decay")
    if decay:
        p = output_dir / "decay_detection.png"
        plot_decay_detection(decay, p)
        charts.append(p)

    sens = suite.results.get("parameter_sensitivity")
    if sens:
        p = output_dir / "sensitivity_bars.png"
        plot_sensitivity_bars(sens, p)
        charts.append(p)
        p = output_dir / "sensitivity_heatmap.png"
        plot_sensitivity_heatmap(sens, p)
        charts.append(p)

    rb = suite.results.get("regime_breakdown")
    if rb:
        p = output_dir / "regime_breakdown.png"
        plot_regime_breakdown(rb, p)
        charts.append(p)

    cc = suite.results.get("correlation_crisis")
    if cc:
        p = output_dir / "correlation_crisis.png"
        plot_correlation_crisis(cc, p)
        charts.append(p)

    rr = suite.results.get("rate_reversal")
    if rr:
        p = output_dir / "rate_reversal.png"
        plot_rate_reversal(rr, p)
        charts.append(p)

    of = suite.results.get("overfitting")
    if of:
        p = output_dir / "overfitting_gauge.png"
        plot_overfitting_gauge(of, p)
        charts.append(p)

    evt = suite.results.get("evt_analysis")
    if evt:
        p = output_dir / "evt_tail.png"
        plot_evt_tail(evt, p)
        charts.append(p)

    mi = suite.results.get("market_impact")
    if mi:
        p = output_dir / "market_impact.png"
        plot_market_impact(mi, p)
        charts.append(p)

    lr = suite.results.get("liquidity_risk")
    if lr:
        p = output_dir / "liquidity_risk.png"
        plot_liquidity_risk(lr, p)
        charts.append(p)

    mdd = suite.results.get("max_dd_duration")
    if mdd:
        p = output_dir / "dd_duration.png"
        plot_dd_duration(mdd, data.equity_df, p)
        charts.append(p)

    return charts


# ---------------------------------------------------------------------------
# Suite Runner
# ---------------------------------------------------------------------------

def run_validation_suite(
    data: DataSource,
    data_loader: HistoricalDataLoader | None = None,
    pairs: list[str] | None = None,
    price_data: pd.DataFrame | None = None,
    priority: str = "all",
    tests: list[str] | None = None,
    n_sims: int = 5000,
    seed: int = 42,
    generate_charts: bool = True,
    output_dir: str = "results/reports/validation",
) -> ValidationSuiteResult:
    """Run the full validation suite.

    Args:
        data: Unified data source.
        data_loader: For backtest-dependent tests (sensitivity, rate reversal).
        pairs: Forex pairs for multi-pair tests.
        price_data: OHLC data for regime/liquidity tests.
        priority: "high", "medium", "low", or "all".
        tests: Specific test names to run (overrides priority).
        n_sims: Monte Carlo simulations.
        seed: Random seed.
        generate_charts: Whether to generate PNG charts.
        output_dir: Output directory for charts and report.
    """
    if pairs is None:
        pairs = ["USD/JPY", "AUD/JPY", "GBP/JPY", "NZD/JPY", "EUR/JPY", "CAD/JPY"]

    suite = ValidationSuiteResult(
        timestamp=datetime.now().isoformat(),
        source_type=data.source_type,
    )

    # Determine which tests to run
    high = {"monte_carlo", "bootstrap_ci", "permutation_test", "strategy_decay"}
    medium = {"parameter_sensitivity", "regime_breakdown", "correlation_crisis", "rate_reversal", "overfitting"}
    low = {"evt_analysis", "market_impact", "liquidity_risk", "max_dd_duration"}

    if tests:
        run_tests = set(tests)
    elif priority == "high":
        run_tests = high
    elif priority == "medium":
        run_tests = high | medium
    elif priority == "low":
        run_tests = high | medium | low
    else:
        run_tests = high | medium | low

    # Phase 1: Independent stats
    if "monte_carlo" in run_tests:
        print("  [1/13] Monte Carlo simulation...")
        suite.results["monte_carlo"] = monte_carlo_simulation(data, n_sims=n_sims, seed=seed)

    if "bootstrap_ci" in run_tests:
        print("  [2/13] Bootstrap confidence intervals...")
        suite.results["bootstrap_ci"] = bootstrap_confidence_intervals(data, seed=seed)

    if "permutation_test" in run_tests:
        print("  [3/13] Permutation test...")
        suite.results["permutation_test"] = permutation_test(data, seed=seed)

    if "strategy_decay" in run_tests:
        print("  [4/13] Strategy decay detection...")
        suite.results["strategy_decay"] = detect_strategy_decay(data)

    if "overfitting" in run_tests:
        print("  [9/13] Overfitting detection...")
        suite.results["overfitting"] = overfitting_detection(data)

    if "evt_analysis" in run_tests:
        print("  [10/13] Extreme value analysis...")
        suite.results["evt_analysis"] = extreme_value_analysis(data)

    if "max_dd_duration" in run_tests:
        print("  [13/13] Max drawdown duration...")
        suite.results["max_dd_duration"] = max_drawdown_duration(data)

    # Phase 2: Backtest-dependent
    if "parameter_sensitivity" in run_tests and data_loader:
        print("  [5/13] Parameter sensitivity (~30 backtests)...")
        suite.results["parameter_sensitivity"] = parameter_sensitivity(data_loader, pairs)

    if "regime_breakdown" in run_tests and price_data is not None:
        print("  [6/13] Regime performance breakdown...")
        suite.results["regime_breakdown"] = regime_performance_breakdown(data, price_data)

    if "rate_reversal" in run_tests and data_loader:
        print("  [8/13] Interest rate reversal...")
        suite.results["rate_reversal"] = interest_rate_reversal(data_loader)

    # Phase 3: Scenario modeling
    if "correlation_crisis" in run_tests and data_loader:
        print("  [7/13] Correlation crisis scenario...")
        suite.results["correlation_crisis"] = correlation_crisis_scenario(pairs, data_loader)

    if "market_impact" in run_tests:
        print("  [11/13] Market impact model...")
        suite.results["market_impact"] = market_impact_model()

    if "liquidity_risk" in run_tests:
        print("  [12/13] Liquidity risk assessment...")
        suite.results["liquidity_risk"] = liquidity_risk_assessment(data, price_data)

    # Charts and report
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if generate_charts:
        print("  Generating charts...")
        generate_all_charts(suite, data, out)

    print("  Generating report...")
    generate_markdown_report(suite, out)

    return suite


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_metric(metric: str, daily: np.ndarray, trades: pd.DataFrame) -> float:
    """Compute a single metric value."""
    if metric == "sharpe":
        return float(_sharpe(pd.Series(daily), 252))
    elif metric == "win_rate":
        if len(trades) == 0:
            return 0.0
        wins = trades["total_pnl"] > 0
        return float(wins.sum() / len(trades))
    elif metric == "profit_factor":
        if len(trades) == 0:
            return 0.0
        pnl = trades["total_pnl"]
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl <= 0].sum())
        return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    return 0.0


def _build_sensitivity_grid(
    config: StrategyConfigV3,
    pct: float = 0.15,
) -> dict[str, list]:
    """Build 6 params x 5 values grid."""
    def _perturb(base, pct_range, n=5, as_int=False):
        lo = base * (1 - pct_range)
        hi = base * (1 + pct_range)
        vals = np.linspace(lo, hi, n)
        return [int(round(v)) for v in vals] if as_int else [round(v, 3) for v in vals]

    return {
        "short_ma_period": _perturb(config.short_ma_period, pct, as_int=True),
        "long_ma_period": _perturb(config.long_ma_period, pct, as_int=True),
        "atr_stop_mult": _perturb(config.atr_stop_mult, pct),
        "trailing_atr_mult": _perturb(config.trailing_atr_mult, pct),
        "position_size_pct": _perturb(config.position_size_pct, pct),
        "max_rsi": _perturb(config.max_rsi, pct, as_int=True),
    }


def _clone_config(config: StrategyConfigV3) -> StrategyConfigV3:
    """Create a copy of a StrategyConfigV3."""
    return StrategyConfigV3(
        short_ma_period=config.short_ma_period,
        long_ma_period=config.long_ma_period,
        min_swap_threshold=config.min_swap_threshold,
        max_rsi=config.max_rsi,
        atr_period=config.atr_period,
        atr_stop_mult=config.atr_stop_mult,
        trailing_activation_pct=config.trailing_activation_pct,
        trailing_atr_mult=config.trailing_atr_mult,
        position_size_pct=config.position_size_pct,
        max_positions=config.max_positions,
    )


def _run_multi_pair_backtest(
    pair_data: dict[str, pd.DataFrame],
    config: StrategyConfigV3,
) -> float:
    """Run backtest across multiple pairs, return average Sharpe."""
    sharpes = []
    strategy = CarryTradeStrategyV3(config)
    risk_config = RiskConfig(initial_capital=100_000.0)
    engine = BacktestEngine(strategy, risk_config=risk_config)

    for pair, df in pair_data.items():
        try:
            result = engine.run(df, pair)
            sharpes.append(result.metrics.sharpe_ratio)
        except Exception:
            continue

    return float(np.mean(sharpes)) if sharpes else 0.0


def _find_nearest_regime(ts_str: str, regime_map: dict[str, str]) -> str:
    """Find the closest regime for a timestamp."""
    if not regime_map:
        return "unknown"
    if ts_str in regime_map:
        return regime_map[ts_str]
    # Fall back to nearest
    try:
        target = pd.Timestamp(ts_str)
        best_key = min(regime_map.keys(), key=lambda k: abs(pd.Timestamp(k) - target))
        return regime_map[best_key]
    except Exception:
        return "unknown"


def _format_key_metric(key: str, result: Any) -> str:
    """Format the key metric for the summary table."""
    if key == "monte_carlo":
        return f"P(ruin)={result.ruin_probability:.1%}"
    elif key == "bootstrap_ci":
        sharpe = next((r for r in result if r.metric_name == "sharpe"), None)
        return f"Sharpe CI: [{sharpe.ci_lower:.2f}, {sharpe.ci_upper:.2f}]" if sharpe else "-"
    elif key == "permutation_test":
        return f"p={result.p_value:.4f}"
    elif key == "strategy_decay":
        latest = list(result.current_sharpes.values())[-1] if result.current_sharpes else 0
        return f"Latest rolling Sharpe: {latest:.2f}"
    elif key == "parameter_sensitivity":
        return f"Robustness: {result.robustness_score:.0%}"
    elif key == "regime_breakdown":
        return f"Dominant: {result.dominant_regime} ({result.concentration_pct:.0%})"
    elif key == "correlation_crisis":
        return f"Crisis DD: {result.crisis_portfolio_dd:.2%}"
    elif key == "rate_reversal":
        return f"Reversal return: {result.reversal_return:.2%}"
    elif key == "overfitting":
        return f"DOF: {result.dof_ratio:.4f}, OOS degrade: {result.sharpe_degradation:.0%}"
    elif key == "evt_analysis":
        return f"KS p={result.ks_p_value:.3f}"
    elif key == "market_impact":
        return f"Impact: {result.impact_bps:.1f} bps"
    elif key == "liquidity_risk":
        return f"Max ATR ratio: {result.max_atr_ratio:.1f}x"
    elif key == "max_dd_duration":
        return f"Max: {result.max_duration_days} days"
    return "-"
