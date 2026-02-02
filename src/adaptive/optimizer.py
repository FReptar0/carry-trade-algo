"""Bayesian parameter optimizer using Optuna.

Optimizes regime parameters by running walk-forward backtests on
recent candle data, maximizing Sharpe ratio. Uses the existing
BacktestEngine and CarryTradeStrategyV3.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.adaptive.param_store import ParamStore
from src.backtest.engine import BacktestEngine
from src.config.settings import RiskConfig
from src.regime.adapters import RegimeAdaptedParams
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3

logger = logging.getLogger(__name__)


class ParamOptimizer:
    """Bayesian optimizer for regime-adapted parameters.

    Uses Optuna to search the parameter space and evaluate each
    candidate via backtest on recent candle data.

    Args:
        store: ParamStore for saving optimized params.
        risk_config: Risk configuration for backtests.

    Example:
        >>> optimizer = ParamOptimizer(param_store)
        >>> best = optimizer.optimize_regime(
        ...     "TREND_LOW_VOL",
        ...     {"USD/JPY": candles_df},
        ... )
    """

    def __init__(
        self,
        store: ParamStore,
        risk_config: Optional[RiskConfig] = None,
    ) -> None:
        self.store = store
        self.risk_config = risk_config or RiskConfig()

    def optimize_regime(
        self,
        regime: str,
        recent_candles: dict[str, pd.DataFrame],
        n_trials: int = 50,
    ) -> RegimeAdaptedParams:
        """Run Bayesian optimization for a specific regime.

        Args:
            regime: CompositeRegime string value.
            recent_candles: Dict of {pair: DataFrame} with recent data.
            n_trials: Number of optimization trials.

        Returns:
            Best RegimeAdaptedParams found.
        """
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning(
                "optuna not installed, returning default params"
            )
            return self._default_params(regime)

        def objective(trial: optuna.Trial) -> float:
            params = RegimeAdaptedParams(
                stop_loss_mult=trial.suggest_float(
                    "stop_loss_mult", 0.5, 3.0
                ),
                position_size_mult=trial.suggest_float(
                    "position_size_mult", 0.1, 2.0
                ),
                entry_threshold=trial.suggest_float(
                    "entry_threshold", 0.0, 1.0
                ),
                should_trade=regime
                not in ("RANGE_HIGH_VOL", "RANGE_LOW_VOL"),
                should_close_on_weakness=regime == "RANGE_HIGH_VOL",
            )

            # Additional strategy params derived from regime params
            atr_stop_mult = trial.suggest_float(
                "atr_stop_mult", 1.5, 5.0
            )
            trailing_activation = trial.suggest_float(
                "trailing_activation_pct", 0.02, 0.10
            )

            total_sharpe = 0.0
            n_pairs = 0

            for pair, df in recent_candles.items():
                if df is None or len(df) < 220:
                    continue

                v3_config = StrategyConfigV3(
                    atr_stop_mult=atr_stop_mult,
                    trailing_activation_pct=trailing_activation,
                    position_size_pct=0.08
                    * params.position_size_mult,
                )

                strategy = CarryTradeStrategyV3(config=v3_config)
                engine = BacktestEngine(
                    strategy=strategy,
                    risk_config=self.risk_config,
                )

                try:
                    result = engine.run(df, pair)
                    total_sharpe += result.metrics.sharpe_ratio
                    n_pairs += 1
                except Exception as e:
                    logger.debug(
                        "Trial backtest failed for %s: %s", pair, e
                    )

            if n_pairs == 0:
                return -10.0

            return total_sharpe / n_pairs

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        best_params = RegimeAdaptedParams(
            stop_loss_mult=best["stop_loss_mult"],
            position_size_mult=best["position_size_mult"],
            entry_threshold=best["entry_threshold"],
            should_trade=regime
            not in ("RANGE_HIGH_VOL", "RANGE_LOW_VOL"),
            should_close_on_weakness=regime == "RANGE_HIGH_VOL",
        )

        # Save to store
        param_id = self.store.save_params(regime, best_params)
        self.store.record_performance(
            param_id,
            sharpe=study.best_value,
            win_rate=0.0,  # Updated after live evaluation
            n=0,
        )
        self.store.activate(param_id, regime)

        logger.info(
            "Optimized %s: sharpe=%.3f, params=%s",
            regime,
            study.best_value,
            best_params,
        )
        return best_params

    @staticmethod
    def _default_params(regime: str) -> RegimeAdaptedParams:
        """Return hardcoded defaults for a regime."""
        defaults = {
            "TREND_LOW_VOL": RegimeAdaptedParams(
                stop_loss_mult=0.8,
                position_size_mult=1.2,
                entry_threshold=0.5,
                should_trade=True,
            ),
            "TREND_HIGH_VOL": RegimeAdaptedParams(
                stop_loss_mult=1.5,
                position_size_mult=0.7,
                entry_threshold=0.7,
                should_trade=True,
            ),
            "RANGE_LOW_VOL": RegimeAdaptedParams(
                stop_loss_mult=1.0,
                position_size_mult=0.3,
                entry_threshold=0.95,
                should_trade=False,
            ),
            "RANGE_HIGH_VOL": RegimeAdaptedParams(
                stop_loss_mult=2.0,
                position_size_mult=0.0,
                entry_threshold=1.0,
                should_trade=False,
                should_close_on_weakness=True,
            ),
        }
        return defaults.get(
            regime,
            RegimeAdaptedParams(
                stop_loss_mult=1.0,
                position_size_mult=1.0,
                entry_threshold=0.5,
                should_trade=True,
            ),
        )
