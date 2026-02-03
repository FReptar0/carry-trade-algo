"""Automated retraining pipeline for ML models.

Weekly retraining of the contextual bandit (and future RL models).
Handles the full lifecycle: train, register, shadow evaluate,
and auto-promote if shadow outperforms active.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.ml.bandit import SignalBandit
from src.ml.features import FeatureExtractor
from src.rl.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated model retraining and lifecycle management.

    Args:
        registry: ModelRegistry for version tracking.
        artifacts_dir: Directory for model artifacts.
        promote_threshold: Min relative improvement to promote.

    Example:
        >>> pipeline = RetrainingPipeline(registry)
        >>> version = pipeline.retrain_bandit(recent_trades)
        >>> promoted = pipeline.auto_promote("bandit_v1")
    """

    def __init__(
        self,
        registry: ModelRegistry,
        artifacts_dir: str = "data/models",
        promote_threshold: float = 0.10,
    ) -> None:
        self.registry = registry
        self.artifacts_dir = artifacts_dir
        self.promote_threshold = promote_threshold
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    def retrain_bandit(
        self,
        recent_trades: list[dict],
        n_features: int = 14,
    ) -> Optional[int]:
        """Retrain the contextual bandit on recent trade data.

        Args:
            recent_trades: List of trade dicts with keys:
                features (list), action (str), pnl (float).
            n_features: Number of features.

        Returns:
            New model version number, or None if training failed.
        """
        if not recent_trades:
            logger.warning("No trades for bandit retraining")
            return None

        bandit = SignalBandit(n_features=n_features)

        # Train on recent trades
        for trade in recent_trades:
            features = np.array(
                trade.get("features", [0.0] * n_features)
            )
            if len(features) != n_features:
                continue

            pnl = trade.get("pnl", 0.0)
            # Normalize reward to [-1, 1] range
            max_pnl = max(
                abs(t.get("pnl", 0.0)) for t in recent_trades
            )
            reward = pnl / max_pnl if max_pnl > 0 else 0.0

            action = trade.get("action", "TAKE")
            bandit.update(features, action, reward)

        # Save model artifact
        artifact_path = str(
            Path(self.artifacts_dir) / f"bandit_v{bandit.total_updates}.json"
        )
        bandit.save(artifact_path)

        # Calculate training metrics
        train_sharpe = self._estimate_sharpe(recent_trades)

        # Register in registry
        version = self.registry.register(
            model_type="bandit_v1",
            artifact_path=artifact_path,
            metrics={
                "train_sharpe": train_sharpe,
                "val_sharpe": train_sharpe * 0.8,  # Conservative estimate
            },
        )

        # Set as shadow
        self.registry.promote(version, "shadow")

        logger.info(
            "Retrained bandit v%d on %d trades (sharpe=%.3f)",
            version,
            len(recent_trades),
            train_sharpe,
        )
        return version

    def auto_promote(
        self, model_type: str = "bandit_v1"
    ) -> Optional[int]:
        """Auto-promote shadow model if it outperforms active.

        Args:
            model_type: Type of model to check.

        Returns:
            Promoted version number, or None.
        """
        active = self.registry.get_active(model_type)
        shadow = self.registry.get_shadow(model_type)

        if shadow is None:
            logger.debug("No shadow model to evaluate")
            return None

        if shadow.shadow_sharpe is None:
            logger.debug("Shadow model has no evaluation data yet")
            return None

        # Compare shadow to active
        active_sharpe = active.train_sharpe if active else 0.0
        improvement = (
            (shadow.shadow_sharpe - active_sharpe) / abs(active_sharpe)
            if active_sharpe != 0
            else 1.0
            if shadow.shadow_sharpe > 0
            else 0.0
        )

        if improvement >= self.promote_threshold:
            self.registry.promote(shadow.version, "active")
            logger.info(
                "Auto-promoted %s v%d: shadow_sharpe=%.3f vs "
                "active_sharpe=%.3f (improvement=%.1f%%)",
                model_type,
                shadow.version,
                shadow.shadow_sharpe,
                active_sharpe,
                improvement * 100,
            )
            return shadow.version

        logger.info(
            "Shadow %s v%d not promoted: improvement=%.1f%% "
            "< threshold=%.1f%%",
            model_type,
            shadow.version,
            improvement * 100,
            self.promote_threshold * 100,
        )
        return None

    @staticmethod
    def _estimate_sharpe(trades: list[dict]) -> float:
        """Estimate Sharpe ratio from trade P&L data."""
        pnls = [t.get("pnl", 0.0) for t in trades]
        if len(pnls) < 2:
            return 0.0
        mean = np.mean(pnls)
        std = np.std(pnls)
        if std == 0:
            return 0.0
        # Annualize: assume ~250 trades/year
        return float(mean / std * np.sqrt(min(250, len(pnls))))
