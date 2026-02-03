"""Contextual bandit for signal filtering.

Uses Thompson Sampling with a Bayesian linear model. Two arms:
TAKE (execute the signal) and SKIP (ignore it). The bandit learns
from trade outcomes which market contexts produce profitable signals.

No heavy ML frameworks needed — pure numpy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ACTIONS = ("TAKE", "SKIP")


class SignalBandit:
    """Contextual bandit with Thompson Sampling for signal gating.

    Maintains separate Bayesian linear regression weights for each
    arm (TAKE/SKIP). Uses Thompson Sampling to choose the arm with
    higher expected reward given the current context features.

    Args:
        n_features: Number of context features.
        prior_strength: Prior precision (higher = more conservative).

    Example:
        >>> bandit = SignalBandit(n_features=14)
        >>> action, confidence = bandit.decide(features)
        >>> # After trade outcome:
        >>> bandit.update(features, "TAKE", reward=0.5)
    """

    def __init__(
        self,
        n_features: int = 14,
        prior_strength: float = 1.0,
    ) -> None:
        self.n_features = n_features
        self.prior_strength = prior_strength

        # Bayesian linear model parameters per arm
        # mean weights and precision (inverse variance)
        self._mu: dict[str, np.ndarray] = {
            "TAKE": np.zeros(n_features),
            "SKIP": np.zeros(n_features),
        }
        self._precision: dict[str, np.ndarray] = {
            "TAKE": np.eye(n_features) * prior_strength,
            "SKIP": np.eye(n_features) * prior_strength,
        }
        self._sigma_sq: dict[str, float] = {
            "TAKE": 1.0,
            "SKIP": 1.0,
        }
        self._update_count: dict[str, int] = {
            "TAKE": 0,
            "SKIP": 0,
        }

    def decide(
        self, features: np.ndarray
    ) -> tuple[str, float]:
        """Choose TAKE or SKIP via Thompson Sampling.

        Args:
            features: 1-D array of context features.

        Returns:
            Tuple of (action, confidence). Action is "TAKE" or "SKIP".
            Confidence is the difference in expected rewards.
        """
        features = np.asarray(features, dtype=np.float64)

        scores = {}
        for action in ACTIONS:
            # Sample weights from posterior
            try:
                cov = np.linalg.inv(self._precision[action])
                cov = (cov + cov.T) / 2  # Ensure symmetry
                sampled_w = np.random.multivariate_normal(
                    self._mu[action], cov * self._sigma_sq[action]
                )
            except np.linalg.LinAlgError:
                sampled_w = self._mu[action]

            scores[action] = float(features @ sampled_w)

        chosen = max(scores, key=scores.get)
        confidence = abs(scores["TAKE"] - scores["SKIP"])

        return chosen, confidence

    def update(
        self,
        features: np.ndarray,
        action: str,
        reward: float,
    ) -> None:
        """Update the model with observed reward.

        Args:
            features: Context features when decision was made.
            action: Action taken ("TAKE" or "SKIP").
            reward: Observed reward (e.g., normalized P&L).
        """
        if action not in ACTIONS:
            logger.warning("Unknown action: %s", action)
            return

        features = np.asarray(features, dtype=np.float64)
        x = features.reshape(-1, 1)

        # Bayesian update: precision += x @ x.T / sigma_sq
        self._precision[action] += (x @ x.T) / self._sigma_sq[action]

        # Mean update: mu = precision^-1 @ (precision_old @ mu_old + x * y / sigma_sq)
        try:
            prec_inv = np.linalg.inv(self._precision[action])
            old_term = (
                self._precision[action]
                - (x @ x.T) / self._sigma_sq[action]
            ) @ self._mu[action]
            new_term = x.flatten() * reward / self._sigma_sq[action]
            self._mu[action] = prec_inv @ (old_term + new_term)
        except np.linalg.LinAlgError:
            logger.warning("Bandit update failed for %s", action)

        self._update_count[action] += 1

    @property
    def total_updates(self) -> int:
        return sum(self._update_count.values())

    def save(self, path: str) -> None:
        """Serialize bandit state to JSON.

        Args:
            path: File path to save to.
        """
        state = {
            "n_features": self.n_features,
            "prior_strength": self.prior_strength,
            "mu": {k: v.tolist() for k, v in self._mu.items()},
            "precision": {
                k: v.tolist() for k, v in self._precision.items()
            },
            "sigma_sq": self._sigma_sq,
            "update_count": self._update_count,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)
        logger.info("Bandit saved to %s", path)

    def load(self, path: str) -> None:
        """Load bandit state from JSON.

        Args:
            path: File path to load from.
        """
        with open(path) as f:
            state = json.load(f)

        self.n_features = state["n_features"]
        self.prior_strength = state["prior_strength"]
        self._mu = {
            k: np.array(v) for k, v in state["mu"].items()
        }
        self._precision = {
            k: np.array(v) for k, v in state["precision"].items()
        }
        self._sigma_sq = state["sigma_sq"]
        self._update_count = state["update_count"]
        logger.info("Bandit loaded from %s", path)
