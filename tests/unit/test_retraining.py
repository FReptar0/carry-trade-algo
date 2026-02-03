"""Tests for the RetrainingPipeline."""

import numpy as np
import pytest

from src.rl.registry import ModelRegistry
from src.rl.retraining import RetrainingPipeline


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        db_path=str(tmp_path / "test_retrain.db"),
        artifacts_dir=str(tmp_path / "models"),
    )


@pytest.fixture
def pipeline(registry, tmp_path):
    return RetrainingPipeline(
        registry=registry,
        artifacts_dir=str(tmp_path / "models"),
    )


@pytest.fixture
def sample_trades():
    np.random.seed(42)
    trades = []
    for _ in range(30):
        trades.append(
            {
                "features": np.random.randn(14).tolist(),
                "action": "TAKE",
                "pnl": np.random.randn() * 100,
            }
        )
    return trades


class TestRetrainingPipeline:
    def test_retrain_creates_version(self, pipeline, sample_trades):
        vid = pipeline.retrain_bandit(sample_trades)
        assert vid is not None
        assert vid > 0

    def test_retrain_empty_trades(self, pipeline):
        vid = pipeline.retrain_bandit([])
        assert vid is None

    def test_retrain_registers_as_shadow(
        self, pipeline, registry, sample_trades
    ):
        vid = pipeline.retrain_bandit(sample_trades)
        shadow = registry.get_shadow("bandit_v1")
        assert shadow is not None
        assert shadow.version == vid

    def test_auto_promote_no_shadow(self, pipeline):
        result = pipeline.auto_promote()
        assert result is None

    def test_auto_promote_no_sharpe(self, pipeline, sample_trades):
        pipeline.retrain_bandit(sample_trades)
        # Shadow has no shadow_sharpe yet
        result = pipeline.auto_promote()
        assert result is None

    def test_auto_promote_succeeds(
        self, pipeline, registry, sample_trades
    ):
        vid = pipeline.retrain_bandit(sample_trades)
        # Simulate shadow evaluation with good results
        registry.update_shadow_sharpe(vid, 2.0)
        result = pipeline.auto_promote()
        assert result == vid
        active = registry.get_active("bandit_v1")
        assert active.version == vid

    def test_auto_promote_fails_insufficient_improvement(
        self, pipeline, registry, sample_trades
    ):
        # Register and activate a strong model
        vid1 = registry.register(
            "bandit_v1", "m1.json",
            {"train_sharpe": 2.0, "val_sharpe": 1.8},
        )
        registry.promote(vid1, "active")

        # Retrain with slightly worse model
        vid2 = pipeline.retrain_bandit(sample_trades)
        registry.update_shadow_sharpe(vid2, 2.05)  # Only 2.5% improvement

        result = pipeline.auto_promote()
        assert result is None  # Not enough improvement

    def test_estimate_sharpe(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 80}, {"pnl": -30}]
        sharpe = RetrainingPipeline._estimate_sharpe(trades)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Net positive trades

    def test_estimate_sharpe_empty(self):
        sharpe = RetrainingPipeline._estimate_sharpe([])
        assert sharpe == 0.0

    def test_retrain_saves_artifact(self, pipeline, sample_trades, tmp_path):
        pipeline.retrain_bandit(sample_trades)
        models = list((tmp_path / "models").glob("*.json"))
        assert len(models) > 0
