"""Tests for the ModelRegistry."""

import pytest

from src.rl.registry import ModelRegistry, ModelVersion


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        db_path=str(tmp_path / "test_registry.db"),
        artifacts_dir=str(tmp_path / "models"),
    )


class TestModelRegistry:
    def test_register_model(self, registry):
        vid = registry.register(
            "bandit_v1",
            "models/b1.json",
            {"train_sharpe": 1.5, "val_sharpe": 1.2},
        )
        assert vid > 0

    def test_get_active_none(self, registry):
        assert registry.get_active("bandit_v1") is None

    def test_promote_to_active(self, registry):
        vid = registry.register(
            "bandit_v1",
            "models/b1.json",
            {"train_sharpe": 1.5, "val_sharpe": 1.2},
        )
        registry.promote(vid, "active")
        active = registry.get_active("bandit_v1")
        assert active is not None
        assert active.version == vid
        assert active.status == "active"

    def test_promote_retires_previous(self, registry):
        vid1 = registry.register(
            "bandit_v1", "m1.json", {"train_sharpe": 1.0, "val_sharpe": 0.8}
        )
        registry.promote(vid1, "active")

        vid2 = registry.register(
            "bandit_v1", "m2.json", {"train_sharpe": 1.5, "val_sharpe": 1.2}
        )
        registry.promote(vid2, "active")

        active = registry.get_active("bandit_v1")
        assert active.version == vid2

        history = registry.get_history("bandit_v1")
        retired = [m for m in history if m.status == "retired"]
        assert len(retired) == 1
        assert retired[0].version == vid1

    def test_get_shadow(self, registry):
        vid = registry.register(
            "bandit_v1", "m.json", {"train_sharpe": 1.0, "val_sharpe": 0.8}
        )
        registry.promote(vid, "shadow")
        shadow = registry.get_shadow("bandit_v1")
        assert shadow is not None
        assert shadow.status == "shadow"

    def test_update_shadow_sharpe(self, registry):
        vid = registry.register(
            "bandit_v1", "m.json", {"train_sharpe": 1.0, "val_sharpe": 0.8}
        )
        registry.update_shadow_sharpe(vid, 1.3)
        history = registry.get_history("bandit_v1")
        assert history[0].shadow_sharpe == 1.3

    def test_get_history(self, registry):
        for i in range(5):
            registry.register(
                "bandit_v1", f"m{i}.json",
                {"train_sharpe": float(i), "val_sharpe": float(i)},
            )
        history = registry.get_history("bandit_v1")
        assert len(history) == 5
        assert history[0].version > history[-1].version

    def test_different_model_types(self, registry):
        vid1 = registry.register(
            "bandit_v1", "b.json", {"train_sharpe": 1.0, "val_sharpe": 0.8}
        )
        vid2 = registry.register(
            "rl_ppo_v1", "r.json", {"train_sharpe": 2.0, "val_sharpe": 1.5}
        )
        registry.promote(vid1, "active")
        registry.promote(vid2, "active")

        assert registry.get_active("bandit_v1").version == vid1
        assert registry.get_active("rl_ppo_v1").version == vid2

    def test_model_version_fields(self, registry):
        vid = registry.register(
            "bandit_v1", "m.json",
            {"train_sharpe": 1.5, "val_sharpe": 1.2},
        )
        registry.promote(vid, "active")
        model = registry.get_active("bandit_v1")
        assert isinstance(model, ModelVersion)
        assert model.train_sharpe == 1.5
        assert model.val_sharpe == 1.2
        assert model.created_at is not None
