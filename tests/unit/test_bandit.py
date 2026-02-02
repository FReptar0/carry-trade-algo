"""Tests for the SignalBandit."""

import os
import tempfile

import numpy as np
import pytest

from src.ml.bandit import SignalBandit


class TestSignalBandit:
    def test_initial_decide(self):
        bandit = SignalBandit(n_features=5)
        features = np.random.randn(5)
        action, confidence = bandit.decide(features)
        assert action in ("TAKE", "SKIP")
        assert isinstance(confidence, float)

    def test_update_does_not_crash(self):
        bandit = SignalBandit(n_features=5)
        features = np.random.randn(5)
        bandit.update(features, "TAKE", reward=1.0)
        assert bandit.total_updates == 1

    def test_multiple_updates(self):
        bandit = SignalBandit(n_features=5)
        for _ in range(10):
            features = np.random.randn(5)
            bandit.update(features, "TAKE", reward=np.random.randn())
        assert bandit.total_updates == 10

    def test_save_and_load(self, tmp_path):
        bandit = SignalBandit(n_features=5)
        features = np.random.randn(5)
        bandit.update(features, "TAKE", reward=1.0)
        bandit.update(features, "SKIP", reward=-0.5)

        path = str(tmp_path / "bandit.json")
        bandit.save(path)

        loaded = SignalBandit(n_features=5)
        loaded.load(path)

        assert loaded.total_updates == bandit.total_updates
        np.testing.assert_array_almost_equal(
            loaded._mu["TAKE"], bandit._mu["TAKE"]
        )

    def test_learns_positive_signals(self):
        np.random.seed(42)
        bandit = SignalBandit(n_features=3, prior_strength=0.1)

        # Train: good signals have feature[0] > 0
        for _ in range(50):
            features = np.random.randn(3)
            if features[0] > 0:
                bandit.update(features, "TAKE", reward=1.0)
            else:
                bandit.update(features, "SKIP", reward=0.5)

        # Test: positive feature should lean toward TAKE
        take_count = 0
        for _ in range(20):
            pos_features = np.array([2.0, 0.0, 0.0])
            action, _ = bandit.decide(pos_features)
            if action == "TAKE":
                take_count += 1

        assert take_count > 5  # Should favor TAKE

    def test_unknown_action_ignored(self):
        bandit = SignalBandit(n_features=5)
        features = np.random.randn(5)
        bandit.update(features, "INVALID", reward=1.0)
        assert bandit.total_updates == 0

    def test_zero_features(self):
        bandit = SignalBandit(n_features=5)
        features = np.zeros(5)
        action, confidence = bandit.decide(features)
        assert action in ("TAKE", "SKIP")

    def test_large_features(self):
        bandit = SignalBandit(n_features=5)
        features = np.ones(5) * 1000
        action, confidence = bandit.decide(features)
        assert action in ("TAKE", "SKIP")

    def test_total_updates_both_arms(self):
        bandit = SignalBandit(n_features=3)
        bandit.update(np.ones(3), "TAKE", 1.0)
        bandit.update(np.ones(3), "SKIP", -1.0)
        assert bandit.total_updates == 2
        assert bandit._update_count["TAKE"] == 1
        assert bandit._update_count["SKIP"] == 1
