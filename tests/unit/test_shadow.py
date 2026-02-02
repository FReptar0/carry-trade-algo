"""Tests for the ShadowEvaluator."""

import numpy as np
import pytest

from src.ml.shadow import ShadowEvaluator


class TestShadowEvaluator:
    def test_insufficient_trades(self):
        shadow = ShadowEvaluator(min_trades=50)
        for _ in range(10):
            shadow.record(np.zeros(5), "TAKE", actual_pnl=100)
        activate, stats = shadow.should_activate()
        assert activate is False
        assert stats["reason"] == "insufficient_trades"

    def test_activates_with_advantage(self):
        shadow = ShadowEvaluator(min_trades=10, min_advantage=0.05)

        # Bandit takes winners, skips losers
        for _ in range(8):
            shadow.record(np.zeros(5), "TAKE", actual_pnl=100)
        for _ in range(8):
            shadow.record(np.zeros(5), "SKIP", actual_pnl=-200)

        activate, stats = shadow.should_activate()
        assert activate is True
        assert stats["advantage"] > 0.05

    def test_does_not_activate_without_advantage(self):
        shadow = ShadowEvaluator(min_trades=10, min_advantage=0.5)

        # Bandit takes everything (no discrimination)
        for _ in range(20):
            shadow.record(np.zeros(5), "TAKE", actual_pnl=50)

        activate, stats = shadow.should_activate()
        assert activate is False

    def test_trade_count(self):
        shadow = ShadowEvaluator()
        assert shadow.trade_count == 0
        shadow.record(np.zeros(5), "TAKE", actual_pnl=100)
        assert shadow.trade_count == 1

    def test_reset(self):
        shadow = ShadowEvaluator()
        shadow.record(np.zeros(5), "TAKE", actual_pnl=100)
        shadow.reset()
        assert shadow.trade_count == 0

    def test_accuracy_calculation(self):
        shadow = ShadowEvaluator(min_trades=4, min_advantage=-99)
        shadow.record(np.zeros(5), "TAKE", actual_pnl=100)   # Correct
        shadow.record(np.zeros(5), "SKIP", actual_pnl=-50)   # Correct
        shadow.record(np.zeros(5), "TAKE", actual_pnl=-20)   # Wrong
        shadow.record(np.zeros(5), "SKIP", actual_pnl=80)    # Wrong
        _, stats = shadow.should_activate()
        assert stats["accuracy"] == 0.5

    def test_stats_contain_expected_keys(self):
        shadow = ShadowEvaluator(min_trades=2, min_advantage=-99)
        shadow.record(np.zeros(5), "TAKE", actual_pnl=100)
        shadow.record(np.zeros(5), "SKIP", actual_pnl=-50)
        _, stats = shadow.should_activate()
        expected_keys = [
            "trades", "bandit_pnl", "random_pnl",
            "advantage", "accuracy", "take_count",
            "skip_count", "avg_take_pnl", "avg_skip_pnl",
        ]
        for key in expected_keys:
            assert key in stats

    def test_all_take_decisions(self):
        shadow = ShadowEvaluator(min_trades=5, min_advantage=-99)
        for _ in range(5):
            shadow.record(np.zeros(5), "TAKE", actual_pnl=50)
        activate, stats = shadow.should_activate()
        assert stats["skip_count"] == 0
        assert stats["take_count"] == 5
