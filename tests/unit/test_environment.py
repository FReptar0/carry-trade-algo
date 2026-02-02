"""Tests for the TradingEnv."""

import numpy as np
import pandas as pd
import pytest

try:
    import gymnasium

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from src.rl.environment import EnvConfig


@pytest.fixture
def sample_candles():
    n = 500
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": prices,
            "high": prices + np.random.rand(n) * 0.5,
            "low": prices - np.random.rand(n) * 0.5,
            "close": prices + np.random.randn(n) * 0.1,
            "volume": np.random.randint(100, 1000, n),
            "swap_long": 0.005,
            "swap_short": -0.005,
        }
    )


@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
class TestTradingEnv:
    def test_env_creation(self, sample_candles):
        from src.rl.environment import TradingEnv

        env = TradingEnv(sample_candles)
        assert env.observation_space.shape == (10,)
        assert env.action_space.n == 3

    def test_reset(self, sample_candles):
        from src.rl.environment import TradingEnv

        env = TradingEnv(sample_candles)
        obs, info = env.reset()
        assert obs.shape == (10,)
        assert isinstance(info, dict)

    def test_step_hold(self, sample_candles):
        from src.rl.environment import TradingEnv

        env = TradingEnv(sample_candles)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        assert obs.shape == (10,)
        assert isinstance(reward, float)

    def test_step_enter_exit(self, sample_candles):
        from src.rl.environment import TradingEnv

        env = TradingEnv(sample_candles)
        env.reset()
        env.step(1)  # ENTER_LONG
        assert env._position is not None
        env.step(2)  # EXIT
        assert env._position is None

    def test_full_episode(self, sample_candles):
        from src.rl.environment import TradingEnv

        config = EnvConfig(episode_length=50)
        env = TradingEnv(sample_candles, config=config)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        assert steps == 50
        assert "equity" in info

    def test_position_flag_in_obs(self, sample_candles):
        from src.rl.environment import TradingEnv

        env = TradingEnv(sample_candles)
        obs, _ = env.reset()
        assert obs[8] == 0.0  # No position
        env.step(1)  # Enter
        obs2 = env._get_obs()
        assert obs2[8] == 1.0  # Has position

    def test_episode_closes_at_end(self, sample_candles):
        from src.rl.environment import TradingEnv

        config = EnvConfig(episode_length=10)
        env = TradingEnv(sample_candles, config=config)
        env.reset()
        env.step(1)  # Enter
        for _ in range(9):
            _, _, terminated, _, _ = env.step(0)  # Hold
        assert terminated is True
        assert env._position is None  # Force closed

    def test_commission_applied(self, sample_candles):
        from src.rl.environment import TradingEnv

        config = EnvConfig(commission=100.0)
        env = TradingEnv(sample_candles, config=config)
        env.reset()
        initial_equity = env._equity
        env.step(1)  # Enter (costs commission)
        assert env._equity < initial_equity


@pytest.mark.skipif(HAS_GYMNASIUM, reason="Test for missing gymnasium")
class TestTradingEnvStub:
    def test_stub_raises(self):
        from src.rl.environment import TradingEnv

        with pytest.raises(ImportError):
            TradingEnv(pd.DataFrame())
