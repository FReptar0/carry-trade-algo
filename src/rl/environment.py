"""Gymnasium-compatible trading environment for RL training.

Wraps the backtest engine's logic into a standard Gymnasium environment.
Observation includes technical indicators and position state.
Actions: HOLD, ENTER_LONG, EXIT.
Reward: change in equity with swap income minus transaction costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.indicators import atr, rsi, sma, adx

logger = logging.getLogger(__name__)

try:
    import gymnasium
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


@dataclass
class EnvConfig:
    """Configuration for the trading environment.

    Attributes:
        initial_equity: Starting account balance.
        position_size_pct: Fraction of equity per trade.
        commission: Cost per trade in base currency.
        swap_daily: Daily swap income for long positions.
        episode_length: Max bars per episode (0 = use all data).
    """

    initial_equity: float = 10000.0
    position_size_pct: float = 0.08
    commission: float = 5.0
    swap_daily: float = 0.005
    episode_length: int = 0


def _compute_features(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Compute observation features at index idx.

    Returns an array of 8 features:
    [rsi_14, atr_ratio, ma_spread, adx_14, price_vs_200ma,
     hour_of_day, day_of_week, swap_rate]
    """
    if idx < 200:
        return np.zeros(8, dtype=np.float32)

    close = df["close"].iloc[: idx + 1]
    high = df["high"].iloc[: idx + 1]
    low = df["low"].iloc[: idx + 1]
    price = float(close.iloc[-1])

    # RSI
    rsi_vals = rsi(close, 14)
    rsi_val = float(rsi_vals.iloc[-1]) if not np.isnan(rsi_vals.iloc[-1]) else 50.0

    # ATR ratio
    atr_14 = atr(high, low, close, 14)
    atr_60 = atr(high, low, close, 60)
    a14 = float(atr_14.iloc[-1]) if not np.isnan(atr_14.iloc[-1]) else 1.0
    a60 = float(atr_60.iloc[-1]) if not np.isnan(atr_60.iloc[-1]) else 1.0
    atr_ratio = a14 / a60 if a60 > 0 else 1.0

    # MA spread
    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    m50 = float(ma50.iloc[-1]) if not np.isnan(ma50.iloc[-1]) else price
    m200 = float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else price
    ma_spread = (m50 - m200) / price if price > 0 else 0.0

    # ADX
    adx_vals, _, _ = adx(high, low, close, 14)
    adx_val = float(adx_vals.iloc[-1]) if not np.isnan(adx_vals.iloc[-1]) else 20.0

    # Price vs 200MA
    p_vs_200 = (price - m200) / m200 if m200 > 0 else 0.0

    # Time features
    ts = df["timestamp"].iloc[idx]
    if isinstance(ts, pd.Timestamp):
        hour = ts.hour / 23.0
        dow = ts.dayofweek / 4.0
    else:
        hour = 0.5
        dow = 0.5

    # Swap
    swap = float(df["swap_long"].iloc[idx]) if "swap_long" in df.columns else 0.0

    return np.array(
        [rsi_val / 100.0, atr_ratio, ma_spread * 100, adx_val / 50.0,
         p_vs_200 * 100, hour, dow, swap * 100],
        dtype=np.float32,
    )


if HAS_GYMNASIUM:

    class TradingEnv(gymnasium.Env):
        """Gymnasium trading environment.

        Observation space: 10-dim continuous (8 features + position_flag + unrealized_pnl)
        Action space: Discrete(3) — HOLD=0, ENTER_LONG=1, EXIT=2
        Reward: Change in equity with swap and costs.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            candles: pd.DataFrame,
            pair: str = "USD/JPY",
            config: EnvConfig | None = None,
        ) -> None:
            super().__init__()
            self.candles = candles.reset_index(drop=True)
            self.pair = pair
            self.config = config or EnvConfig()

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)

            self._equity = self.config.initial_equity
            self._position: Optional[dict] = None
            self._step_idx = 200  # Skip warmup
            self._prev_equity = self._equity
            self._last_date = None
            self._trade_count = 0

        def reset(
            self, *, seed=None, options=None
        ) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self._equity = self.config.initial_equity
            self._position = None
            self._step_idx = 200
            self._prev_equity = self._equity
            self._last_date = None
            self._trade_count = 0
            return self._get_obs(), {}

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            price = float(self.candles["close"].iloc[self._step_idx])

            # Accrue daily swap
            ts = self.candles["timestamp"].iloc[self._step_idx]
            current_date = ts.date() if isinstance(ts, pd.Timestamp) else None
            if (
                self._position is not None
                and self._last_date is not None
                and current_date != self._last_date
            ):
                self._equity += self.config.swap_daily
            self._last_date = current_date

            # Execute action
            if action == 1 and self._position is None:  # ENTER_LONG
                units = int(
                    self._equity * self.config.position_size_pct / price
                )
                if units > 0:
                    self._position = {
                        "entry_price": price,
                        "units": units,
                    }
                    self._equity -= self.config.commission
                    self._trade_count += 1

            elif action == 2 and self._position is not None:  # EXIT
                pnl = (price - self._position["entry_price"]) * self._position["units"]
                self._equity += pnl - self.config.commission
                self._position = None

            # Calculate unrealized equity
            total_equity = self._equity
            if self._position is not None:
                unrealized = (price - self._position["entry_price"]) * self._position["units"]
                total_equity += unrealized

            # Reward = change in total equity
            reward = total_equity - self._prev_equity

            # Small penalty for excessive trading
            if action in (1, 2):
                reward -= 0.5

            self._prev_equity = total_equity
            self._step_idx += 1

            # Episode termination
            max_idx = len(self.candles) - 1
            if self.config.episode_length > 0:
                max_idx = min(
                    max_idx, 200 + self.config.episode_length
                )

            terminated = self._step_idx >= max_idx
            truncated = False

            if terminated and self._position is not None:
                # Force close at end
                close_price = float(
                    self.candles["close"].iloc[min(self._step_idx, len(self.candles) - 1)]
                )
                pnl = (close_price - self._position["entry_price"]) * self._position["units"]
                self._equity += pnl
                self._position = None

            return self._get_obs(), float(reward), terminated, truncated, {
                "equity": total_equity,
                "trades": self._trade_count,
            }

        def _get_obs(self) -> np.ndarray:
            idx = min(self._step_idx, len(self.candles) - 1)
            features = _compute_features(self.candles, idx)

            position_flag = 1.0 if self._position is not None else 0.0
            unrealized = 0.0
            if self._position is not None:
                price = float(self.candles["close"].iloc[idx])
                unrealized = (
                    (price - self._position["entry_price"])
                    * self._position["units"]
                    / self.config.initial_equity
                )

            return np.concatenate([
                features,
                np.array([position_flag, unrealized], dtype=np.float32),
            ])

else:
    # Stub when gymnasium is not installed
    class TradingEnv:
        """Stub TradingEnv when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gymnasium is required for TradingEnv. "
                "Install with: uv add gymnasium"
            )
