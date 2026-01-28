"""Carry Trade Strategy v1.

How this strategy decides when to trade:

ENTRY (when to open a position):
    All 3 conditions must be true:
    1. The daily swap is above a minimum threshold
       → Only trade pairs where the carry is worth it
    2. Price is above the 50-period SMA (for longs)
       → Only buy when the trend supports you
    3. RSI is below 70
       → Don't buy when the pair is already "overbought"

EXIT (when to close a position):
    Any 1 condition triggers exit:
    1. Stop loss: price drops 2% from entry → cut losses
    2. Take profit: price rises 6% from entry → lock in gains
    3. Trailing stop: once price is up 3%, the stop follows price up
       → lets winners run but protects profits

The idea: earn swap income daily WHILE the trend is in your favor.
Exit quickly if the trend reverses (stop loss) or if you've made enough
(take profit). The trailing stop is the best of both worlds -- it lets
profits grow but automatically exits if the price turns around.
"""

import pandas as pd

from src.config.settings import StrategyConfig
from src.strategy.base import Signal, Strategy, TradeSignal
from src.strategy.indicators import rsi, sma


class CarryTradeStrategy(Strategy):
    """Basic carry trade strategy with trend filter and RSI guard.

    Args:
        config: Strategy parameters. Uses defaults if not provided.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Scan price data and produce buy/sell/close signals.

        Args:
            data: DataFrame with standard columns (timestamp, OHLCV, swaps).

        Returns:
            List of TradeSignals for bars where a decision is made.
        """
        signals: list[TradeSignal] = []

        # Calculate indicators once for the whole dataset
        close = data["close"]
        trend_line = sma(close, self.config.trend_filter_period)
        momentum = rsi(close, period=14)

        # Track position state
        in_position = False
        entry_price = 0.0
        max_price_since_entry = 0.0
        trailing_active = False

        for i in range(len(data)):
            ts = data["timestamp"].iloc[i]
            price = close.iloc[i]
            swap = data["swap_long"].iloc[i]
            sma_val = trend_line.iloc[i]
            rsi_val = momentum.iloc[i]

            # Skip bars where indicators aren't ready yet
            if pd.isna(sma_val) or pd.isna(rsi_val):
                continue

            if not in_position:
                signal = self._check_entry(price, swap, sma_val, rsi_val)
                if signal is not None:
                    signals.append(TradeSignal(
                        timestamp=ts,
                        signal=Signal.LONG,
                        strength=signal,
                        reason=f"Entry: swap={swap:.4f}, price>{sma_val:.2f}, RSI={rsi_val:.1f}",
                    ))
                    in_position = True
                    entry_price = price
                    max_price_since_entry = price
                    trailing_active = False
            else:
                max_price_since_entry = max(max_price_since_entry, price)
                exit_signal = self._check_exit(
                    price, entry_price, max_price_since_entry, trailing_active
                )
                if exit_signal is not None:
                    reason, trailing_active = exit_signal
                    if reason:  # actual exit
                        signals.append(TradeSignal(
                            timestamp=ts,
                            signal=Signal.CLOSE,
                            reason=reason,
                        ))
                        in_position = False

        return signals

    def _check_entry(
        self, price: float, swap: float, sma_val: float, rsi_val: float
    ) -> float | None:
        """Check if entry conditions are met.

        Returns:
            Signal strength (0-1) if entry is valid, None otherwise.
        """
        # Condition 1: swap must be worth it
        if swap < self.config.min_swap_threshold:
            return None
        # Condition 2: price above SMA (trend is up)
        if price <= sma_val:
            return None
        # Condition 3: not overbought
        if rsi_val >= self.config.rsi_threshold:
            return None

        # Signal strength based on how far above SMA and how much swap
        trend_strength = min((price - sma_val) / sma_val * 100, 1.0)
        return max(0.1, trend_strength)

    def _check_exit(
        self,
        price: float,
        entry_price: float,
        max_price: float,
        trailing_active: bool,
    ) -> tuple[str, bool] | None:
        """Check if exit conditions are met.

        Returns:
            (reason_string, trailing_active) if exiting or updating trailing,
            None if no action needed.
        """
        pnl_pct = (price - entry_price) / entry_price

        # Stop loss: cut losses at -2%
        if pnl_pct <= -self.config.stop_loss_pct:
            return f"Stop loss: {pnl_pct:.2%}", trailing_active

        # Take profit: lock in at +6%
        if pnl_pct >= self.config.take_profit_pct:
            return f"Take profit: {pnl_pct:.2%}", trailing_active

        # Trailing stop: activate after 3% profit
        profit_from_peak = (price - max_price) / max_price
        if pnl_pct >= 0.03:
            trailing_active = True

        if trailing_active and profit_from_peak <= -0.01:
            return f"Trailing stop: {pnl_pct:.2%} (peak drawdown {profit_from_peak:.2%})", trailing_active

        return None  # No exit

    def validate_params(self) -> bool:
        """Verify parameters are within reasonable ranges."""
        c = self.config
        return (
            0 < c.min_swap_threshold < 1.0
            and 5 <= c.trend_filter_period <= 500
            and 50 <= c.rsi_threshold <= 90
            and 0 < c.stop_loss_pct < 0.10
            and c.stop_loss_pct < c.take_profit_pct
            and 0 < c.position_size_pct <= 0.5
        )
