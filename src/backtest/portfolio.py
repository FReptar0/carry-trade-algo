"""Portfolio management -- tracks your money and positions.

Think of this as your trading account ledger. It knows:
- How much cash you have
- What positions are open (what you bought, at what price)
- Your profit/loss history over time

Key concepts:
    - Equity = cash + value of open positions (your total worth)
    - Drawdown = how far equity has fallen from its peak
      (if your peak was $10,000 and now you have $9,000, drawdown = 10%)
    - A "trade" is a complete round-trip: entry -> hold -> exit
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Trade:
    """A completed trade (entry + exit).

    Attributes:
        pair: Which forex pair was traded.
        entry_time: When the position was opened.
        exit_time: When the position was closed.
        entry_price: Price at entry.
        exit_price: Price at exit.
        size: Position size in units (number of currency units).
        direction: "long" or "short".
        pnl: Profit or loss in account currency.
        swap_earned: Total swap income earned during the trade.
        exit_reason: Why the trade was closed.
    """

    pair: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    direction: str
    pnl: float
    swap_earned: float
    exit_reason: str


@dataclass
class Position:
    """An open position (not yet closed)."""

    pair: str
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    direction: str
    swap_accumulated: float = 0.0


class Portfolio:
    """Manages cash, positions, and trade history.

    Args:
        initial_capital: Starting cash balance.
        position_size_pct: What fraction of capital to risk per trade.
        commission: Fixed cost per trade (one-way).
        slippage_pct: Simulated slippage as fraction of price.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        position_size_pct: float = 0.10,
        commission: float = 5.0,
        slippage_pct: float = 0.0001,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.slippage_pct = slippage_pct

        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.equity_history: list[dict] = []
        self.peak_equity = initial_capital

    def open_position(
        self, pair: str, price: float, timestamp: pd.Timestamp, direction: str = "long"
    ) -> bool:
        """Open a new position.

        Args:
            pair: Forex pair name.
            price: Current market price.
            timestamp: Current time.
            direction: "long" (buy) or "short" (sell).

        Returns:
            True if position was opened, False if not enough cash.
        """
        if self.position is not None:
            return False  # Already in a position

        # Calculate position size
        capital_to_use = self.cash * self.position_size_pct
        slipped_price = price * (1 + self.slippage_pct)  # Buy slightly higher
        size = (capital_to_use - self.commission) / slipped_price

        if size <= 0:
            return False

        self.cash -= capital_to_use
        self.position = Position(
            pair=pair,
            entry_time=timestamp,
            entry_price=slipped_price,
            size=size,
            direction=direction,
        )
        return True

    def close_position(
        self, price: float, timestamp: pd.Timestamp, reason: str = ""
    ) -> Trade | None:
        """Close the current position and record the trade.

        Args:
            price: Current market price.
            timestamp: Current time.
            reason: Why closing (stop loss, take profit, etc.).

        Returns:
            The completed Trade, or None if no position was open.
        """
        if self.position is None:
            return None

        pos = self.position
        slipped_price = price * (1 - self.slippage_pct)  # Sell slightly lower

        # Calculate P&L
        if pos.direction == "long":
            pnl = (slipped_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - slipped_price) * pos.size

        pnl -= self.commission  # Closing commission

        trade = Trade(
            pair=pos.pair,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=slipped_price,
            size=pos.size,
            direction=pos.direction,
            pnl=pnl,
            swap_earned=pos.swap_accumulated,
            exit_reason=reason,
        )

        self.cash += (pos.entry_price * pos.size) + pnl  # Return capital + profit
        self.trades.append(trade)
        self.position = None
        return trade

    def accrue_swap(self, swap_value: float) -> None:
        """Credit/debit daily swap to the open position.

        Called once per day while holding a position.
        Swap goes directly into the position's accumulated swap.
        """
        if self.position is not None:
            self.position.swap_accumulated += swap_value * self.position.size

    def record_equity(self, timestamp: pd.Timestamp, price: float) -> None:
        """Snapshot current equity for the equity curve.

        Equity = cash + market value of open position.
        """
        position_value = 0.0
        if self.position is not None:
            pos = self.position
            if pos.direction == "long":
                position_value = (price - pos.entry_price) * pos.size
            else:
                position_value = (pos.entry_price - price) * pos.size
            position_value += pos.swap_accumulated

        equity = self.cash + position_value + (
            self.position.entry_price * self.position.size if self.position else 0.0
        )
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        self.equity_history.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": self.cash,
            "position_value": position_value,
            "drawdown": drawdown,
        })

    def get_equity_df(self) -> pd.DataFrame:
        """Return equity history as a DataFrame."""
        return pd.DataFrame(self.equity_history)

    def get_trades_df(self) -> pd.DataFrame:
        """Return trade history as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                "pair": t.pair,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "direction": t.direction,
                "pnl": t.pnl,
                "swap_earned": t.swap_earned,
                "total_pnl": t.pnl + t.swap_earned,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(records)
