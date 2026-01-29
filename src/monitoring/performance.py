"""Real-time performance monitoring for paper trading.

Tracks:
- Equity curve in real-time
- Drawdown (current and maximum)
- Win rate and profit factor
- Sharpe ratio (rolling)
- Position exposure
- Trade statistics

Can emit alerts when:
- Drawdown exceeds threshold
- Daily loss limit approached
- Win rate drops below target
- System health issues detected
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import threading
import time

import numpy as np


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics."""
    timestamp: datetime
    equity: float
    balance: float
    unrealized_pnl: float
    daily_pnl: float
    drawdown: float  # Current drawdown %
    max_drawdown: float  # Maximum drawdown %
    high_water_mark: float
    positions_count: int
    win_rate: float
    profit_factor: float
    sharpe_30d: float  # 30-day rolling Sharpe
    trades_today: int
    total_trades: int


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    timestamp: datetime
    pair: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration: timedelta
    swap_earned: float


class PerformanceMonitor:
    """Real-time performance monitoring.

    Example:
        >>> monitor = PerformanceMonitor(initial_equity=100000)
        >>> monitor.update(equity=100500, balance=100200, ...)
        >>> snapshot = monitor.get_snapshot()
        >>> print(f"Drawdown: {snapshot.drawdown:.2%}")
    """

    def __init__(
        self,
        initial_equity: float,
        alert_callback: Callable[[str, str, float], None] | None = None,
        max_drawdown_alert: float = 0.10,  # Alert at 10% drawdown
        daily_loss_alert: float = 0.02,  # Alert at 2% daily loss
    ) -> None:
        self.initial_equity = initial_equity
        self.alert_callback = alert_callback
        self.max_drawdown_alert = max_drawdown_alert
        self.daily_loss_alert = daily_loss_alert

        # Current state
        self.equity = initial_equity
        self.balance = initial_equity
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        self.positions_count = 0

        # Historical tracking
        self.high_water_mark = initial_equity
        self.max_drawdown = 0.0
        self.equity_history: deque[tuple[datetime, float]] = deque(maxlen=10000)
        self.daily_returns: deque[float] = deque(maxlen=252)  # 1 year of daily returns

        # Trade tracking
        self.trades: list[TradeRecord] = []
        self.trades_today = 0
        self.today_date: Optional[datetime] = None

        # Alerts
        self.alerts_triggered: set[str] = set()

        # Lock for thread safety
        self._lock = threading.Lock()

    def update(
        self,
        equity: float,
        balance: float,
        unrealized_pnl: float,
        daily_pnl: float,
        positions_count: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update current performance metrics.

        Call this periodically (every few seconds or on each price update).

        Args:
            equity: Current equity (balance + unrealized P&L).
            balance: Current cash balance.
            unrealized_pnl: Current unrealized P&L.
            daily_pnl: Today's realized P&L.
            positions_count: Number of open positions.
            timestamp: Timestamp of update (defaults to now).
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            self.equity = equity
            self.balance = balance
            self.unrealized_pnl = unrealized_pnl
            self.daily_pnl = daily_pnl
            self.positions_count = positions_count

            # Update high water mark
            if equity > self.high_water_mark:
                self.high_water_mark = equity

            # Calculate current drawdown
            current_dd = (self.high_water_mark - equity) / self.high_water_mark
            if current_dd > self.max_drawdown:
                self.max_drawdown = current_dd

            # Record equity point
            self.equity_history.append((timestamp, equity))

            # Check for new day
            if self.today_date is None or timestamp.date() != self.today_date.date():
                self._on_new_day(timestamp)

            # Check alerts
            self._check_alerts(current_dd, daily_pnl)

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade.

        Args:
            trade: TradeRecord object.
        """
        with self._lock:
            self.trades.append(trade)
            self.trades_today += 1

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot.

        Returns:
            PerformanceSnapshot with all current metrics.
        """
        with self._lock:
            current_dd = (self.high_water_mark - self.equity) / self.high_water_mark if self.high_water_mark > 0 else 0

            return PerformanceSnapshot(
                timestamp=datetime.now(),
                equity=self.equity,
                balance=self.balance,
                unrealized_pnl=self.unrealized_pnl,
                daily_pnl=self.daily_pnl,
                drawdown=current_dd,
                max_drawdown=self.max_drawdown,
                high_water_mark=self.high_water_mark,
                positions_count=self.positions_count,
                win_rate=self._calculate_win_rate(),
                profit_factor=self._calculate_profit_factor(),
                sharpe_30d=self._calculate_rolling_sharpe(30),
                trades_today=self.trades_today,
                total_trades=len(self.trades),
            )

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from completed trades."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _calculate_rolling_sharpe(self, days: int) -> float:
        """Calculate rolling Sharpe ratio.

        Args:
            days: Number of days for rolling window.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(self.daily_returns) < days:
            return 0.0

        recent = list(self.daily_returns)[-days:]
        mean = np.mean(recent)
        std = np.std(recent)
        if std == 0:
            return 0.0
        return (mean / std) * np.sqrt(252)

    def _on_new_day(self, timestamp: datetime) -> None:
        """Handle day rollover."""
        if self.today_date is not None and len(self.equity_history) >= 2:
            # Calculate previous day's return
            prev_equity = None
            for ts, eq in reversed(self.equity_history):
                if ts.date() < timestamp.date():
                    prev_equity = eq
                    break

            if prev_equity and prev_equity > 0:
                daily_ret = (self.equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_ret)

        self.today_date = timestamp
        self.trades_today = 0
        self.alerts_triggered.clear()

    def _check_alerts(self, current_dd: float, daily_pnl: float) -> None:
        """Check and trigger alerts."""
        if self.alert_callback is None:
            return

        # Drawdown alert
        if current_dd >= self.max_drawdown_alert and "max_drawdown" not in self.alerts_triggered:
            self.alert_callback("MAX_DRAWDOWN", f"Drawdown reached {current_dd:.1%}", current_dd)
            self.alerts_triggered.add("max_drawdown")

        # Daily loss alert
        daily_loss_pct = abs(daily_pnl) / self.initial_equity if daily_pnl < 0 else 0
        if daily_loss_pct >= self.daily_loss_alert and "daily_loss" not in self.alerts_triggered:
            self.alert_callback("DAILY_LOSS", f"Daily loss reached {daily_loss_pct:.1%}", daily_loss_pct)
            self.alerts_triggered.add("daily_loss")

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        """Get equity curve data."""
        with self._lock:
            return list(self.equity_history)

    def get_trade_stats(self) -> dict:
        """Get comprehensive trade statistics."""
        with self._lock:
            if not self.trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                    "avg_duration": timedelta(0),
                    "total_swap": 0.0,
                }

            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl < 0]

            return {
                "total_trades": len(self.trades),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate": len(wins) / len(self.trades),
                "profit_factor": self._calculate_profit_factor(),
                "avg_win": np.mean([t.pnl for t in wins]) if wins else 0.0,
                "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0.0,
                "largest_win": max((t.pnl for t in wins), default=0.0),
                "largest_loss": min((t.pnl for t in losses), default=0.0),
                "avg_duration": sum((t.duration for t in self.trades), timedelta(0)) / len(self.trades),
                "total_swap": sum(t.swap_earned for t in self.trades),
            }

    def reset(self, new_initial: Optional[float] = None) -> None:
        """Reset all tracking.

        Args:
            new_initial: New initial equity (defaults to current).
        """
        with self._lock:
            self.initial_equity = new_initial or self.equity
            self.high_water_mark = self.initial_equity
            self.max_drawdown = 0.0
            self.equity_history.clear()
            self.daily_returns.clear()
            self.trades.clear()
            self.trades_today = 0
            self.today_date = None
            self.alerts_triggered.clear()
