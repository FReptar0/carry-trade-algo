"""Performance metrics -- measuring how well a strategy did.

After a backtest, you need numbers to answer: "Was this good?"
These metrics give different perspectives on performance:

Return metrics: How much money did you make?
Risk metrics: How much pain did you endure to make it?
Trading metrics: How efficient were your trades?
Carry-specific: How much came from swap vs price movement?

The most important metric: Sharpe Ratio
    = (return - risk_free) / volatility
    It answers: "How much return did I get PER UNIT of risk?"
    - Sharpe < 0.5 = bad
    - Sharpe 0.5-1.0 = okay
    - Sharpe 1.0-2.0 = good
    - Sharpe > 2.0 = great (or suspicious -- check for overfitting)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """All metrics from a single backtest run."""

    # Return metrics
    total_return: float
    annual_return: float

    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Carry trade specific
    avg_swap_received: float
    total_swap_profit: float
    swap_contribution_pct: float


def calculate_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    trading_days: int = 252,
) -> BacktestMetrics:
    """Calculate all performance metrics from backtest results.

    Args:
        equity_df: DataFrame with 'timestamp', 'equity', 'drawdown' columns.
        trades_df: DataFrame with trade records.
        initial_capital: Starting capital.
        trading_days: Trading days per year (252 for forex weekdays).

    Returns:
        BacktestMetrics with all computed values.
    """
    # --- Return metrics ---
    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital

    # Approximate years from data length
    if len(equity_df) > 1:
        days = (equity_df["timestamp"].iloc[-1] - equity_df["timestamp"].iloc[0]).days
        years = max(days / 365.25, 0.01)
    else:
        years = 1.0
    annual_return = (1 + total_return) ** (1 / years) - 1

    # --- Risk metrics ---
    max_drawdown = equity_df["drawdown"].max() if len(equity_df) > 0 else 0.0

    # Daily equity returns for Sharpe/Sortino
    equity_daily = equity_df.set_index("timestamp")["equity"].resample("D").last().dropna()
    daily_returns = equity_daily.pct_change().dropna()

    sharpe_ratio = _sharpe(daily_returns, trading_days)
    sortino_ratio = _sortino(daily_returns, trading_days)
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    # --- Trading metrics ---
    total_trades = len(trades_df)

    if total_trades > 0:
        total_pnl = trades_df["total_pnl"]
        wins = total_pnl[total_pnl > 0]
        losses = total_pnl[total_pnl <= 0]

        win_rate = len(wins) / total_trades
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Carry-specific
        total_swap = trades_df["swap_earned"].sum()
        total_pnl_sum = total_pnl.sum()
        avg_swap = trades_df["swap_earned"].mean()
        swap_pct = (total_swap / total_pnl_sum * 100) if total_pnl_sum != 0 else 0.0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        total_swap = avg_swap = swap_pct = 0.0

    return BacktestMetrics(
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_swap_received=avg_swap,
        total_swap_profit=total_swap,
        swap_contribution_pct=swap_pct,
    )


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    """Sharpe ratio = mean / std, annualized.

    Simple version with risk-free rate = 0.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def _sortino(returns: pd.Series, periods_per_year: int) -> float:
    """Sortino ratio = mean / downside_std, annualized.

    Like Sharpe but only penalizes DOWNSIDE volatility.
    Upside volatility (big gains) doesn't count against you.
    """
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return (returns.mean() / downside_std) * np.sqrt(periods_per_year)
