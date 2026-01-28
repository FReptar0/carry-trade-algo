"""Backtesting engine -- replays history and simulates trading.

How it works (simple version):
1. Take historical price data (our synthetic data)
2. Run the strategy on it to get buy/sell signals
3. Simulate executing those trades (with costs, slippage)
4. Track your account balance over time
5. Calculate performance metrics

This is NOT live trading. It's a "what if" simulation:
"If I had used this strategy over the past year, how would I have done?"

Important caveat: backtest results are always better than real trading
because of lookahead bias, curve fitting, and no emotional decisions.
That's why we add costs (commission, slippage) to be more realistic.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.metrics import BacktestMetrics, calculate_metrics
from src.backtest.portfolio import Portfolio
from src.config.settings import RiskConfig, StrategyConfig
from src.strategy.base import Signal, Strategy


class BacktestEngine:
    """Runs a strategy against historical data and produces results.

    Args:
        strategy: The trading strategy to test.
        strategy_config: Strategy parameters.
        risk_config: Risk management parameters.
    """

    def __init__(
        self,
        strategy: Strategy,
        strategy_config: StrategyConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.strategy_config = strategy_config or StrategyConfig()
        self.risk_config = risk_config or RiskConfig()

    def run(self, data: pd.DataFrame, pair_name: str = "UNKNOWN") -> BacktestResult:
        """Run the backtest on a single pair's data.

        Args:
            data: DataFrame with columns: timestamp, open, high, low,
                close, volume, swap_long, swap_short.
            pair_name: Name of the forex pair (for logging).

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        portfolio = Portfolio(
            initial_capital=self.risk_config.initial_capital,
            position_size_pct=self.strategy_config.position_size_pct,
            commission=5.0,
            slippage_pct=0.0001,
        )

        # Step 1: Generate all signals from the strategy
        signals = self.strategy.generate_signals(data)
        signal_map: dict[pd.Timestamp, tuple[Signal, str]] = {
            s.timestamp: (s.signal, s.reason) for s in signals
        }

        # Step 2: Walk through each bar and execute signals
        last_date = None
        for i in range(len(data)):
            ts = data["timestamp"].iloc[i]
            price = data["close"].iloc[i]
            current_date = ts.date()

            # Accrue daily swap (once per calendar day)
            if last_date is not None and current_date != last_date:
                swap_val = data["swap_long"].iloc[i]
                portfolio.accrue_swap(swap_val)
            last_date = current_date

            # Check for signal at this timestamp
            if ts in signal_map:
                action, reason = signal_map[ts]

                if action == Signal.LONG:
                    portfolio.open_position(pair_name, price, ts, "long")
                elif action == Signal.CLOSE:
                    portfolio.close_position(price, ts, reason)

            # Record equity every 24 bars (~daily) to keep dataframe manageable
            if i % 24 == 0:
                portfolio.record_equity(ts, price)

        # Close any remaining position at the end
        if portfolio.position is not None:
            last_price = data["close"].iloc[-1]
            last_ts = data["timestamp"].iloc[-1]
            portfolio.close_position(last_price, last_ts, "End of backtest")

        # Final equity snapshot
        portfolio.record_equity(
            data["timestamp"].iloc[-1], data["close"].iloc[-1]
        )

        # Step 3: Calculate metrics
        equity_df = portfolio.get_equity_df()
        trades_df = portfolio.get_trades_df()
        metrics = calculate_metrics(
            equity_df, trades_df, self.risk_config.initial_capital
        )

        return BacktestResult(
            pair=pair_name,
            equity_df=equity_df,
            trades_df=trades_df,
            metrics=metrics,
        )


class BacktestResult:
    """Container for all backtest outputs.

    Attributes:
        pair: Which pair was tested.
        equity_df: Equity curve over time.
        trades_df: All completed trades.
        metrics: Computed performance metrics.
    """

    def __init__(
        self,
        pair: str,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: BacktestMetrics,
    ) -> None:
        self.pair = pair
        self.equity_df = equity_df
        self.trades_df = trades_df
        self.metrics = metrics

    def summary(self) -> str:
        """Human-readable summary of backtest results."""
        m = self.metrics
        lines = [
            f"=== Backtest: {self.pair} ===",
            f"Total Return:      {m.total_return:>8.2%}",
            f"Annual Return:     {m.annual_return:>8.2%}",
            f"Max Drawdown:      {m.max_drawdown:>8.2%}",
            f"Sharpe Ratio:      {m.sharpe_ratio:>8.2f}",
            f"Sortino Ratio:     {m.sortino_ratio:>8.2f}",
            f"Calmar Ratio:      {m.calmar_ratio:>8.2f}",
            f"",
            f"Total Trades:      {m.total_trades:>8d}",
            f"Win Rate:          {m.win_rate:>8.1%}",
            f"Profit Factor:     {m.profit_factor:>8.2f}",
            f"Avg Win:           {m.avg_win:>8.2f}",
            f"Avg Loss:          {m.avg_loss:>8.2f}",
            f"",
            f"Total Swap Profit: {m.total_swap_profit:>8.2f}",
            f"Swap Contribution: {m.swap_contribution_pct:>8.1f}%",
        ]
        return "\n".join(lines)
