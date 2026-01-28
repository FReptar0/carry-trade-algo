"""Chart generation for backtest and optimization results.

All charts save to PNG files. Each function produces one chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult


def plot_equity_with_drawdown(
    result: BacktestResult, output_path: Path
) -> None:
    """Equity curve on top, drawdown on bottom."""
    eq = result.equity_df
    if len(eq) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    ax1.plot(eq["timestamp"], eq["equity"], linewidth=1.2)
    ax1.axhline(eq["equity"].iloc[0], color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(f"{result.pair} - Equity Curve")
    ax1.set_ylabel("Account Value ($)")

    ax2.fill_between(eq["timestamp"], 0, -eq["drawdown"] * 100, color="red", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_returns_distribution(
    result: BacktestResult, output_path: Path
) -> None:
    """Histogram of daily equity returns."""
    eq = result.equity_df
    if len(eq) < 2:
        return

    returns = eq["equity"].pct_change().dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns, bins=50, alpha=0.7, edgecolor="black", linewidth=0.3, density=True)
    ax.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
    ax.set_title(f"{result.pair} - Daily Returns Distribution")
    ax.set_xlabel("Return")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_trade_timeline(
    result: BacktestResult, price_data: pd.DataFrame, output_path: Path
) -> None:
    """Price chart with trade entry/exit markers."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(price_data["timestamp"], price_data["close"], linewidth=0.4, alpha=0.6, color="gray")

    trades = result.trades_df
    if len(trades) > 0:
        wins = trades[trades["total_pnl"] > 0]
        losses = trades[trades["total_pnl"] <= 0]

        if len(wins) > 0:
            ax.scatter(wins["entry_time"], wins["entry_price"],
                      marker="^", color="green", s=40, zorder=5, label="Win entry")
            ax.scatter(wins["exit_time"], wins["exit_price"],
                      marker="v", color="darkgreen", s=40, zorder=5, label="Win exit")
        if len(losses) > 0:
            ax.scatter(losses["entry_time"], losses["entry_price"],
                      marker="^", color="orange", s=40, zorder=5, label="Loss entry")
            ax.scatter(losses["exit_time"], losses["exit_price"],
                      marker="v", color="red", s=40, zorder=5, label="Loss exit")
        ax.legend()

    ax.set_title(f"{result.pair} - Trade Timeline")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_optimization_heatmap(
    results_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    output_path: Path,
) -> None:
    """2D heatmap showing how two parameters affect a metric.

    This is a sensitivity analysis: dark = bad, bright = good.
    Helps you see which parameter ranges work and which don't.
    """
    pivot = results_df.pivot_table(
        values=metric, index=y_param, columns=x_param, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3f}" if isinstance(v, float) else str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3f}" if isinstance(v, float) else str(v) for v in pivot.index])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"Parameter Sensitivity: {metric}")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_walk_forward(
    wf_df: pd.DataFrame, output_path: Path
) -> None:
    """Bar chart comparing train vs test Sharpe for each window."""
    if len(wf_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(wf_df))
    width = 0.35

    ax.bar([i - width / 2 for i in x], wf_df["train_sharpe"], width, label="Train Sharpe", alpha=0.8)
    ax.bar([i + width / 2 for i in x], wf_df["test_sharpe"], width, label="Test Sharpe", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Window")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward: Train vs Test Performance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(wf_df["test_period"], rotation=45, ha="right", fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_buy_and_hold_comparison(
    result: BacktestResult, price_data: pd.DataFrame, output_path: Path
) -> None:
    """Compare strategy equity vs simply buying and holding.

    Buy-and-hold = buy on day 1, sell on last day. No trading rules.
    If your strategy can't beat buy-and-hold, what's the point?
    """
    eq = result.equity_df
    if len(eq) == 0:
        return

    initial_price = price_data["close"].iloc[0]
    initial_capital = eq["equity"].iloc[0]

    # Buy and hold equity: invest all capital at day 1
    bh_equity = (price_data["close"] / initial_price) * initial_capital

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq["timestamp"], eq["equity"], linewidth=1.2, label="Carry Trade Strategy")

    # Resample buy-and-hold to match equity timestamps roughly
    bh_daily = pd.DataFrame({"timestamp": price_data["timestamp"], "equity": bh_equity})
    bh_daily = bh_daily.set_index("timestamp").resample("D").last().dropna().reset_index()
    ax.plot(bh_daily["timestamp"], bh_daily["equity"], linewidth=1.2,
            linestyle="--", label="Buy & Hold", alpha=0.8)

    ax.axhline(initial_capital, color="gray", linestyle=":", alpha=0.4)
    ax.set_title(f"{result.pair} - Strategy vs Buy & Hold")
    ax.set_ylabel("Account Value ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
