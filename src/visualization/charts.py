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


def plot_risk_heatmap(
    risk_by_pair: dict[str, dict],
    output_path: Path,
) -> None:
    """Portfolio risk heat map showing exposure and risk metrics by pair.

    Visual dashboard for monitoring:
    - Position size exposure
    - VaR contribution
    - Current drawdown
    - Correlation risk

    Color scale: Green = low risk, Red = high risk
    """
    if not risk_by_pair:
        return

    pairs = list(risk_by_pair.keys())
    metrics = ["exposure_pct", "var_contribution", "drawdown_pct", "volatility"]

    # Build matrix of risk values
    data = []
    for pair in pairs:
        row = []
        for metric in metrics:
            val = risk_by_pair[pair].get(metric, 0)
            row.append(val)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.8)))

    # Normalize each column for color mapping
    with np.errstate(invalid='ignore', divide='ignore'):
        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

    im = ax.imshow(data_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Exposure", "VaR", "Drawdown", "Volatility"])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_title("Portfolio Risk Heat Map")

    # Add text annotations with actual values
    for i in range(len(pairs)):
        for j in range(len(metrics)):
            val = data[i, j]
            fmt = f"{val:.1%}" if metrics[j] != "volatility" else f"{val:.2%}"
            text_color = "white" if data_norm[i, j] > 0.5 else "black"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9, color=text_color)

    fig.colorbar(im, label="Risk Level (higher = more risk)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_var_comparison(
    returns: pd.Series,
    portfolio_value: float,
    output_path: Path,
) -> None:
    """Compare VaR estimates from different methods.

    Shows historical, parametric, and Monte Carlo VaR side by side.
    Helps understand model risk - different methods give different answers.
    """
    from src.risk.risk_metrics import RiskAnalyzer

    analyzer = RiskAnalyzer()
    returns_arr = returns.dropna().values

    # Calculate VaR at different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    methods = ["Historical", "Parametric", "Monte Carlo"]

    results = {method: [] for method in methods}
    results_cvar = {method: [] for method in methods}

    for conf in confidence_levels:
        hist = analyzer.historical_var(returns_arr, conf, portfolio_value)
        param = analyzer.parametric_var(returns_arr, conf, portfolio_value)
        mc = analyzer.monte_carlo_var(returns_arr, conf, portfolio_value, n_simulations=5000)

        results["Historical"].append(hist.var)
        results["Parametric"].append(param.var)
        results["Monte Carlo"].append(mc.var)

        results_cvar["Historical"].append(hist.cvar)
        results_cvar["Parametric"].append(param.cvar)
        results_cvar["Monte Carlo"].append(mc.cvar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # VaR comparison
    x = np.arange(len(confidence_levels))
    width = 0.25
    for i, method in enumerate(methods):
        ax1.bar(x + i * width, results[method], width, label=method, alpha=0.8)

    ax1.set_xlabel("Confidence Level")
    ax1.set_ylabel("Value at Risk ($)")
    ax1.set_title("VaR Comparison by Method")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f"{int(c*100)}%" for c in confidence_levels])
    ax1.legend()

    # CVaR comparison
    for i, method in enumerate(methods):
        ax2.bar(x + i * width, results_cvar[method], width, label=method, alpha=0.8)

    ax2.set_xlabel("Confidence Level")
    ax2.set_ylabel("Conditional VaR ($)")
    ax2.set_title("CVaR (Expected Shortfall) Comparison")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"{int(c*100)}%" for c in confidence_levels])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_circuit_breaker_status(
    status: dict,
    output_path: Path,
) -> None:
    """Visual gauge showing how close we are to each risk limit.

    Like a car dashboard - shows utilization of each limit.
    Green = safe, Yellow = caution, Red = near limit.
    """
    metrics = [
        ("Daily Loss", status.get("daily_loss_utilization", 0)),
        ("Weekly Loss", status.get("weekly_loss_utilization", 0)),
        ("Drawdown", status.get("drawdown_utilization", 0)),
        ("Positions", status.get("position_utilization", 0)),
        ("Pair Exposure", status.get("pair_exposure_utilization", 0)),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [m[0] for m in metrics]
    values = [min(m[1], 1.5) for m in metrics]  # Cap at 150% for display

    colors = []
    for v in values:
        if v < 0.5:
            colors.append("green")
        elif v < 0.8:
            colors.append("yellow")
        else:
            colors.append("red")

    bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor="black")

    # Add 100% limit line
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Limit")

    # Add percentage labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=10)

    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Utilization (100% = Limit)")
    ax.set_title("Risk Limit Utilization Dashboard")

    if status.get("trading_halted"):
        ax.text(0.5, 0.98, "TRADING HALTED", transform=ax.transAxes,
                fontsize=14, color="red", fontweight="bold",
                ha="center", va="top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
