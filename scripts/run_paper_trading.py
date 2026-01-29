"""
Phase 6: Paper Trading Demo
===========================
Simulates live trading with realistic broker behavior.

Usage:
    uv run python scripts/run_paper_trading.py

What this does:
    1. Loads historical price data (simulates streaming)
    2. Runs strategy signals on each bar
    3. Executes orders through broker simulator
    4. Tracks performance in real-time
    5. Logs all activity to files

This is PAPER TRADING - no real money, no real orders.
The goal is to test the full system before live deployment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timedelta

from src.data.loader import HistoricalDataLoader
from src.data.preprocessor import clean_data
from src.strategy.carry_trade import CarryTradeStrategy
from src.config.settings import StrategyConfig
from src.broker.simulator import BrokerSimulator, BrokerConfig
from src.broker.orders import OrderSide, Fill
from src.monitoring.performance import PerformanceMonitor, TradeRecord
from src.utils.logger import TradingLogger
from src.risk.circuit_breakers import CircuitBreaker, RiskLimits, PortfolioState

OUT = Path("results/reports/phase6")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("PHASE 6: Paper Trading Simulation")
    print("=" * 60)

    # ===== SETUP =====
    print("\n--- 1. SYSTEM SETUP ---")

    # Initialize logger
    logger = TradingLogger(log_dir="logs")
    logger.log_startup({"mode": "paper_trading", "pairs": ["USD/JPY"]})

    # Initialize broker
    broker_config = BrokerConfig(
        initial_balance=100000,
        leverage=50.0,
        min_slippage_pips=0.5,
        max_slippage_pips=1.5,
        commission_per_lot=7.0,
        order_latency_ms=(50, 150),
    )
    broker = BrokerSimulator(
        config=broker_config,
        on_fill=lambda f: print(f"  [FILL] {f.order_id} @ {f.price:.2f}"),
    )

    # Initialize strategy
    strategy_config = StrategyConfig(
        stop_loss_pct=0.04,
        take_profit_pct=0.08,
        trend_filter_period=50,
        min_swap_threshold=0.005,
    )
    strategy = CarryTradeStrategy(strategy_config)

    # Initialize performance monitor
    monitor = PerformanceMonitor(
        initial_equity=100000,
        alert_callback=lambda t, m, v: print(f"  [ALERT] {t}: {m}"),
        max_drawdown_alert=0.05,
        daily_loss_alert=0.02,
    )

    # Initialize risk limits
    risk_limits = RiskLimits(
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        max_positions=2,
    )
    circuit_breaker = CircuitBreaker(risk_limits)

    print(f"  Broker: {broker_config.initial_balance:,.0f} balance, {broker_config.leverage}:1 leverage")
    print(f"  Strategy: SL={strategy_config.stop_loss_pct:.0%}, TP={strategy_config.take_profit_pct:.0%}")
    print(f"  Risk: Max DD={risk_limits.max_drawdown_pct:.0%}, Daily loss={risk_limits.max_daily_loss_pct:.0%}")

    # ===== LOAD DATA =====
    print("\n--- 2. LOADING PRICE DATA ---")
    loader = HistoricalDataLoader(cache_dir="data/cache")
    df = clean_data(loader.load_pair("USD/JPY", start="2024-06-01", end="2025-12-31", interval="1h"))
    print(f"  Loaded {len(df)} hourly bars")

    # Limit for demo
    df = df.tail(500)  # Last 500 bars for quick demo
    print(f"  Using last {len(df)} bars for demo")

    # ===== PAPER TRADING LOOP =====
    print("\n--- 3. PAPER TRADING SIMULATION ---")
    print("  (Simulating bar-by-bar execution...)\n")

    current_position = None
    entry_bar = None
    trade_count = 0

    # Pre-generate all signals
    all_signals = strategy.generate_signals(df)
    signal_map = {s.timestamp: s for s in all_signals}

    for i in range(strategy_config.trend_filter_period + 1, len(df)):
        row = df.iloc[i]
        timestamp = row["timestamp"]
        price = row["close"]
        high = row["high"]
        low = row["low"]

        # Update broker with current prices
        prices = {"USD/JPY": price}
        broker.execute_orders(prices)

        # Get account state
        state = broker.get_account_state()

        # Update performance monitor
        monitor.update(
            equity=state.equity,
            balance=state.balance,
            unrealized_pnl=state.unrealized_pnl,
            daily_pnl=state.daily_pnl,
            positions_count=state.positions_count,
            timestamp=timestamp,
        )

        # Check circuit breaker
        portfolio_state = PortfolioState(
            equity=state.equity,
            high_water_mark=monitor.high_water_mark,
            daily_pnl=state.daily_pnl,
            weekly_pnl=state.daily_pnl,  # Simplified
            open_positions=state.positions_count,
        )
        circuit_breaker.check_limits(portfolio_state, timestamp)

        if circuit_breaker.is_trading_halted:
            continue

        # Get strategy signal for this bar
        signal = signal_map.get(timestamp)

        # Execute signal
        pos = broker.get_position("USD/JPY")

        if signal is None:
            continue

        if signal.signal.name == "LONG" and pos is None:
            # Open long position
            order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)
            broker.execute_orders(prices)

            if order.status.name == "FILLED":
                current_position = broker.get_position("USD/JPY")
                entry_bar = i
                trade_count += 1
                logger.log_signal("USD/JPY", "LONG", signal.reason)
                print(f"  [{timestamp.date()}] OPEN LONG @ {price:.2f} ({signal.reason[:30]}...)")

        elif signal.signal.name == "CLOSE" and pos is not None:
            # Close position
            pnl = pos.total_pnl
            close_order = broker.close_position("USD/JPY")
            broker.execute_orders(prices)

            if close_order and close_order.status.name == "FILLED":
                # Record trade
                trade = TradeRecord(
                    timestamp=timestamp,
                    pair="USD/JPY",
                    side="BUY",
                    entry_price=pos.entry_price,
                    exit_price=price,
                    quantity=pos.quantity,
                    pnl=pnl,
                    pnl_pct=pos.return_pct,
                    duration=timedelta(hours=i - entry_bar) if entry_bar else timedelta(0),
                    swap_earned=pos.accumulated_swap,
                )
                monitor.record_trade(trade)
                logger.log_position_close("USD/JPY", pnl, signal.reason)

                status = "WIN" if pnl > 0 else "LOSS"
                print(f"  [{timestamp.date()}] CLOSE @ {price:.2f} ({status}: ${pnl:+.2f})")

                current_position = None
                entry_bar = None

        # Accrue swap at 21:00 UTC
        if timestamp.hour == 21:
            broker.accrue_swap(timestamp)

    # ===== FINAL RESULTS =====
    print("\n--- 4. SIMULATION RESULTS ---")

    final_state = broker.get_account_state()
    snapshot = monitor.get_snapshot()
    trade_stats = monitor.get_trade_stats()

    print(f"\n  Account Summary:")
    print(f"    Initial Balance: ${broker_config.initial_balance:,.2f}")
    print(f"    Final Equity:    ${final_state.equity:,.2f}")
    print(f"    Total Return:    {(final_state.equity / broker_config.initial_balance - 1):+.2%}")
    print(f"    Max Drawdown:    {snapshot.max_drawdown:.2%}")

    print(f"\n  Trade Statistics:")
    print(f"    Total Trades:    {trade_stats['total_trades']}")
    print(f"    Win Rate:        {trade_stats['win_rate']:.1%}")
    print(f"    Profit Factor:   {trade_stats['profit_factor']:.2f}")
    print(f"    Avg Win:         ${trade_stats['avg_win']:.2f}")
    print(f"    Avg Loss:        ${trade_stats['avg_loss']:.2f}")
    print(f"    Total Swap:      ${trade_stats['total_swap']:.2f}")

    # Close any remaining position
    if broker.get_position("USD/JPY"):
        broker.close_position("USD/JPY")
        broker.execute_orders(prices)
        print("\n  (Closed remaining position at end of simulation)")

    logger.log_shutdown("simulation_complete")

    print(f"\n  Logs written to: logs/")
    print("=" * 60)
    print("PAPER TRADING SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
