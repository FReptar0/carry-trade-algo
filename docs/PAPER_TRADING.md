# Paper Trading Transition Guide

This document describes how to transition from backtesting to paper trading and eventually to live trading.

## Overview

Paper trading is a critical step between backtesting and live trading. It validates that your system works in a realistic environment without risking real money.

```
Backtesting -> Paper Trading -> Live Trading (small) -> Live Trading (full)
```

## What Paper Trading Adds

| Feature | Backtesting | Paper Trading | Live Trading |
|---------|-------------|---------------|--------------|
| Historical data | Yes | Yes (simulated stream) | Real-time stream |
| Slippage | Simulated | Realistic simulation | Actual market impact |
| Latency | None | Simulated (50-200ms) | Real network latency |
| Order rejection | None | Margin-based | Exchange rules |
| Swap accrual | Estimated | Time-based | Broker-specific |
| Position limits | None | Configurable | Broker/exchange rules |

## Components

### 1. Broker Simulator (`src/broker/simulator.py`)

Simulates a forex broker with:
- Market, limit, and stop orders
- Realistic slippage (0.5-2 pips)
- Commission charges ($7/lot round trip)
- Margin requirements (50:1 leverage)
- Swap accrual at 21:00 UTC

```python
from src.broker.simulator import BrokerSimulator, BrokerConfig

config = BrokerConfig(
    initial_balance=100000,
    leverage=50.0,
    min_slippage_pips=0.5,
    max_slippage_pips=2.0,
    commission_per_lot=7.0,
)
broker = BrokerSimulator(config)

# Place order
order = broker.submit_market_order("USD/JPY", OrderSide.BUY, 0.1)

# Execute at current price
fills = broker.execute_orders({"USD/JPY": 150.50})
```

### 2. Logging System (`src/utils/logger.py`)

Structured logging for audit and analysis:

```
logs/
├── trades/       # Every signal, order, fill
├── errors/       # Exceptions with traceback
├── performance/  # P&L updates, metrics
└── system/       # Application lifecycle
```

```python
from src.utils.logger import TradingLogger

logger = TradingLogger()
logger.log_signal("USD/JPY", "LONG", "High positive carry")
logger.log_fill(fill_dict)
logger.log_pnl(equity=105000, daily_pnl=500, ...)
```

### 3. Performance Monitor (`src/monitoring/performance.py`)

Real-time tracking:
- Equity curve
- Drawdown (current and maximum)
- Win rate and profit factor
- Rolling Sharpe ratio
- Trade statistics

```python
from src.monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor(
    initial_equity=100000,
    alert_callback=lambda t, m, v: print(f"ALERT: {m}"),
    max_drawdown_alert=0.10,
)

monitor.update(equity=95000, balance=95000, ...)
snapshot = monitor.get_snapshot()
print(f"Drawdown: {snapshot.drawdown:.2%}")
```

### 4. Circuit Breakers (`src/risk/circuit_breakers.py`)

Automatic trading halt when limits are breached:
- Daily loss > 3%
- Weekly loss > 7%
- Drawdown > 20%
- Too many positions

## Running Paper Trading

```bash
uv run python scripts/run_paper_trading.py
```

The script:
1. Loads historical data (simulates streaming)
2. Runs strategy on each bar
3. Executes orders through broker simulator
4. Monitors performance in real-time
5. Logs all activity

## Transitioning to Live Trading

### Prerequisites

Before live trading, verify:

- [ ] Paper trading ran 1+ month without critical errors
- [ ] Win rate and profit factor match backtested expectations (within 20%)
- [ ] Maximum drawdown stayed within limits
- [ ] All circuit breakers functioned correctly
- [ ] Logging captured all necessary audit information
- [ ] Swap accrual matched broker's actual rates (within 10%)

### Broker Integration

This educational project does NOT include live broker integration. For live trading, you would need to:

1. **Choose a broker** with API access:
   - OANDA (popular for forex API)
   - Interactive Brokers (professional-grade)
   - Alpaca (stocks/crypto, easy API)

2. **Replace `BrokerSimulator`** with broker API client:
   ```python
   # Replace this:
   from src.broker.simulator import BrokerSimulator
   broker = BrokerSimulator(config)

   # With this (example):
   from oandapyV20 import API
   import oandapyV20.endpoints.orders as orders
   client = API(access_token="your_token")
   ```

3. **Handle real-time data**:
   - WebSocket connections for price streaming
   - Reconnection logic for network failures
   - Timestamp synchronization

4. **Add safety measures**:
   - API key encryption
   - Rate limiting
   - Order confirmation
   - Emergency stop mechanism

### Risk Checklist for Live Trading

Before deploying live:

- [ ] Start with minimum position size (0.01 lot)
- [ ] Set broker-level stop loss
- [ ] Monitor first trades manually
- [ ] Have a kill switch ready
- [ ] Keep capital you can afford to lose
- [ ] Document everything

## Differences from Live

Paper trading cannot perfectly simulate:

1. **Liquidity**: Real markets have limited depth
2. **Market impact**: Large orders move price
3. **News events**: Gaps and extreme volatility
4. **Requotes**: Broker may reject your price
5. **Outages**: Network and exchange failures
6. **Emotional factors**: Real money feels different

Always expect live performance to be worse than paper trading.

## Recommended Testing Period

| Phase | Duration | Position Size | Monitoring |
|-------|----------|---------------|------------|
| Paper | 4+ weeks | N/A | Daily review |
| Live (tiny) | 2+ weeks | 0.01 lot | Per-trade review |
| Live (small) | 4+ weeks | 0.1 lot | Daily review |
| Live (target) | Ongoing | Risk-based | Weekly review |

## Conclusion

Paper trading is not optional. It's the final validation step that catches issues backtesting misses:
- Order execution logic bugs
- Position sizing errors
- Risk limit implementation
- Data handling edge cases

Never skip paper trading. Your capital depends on it.
