# Carry Trade Algorithm Project

## Project Context
Educational algorithmic trading system focused on forex carry trade strategies.
**ALL 10 PHASES COMPLETE.**
Full system: synthetic data, strategy, backtest, optimization, real data, risk management, paper trading, regime detection, multi-timeframe analysis, exit optimization, and validation.
No real money. No live broker connections. Educational purposes only.

## DFR Location
Full design & functional requirements: `../dfr.md`

## Current Phase Status
See `PHASES.md` for detailed phase tracking.

## Tech Stack
- Python 3.13+ with UV package manager
- pandas, numpy, matplotlib, plotly for data/viz
- pytest for testing
- black, isort, ruff, mypy for code quality

## Key Commands
```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/unit/test_generator.py -v

# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Project Structure
```
src/
├── config/settings.py       - Centralized configuration (dataclasses)
├── data/
│   ├── generator.py         - Synthetic forex data generator
│   ├── loader.py            - Historical data loader (Phase 4)
│   ├── multi_timeframe.py   - Multi-timeframe data loading (Phase 8)
│   └── preprocessor.py      - Data cleaning and transformation
├── strategy/
│   ├── base.py              - Abstract base Strategy class
│   ├── carry_trade.py       - Carry trade strategy V1
│   ├── carry_trade_v2.py    - Carry trade strategy V2 (filtered)
│   ├── carry_trade_v3.py    - Carry trade strategy V3 (best performer)
│   ├── carry_trade_v4.py    - Carry trade strategy V4 (multi-timeframe)
│   ├── exit_manager.py      - Centralized exit logic (Phase 9)
│   └── indicators.py        - Technical indicators (SMA, RSI, ATR, ADX)
├── backtest/
│   ├── engine.py            - Backtesting engine
│   ├── metrics.py           - Performance metrics calculation
│   └── portfolio.py         - Portfolio and position management
├── regime/
│   ├── detector.py          - Regime detection (ADX + ATR) (Phase 7)
│   └── adapters.py          - Parameter adaptation by regime
├── risk/
│   ├── position_sizing.py   - Kelly, fixed fractional, vol-based sizing
│   ├── stop_loss.py         - Stop loss logic
│   ├── circuit_breakers.py  - Risk limits and circuit breakers
│   └── risk_metrics.py      - VaR, drawdown, risk calculations
├── validation/
│   ├── validator.py         - Backtest vs paper comparison (Phase 10)
│   └── protocol.py          - 30-day validation protocol
├── broker/
│   ├── orders.py            - Order management
│   └── simulator.py         - Broker simulation
├── monitoring/
│   └── performance.py       - Real-time performance tracking
├── visualization/
│   └── charts.py            - Static result charts
└── utils/
    └── logger.py            - Logging configuration
```

## Development Guidelines
- Google-style docstrings on all public functions
- Type hints on all new code
- Max 50 lines per function, max 500 lines per file
- Cyclomatic complexity < 10 per function
- 80%+ test coverage minimum
- Never use real money - this is educational only

## Architecture Decisions
- Dataclass-based configuration (no YAML/JSON config files for now)
- Strategy pattern: all strategies inherit from `Strategy` base class
- Backtesting engine is independent of strategy implementation
- Data generators produce pandas DataFrames with standardized columns:
  `timestamp, open, high, low, close, volume, swap_long, swap_short`

## Session Memory
- Project initialized: 2026-01-27
- Phase 1 started: 2026-01-27
- UV installed, Python 3.13 pinned
- Directory structure created
- Phase 1 completed: 2026-01-27
  - src/config/settings.py: DataConfig, PairConfig, StrategyConfig, RiskConfig, BacktestConfig + DEFAULT_PAIRS
  - src/data/generator.py: SyntheticDataGenerator (GBM + mean reversion, static/dynamic swaps)
  - src/data/interest_rates.py: InterestRateSimulator (Vasicek model, discrete meeting steps)
  - 48 tests passing (test_settings, test_generator, test_interest_rates)
  - Exploration notebook: notebooks/01_exploration/synthetic_data_exploration.ipynb
  - Data stats: 6264 hourly bars/pair, 3 pairs, weekends excluded
- Phase 2 completed: 2026-01-28
  - src/strategy/base.py: Strategy ABC, Signal enum, TradeSignal dataclass
  - src/strategy/indicators.py: SMA, RSI, ATR functions
  - src/strategy/carry_trade.py: CarryTradeStrategy v1 (swap+SMA+RSI entry, stop/TP/trailing exit)
  - src/backtest/portfolio.py: Portfolio, Position, Trade classes
  - src/backtest/metrics.py: BacktestMetrics, Sharpe, Sortino, Calmar, carry-specific
  - src/backtest/engine.py: BacktestEngine + BacktestResult
  - scripts/run_backtest.py: Runnable backtest with charts
  - 77 tests passing in 3.5s
  - min_swap_threshold changed from 0.15 to 0.005 (scale mismatch with synthetic data)
  - AUD/JPY showed stop-loss whipsaw problem (238 trades, 12.6% win rate, -55%)
  - USD/JPY conservative (2 trades), EUR/USD correctly avoided (negative carry)
- Phase 3 completed: 2026-01-28
  - src/optimization/grid_search.py: GridSearchOptimizer (itertools.product over param combos)
  - src/optimization/walk_forward.py: WalkForwardAnalyzer (train/test window sliding)
  - src/visualization/charts.py: 6 chart functions (equity+DD, returns dist, trade timeline, heatmap, WF bars, buy&hold)
  - scripts/run_optimization.py: Full optimization pipeline with charts
  - notebooks/02_strategy/backtest_results.ipynb: Phase 2 notebook
  - notebooks/03_analysis/optimization_analysis.ipynb: Phase 3 notebook
  - 87 tests passing
  - Key insight: stop loss 2% → 6% improved AUD/JPY from -55% to -0.24%
  - Walk-forward: 1/7 windows profitable, strategy needs real trends (Phase 4)
- Phase 4 completed: 2026-01-28
  - src/data/loader.py: HistoricalDataLoader (yfinance download, parquet caching, FRED API support)
  - src/data/preprocessor.py: DataQualityReport, validate_data(), clean_data(), compare_synthetic_vs_real()
  - scripts/run_real_data.py: Full real data pipeline (download → validate → backtest → optimize)
  - notebooks/04_real_data/real_data_analysis.ipynb: Phase 4 notebook
  - tests/unit/test_loader.py: 16 tests for loader and preprocessor
  - 103 tests passing
  - Downloaded: 1042 daily bars (2021-2024), ~9700 hourly bars for USD/JPY, AUD/JPY, EUR/USD
  - Data quality: 100% completeness, 0 duplicates across all pairs
  - Synthetic vs Real: Real has higher kurtosis (fat tails), negative skew, trending prices
  - USD/JPY rally 104→161 during 2022-2023 Fed hikes (synthetic data couldn't capture this)
  - Default params on real hourly: USD/JPY -1.46%, AUD/JPY -1.92%, EUR/USD 0% (no trades)
  - Optimized params: stop_loss=8%, SMA=50, position_size=15% → AUD/JPY improved to -0.10%
  - 13 PNG charts generated in results/reports/phase4/
- Phase 5 completed: 2026-01-28
  - src/risk/position_sizing.py: PositionSizer (Kelly, Fixed Fractional, Volatility-based)
  - src/risk/stop_loss.py: AdaptiveStopLoss (ATR-based initial + trailing stops)
  - src/risk/circuit_breakers.py: CircuitBreaker, RiskLimits, PortfolioState, LimitViolation
  - src/risk/risk_metrics.py: RiskAnalyzer (VaR, CVaR, Monte Carlo), TradeExcursion (MAE/MFE)
  - Updated src/visualization/charts.py: plot_risk_heatmap, plot_var_comparison, plot_circuit_breaker_status
  - scripts/run_risk_analysis.py: Full risk management demo with charts
  - notebooks/05_risk/risk_management.ipynb: Phase 5 notebook
  - tests/unit/test_risk.py: 27 tests for all risk components
  - Added scipy dependency for parametric VaR calculation
  - 130 tests passing
  - VaR 95% USD/JPY: $939 (0.94% daily), CVaR: $1,433 (1.43%)
  - Kelly Criterion for 50% win rate, 4%/2.5% win/loss: 5.7% full, 1.4% quarter Kelly
  - ATR-based stops: 2x ATR gives ~2.9% stop distance (adapts to volatility)
  - Circuit breakers: daily 3%, weekly 7%, drawdown 20% limits
  - 3 PNG charts generated in results/reports/phase5/
- Phase 6 completed: 2026-01-28
  - src/broker/orders.py: Order, OrderType, OrderSide, OrderStatus, Fill, Position classes
  - src/broker/simulator.py: BrokerSimulator (market/limit/stop orders, slippage, margin, swap)
  - src/utils/logger.py: TradingLogger, setup_logging(), JSON-formatted structured logs
  - src/monitoring/performance.py: PerformanceMonitor (equity curve, drawdown, alerts, trade stats)
  - scripts/run_paper_trading.py: Full paper trading simulation demo
  - docs/PAPER_TRADING.md: Transition guide with broker integration checklist
  - tests/unit/test_broker.py: 27 tests for broker and monitoring
  - 157 tests passing
  - BrokerSimulator: 50:1 leverage, 0.5-2 pip slippage, $7/lot commission
  - Performance alerts: triggers on max drawdown and daily loss thresholds
  - Paper trading demo: +12.62% return, 19.84% max drawdown on 500 bars

- Phase 7 completed: 2026-01-28
  - src/regime/detector.py: RegimeDetector (ADX + ATR based detection)
  - src/regime/adapters.py: RegimeAdapter (parameter adaptation by regime)
  - VolatilityRegime: HIGH / NORMAL / LOW (ATR vs 60-day average)
  - TrendRegime: STRONG_UP / WEAK_UP / RANGING / WEAK_DOWN / STRONG_DOWN
  - CompositeRegime: 4 states (TREND_LOW_VOL, TREND_HIGH_VOL, RANGE_LOW_VOL, RANGE_HIGH_VOL)
  - Added ADX indicator to src/strategy/indicators.py
  - scripts/run_regime_analysis.py: Regime visualization
  - tests/unit/test_regime.py: 29 tests
  - 186 tests passing
  - USD/JPY: 79.3% TREND_LOW_VOL, 86.9% favorable for trading
  - AUD/JPY: 75.6% TREND_LOW_VOL, 79.2% favorable for trading
- Phase 8 completed: 2026-01-28
  - src/data/multi_timeframe.py: MultiTimeframeLoader (weekly resampling, no lookahead)
  - src/strategy/carry_trade_v4.py: Multi-TF strategy (weekly trend + daily entry)
  - scripts/run_multi_timeframe_backtest.py: V3 vs V4 comparison
  - tests/unit/test_multi_timeframe.py: 23 tests
  - 209 tests passing
  - V4 over-trades (34-35 trades vs V3's 8-10), needs parameter tuning
- Phase 9 completed: 2026-01-28
  - src/strategy/exit_manager.py: ExitManager with multiple exit types
  - Exit types: STOP_LOSS, TRAILING_STOP, TIME_BASED, PROFIT_TARGET, SUPPORT_BREAK, REGIME_CHANGE
  - Profit-scaled trailing: 4 levels (2%/4%/6%/10% profit)
  - scripts/analyze_exits.py: MAE/MFE analysis
  - tests/unit/test_exit_manager.py: 26 tests
  - 235 tests passing
  - USD/JPY Winners: 69.6% MFE capture, 8.29% avg MFE, 70.7 day avg duration
  - AUD/JPY Winners: 61.1% MFE capture, 7.57% avg MFE, 82.5 day avg duration
- Phase 10 completed: 2026-01-29
  - src/validation/validator.py: PaperTradingValidator (backtest vs paper comparison)
  - src/validation/protocol.py: TradingProtocol (30-day validation with checkpoints)
  - ValidationCriteria: 30% return, 25% Sharpe, 20% DD degradation limits
  - AbortCriteria: 15% DD, 5 consecutive losses, 3 circuit breakers
  - Checkpoints at days 7, 14, 21, 30
  - scripts/run_validation_protocol.py: Full 30-day protocol demo
  - tests/integration/test_paper_trading.py: 22 integration tests
  - 257 tests passing

PROJECT COMPLETE - All 10 phases implemented and tested.
