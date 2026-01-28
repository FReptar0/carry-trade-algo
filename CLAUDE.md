# Carry Trade Algorithm Project

## Project Context
Educational algorithmic trading system focused on forex carry trade strategies.
**Phases 1-3 complete. Ready for Phase 4 - Real Data & Refinement.**
Using synthetic data only. No real money. No live broker connections.

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
│   └── preprocessor.py      - Data cleaning and transformation
├── strategy/
│   ├── base.py              - Abstract base Strategy class
│   ├── carry_trade.py       - Carry trade strategy implementation
│   └── indicators.py        - Technical indicators (SMA, RSI, ATR)
├── backtest/
│   ├── engine.py            - Backtesting engine
│   ├── metrics.py           - Performance metrics calculation
│   └── portfolio.py         - Portfolio and position management
├── risk/
│   ├── position_sizing.py   - Kelly, fixed fractional, vol-based sizing
│   ├── stop_loss.py         - Stop loss logic
│   └── risk_metrics.py      - VaR, drawdown, risk calculations
├── visualization/
│   ├── charts.py            - Static result charts
│   └── dashboard.py         - Interactive Plotly/Streamlit dashboard
└── utils/
    ├── logger.py            - Logging configuration
    └── validators.py        - Input validation helpers
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
