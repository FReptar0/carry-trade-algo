# DFR: Educational Carry Trade Algorithm
## Design & Functional Requirements Document

**Project:** Carry Trade Algorithm - Educational Bot  
**Version:** 1.0  
**Date:** January 27, 2026  
**Purpose:** Gradual construction of an educational algorithmic trading system from scratch  

---

## 1. CONTEXT AND OBJECTIVES

### 1.1 Project Vision
Develop a complete algorithmic trading system focused on carry trade strategies, starting with educational simulation and evolving toward a scalable system with backtesting, paper trading, and eventually live broker API connection capabilities.

### 1.2 Learning Objectives
- Understand carry trade and swap rate fundamentals
- Master financial strategy backtesting
- Implement robust risk management
- Build a complete pipeline from data to decisions
- Develop software engineering best practices for financial systems

### 1.3 Fundamental Principles
- **Education first:** Every component must be understandable and well-documented
- **Safety:** Never use real money until explicit production phase
- **Modularity:** Independent and reusable components
- **Scalability:** Architecture that allows growth from local to production
- **Testing:** Code tested and validated at each stage

---

## 2. TECHNICAL ARCHITECTURE

### 2.1 Core Technology Stack

**Base Language:**
- Python 3.11+

**Package Manager:**
- **UV** (recommended) - Fast, modern Python package and project manager written in Rust
  - 10-100x faster than pip
  - Automatic lockfiles for reproducibility
  - Built-in Python version management
  - Global cache for disk efficiency
  - Drop-in pip compatibility
  
**UV Installation:**
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

**Alternative:** Traditional `pip` + `venv` if preferred (fully compatible with this project)

**Core Dependencies:**
```toml
# pyproject.toml (managed by UV)
[project]
name = "carry-trade-algo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "plotly>=5.14.0",
    "jupyter>=1.0.0",
    "pytest>=7.3.0",
]

[project.optional-dependencies]
phase2 = [
    "yfinance>=0.2.28",
    "fredapi>=0.5.0",
    "backtesting>=0.3.3",
    "pandas-ta>=0.3.14b",
]

dev = [
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.3.0",
    "ruff>=0.1.0",
]
```

### 2.2 Directory Structure

```
carry-trade-algo/
├── .claude/
│   └── commands/              # Reusable commands for Claude Code
│       ├── test.md
│       ├── analyze-backtest.md
│       └── generate-report.md
├── CLAUDE.md                  # Context documentation for Claude Code
├── README.md
├── pyproject.toml             # Project config (UV managed)
├── uv.lock                    # Dependency lockfile (auto-generated)
├── .python-version            # Pinned Python version (e.g., 3.11)
├── data/
│   ├── raw/                  # Unprocessed data
│   ├── processed/            # Clean and prepared data
│   └── synthetic/            # Synthetic data for phase 1
├── notebooks/
│   ├── 01_exploration/       # Initial data exploration
│   ├── 02_strategy/          # Strategy development
│   └── 03_analysis/          # Results analysis
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py       # Centralized configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py      # Synthetic data generator
│   │   ├── loader.py         # Historical data loader
│   │   └── preprocessor.py   # Cleaning and transformation
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py           # Base class for strategies
│   │   ├── carry_trade.py    # Carry trade implementation
│   │   └── indicators.py     # Custom technical indicators
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py         # Backtesting engine
│   │   ├── metrics.py        # Metrics calculation
│   │   └── portfolio.py      # Portfolio management
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizing.py
│   │   ├── stop_loss.py
│   │   └── risk_metrics.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py         # Results charts
│   │   └── dashboard.py      # Interactive dashboard
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── validators.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── test_data/
├── results/
│   ├── backtests/            # Backtest results
│   ├── reports/              # Generated reports
│   └── logs/                 # Execution logs
└── docs/
    ├── architecture.md
    ├── strategies.md
    └── api_integration.md    # For future phases
```

### 2.3 Data Flow

```
[Data Source] → [Data Loader] → [Preprocessor] → [Strategy Engine]
                                                         ↓
[Dashboard] ← [Visualizer] ← [Metrics Calculator] ← [Backtest Engine]
```

---

## 3. DEVELOPMENT PHASES

### PHASE 1: FOUNDATIONS AND SYNTHETIC DATA (Weeks 1-2)
**Objective:** Establish project base with simulated data

#### 3.1.1 Deliverables
- [x] Project structure initialized
- [ ] Configuration system functional
- [ ] Forex synthetic data generator
- [ ] Simulated interest rate generator
- [ ] Unit tests for generators
- [ ] Synthetic data exploration notebook

#### 3.1.2 Technical Specifications

**Required Synthetic Data:**
```python
# data/synthetic/forex_pair_config.json
{
  "USD/JPY": {
    "price_range": [145.0, 152.0],
    "volatility": 0.008,
    "trend": "sideways",
    "interest_rate_usd": 0.0525,
    "interest_rate_jpy": 0.0025
  },
  "AUD/JPY": {
    "price_range": [95.0, 102.0],
    "volatility": 0.012,
    "trend": "uptrend",
    "interest_rate_aud": 0.0435,
    "interest_rate_jpy": 0.0025
  }
}
```

**Generated Data Format:**
```csv
timestamp,open,high,low,close,volume,swap_long,swap_short
2024-01-01 00:00:00,147.50,147.80,147.40,147.65,1000000,0.25,-0.35
2024-01-01 01:00:00,147.65,147.90,147.55,147.75,1100000,0.25,-0.35
```

#### 3.1.3 Phase 1 Success Criteria
- Generate 1 year of hourly data for 3 pairs (USD/JPY, AUD/JPY, EUR/USD)
- Synthetic data must show realistic volatility patterns
- Swaps correctly calculated based on rate differentials
- 100% test coverage for data generator
- Executable notebook visualizing data characteristics

---

### PHASE 2: BASIC STRATEGY AND BACKTESTING (Weeks 3-4)
**Objective:** Implement first carry trade strategy and backtesting system

#### 3.2.1 Deliverables
- [ ] Base `Strategy` class with defined interface
- [ ] Basic `CarryTradeStrategy` implementation
- [ ] Functional backtesting engine
- [ ] Portfolio management system
- [ ] Essential metrics calculation (Sharpe, Sortino, Drawdown)
- [ ] Complete backtest notebook

#### 3.2.2 Strategy Specifications

**Carry Trade V1 Logic:**
```python
class CarryTradeStrategy(Strategy):
    """
    Basic carry trade strategy with trend filter.
    
    Entry rules:
    1. Positive swap > minimum threshold (0.15 per day)
    2. Price above SMA(50) for long positions
    3. No trend reversal signals (RSI < 70)
    
    Exit rules:
    1. Stop loss: 2% of capital
    2. Take profit: 6% of capital
    3. Trailing stop: activate after 3% profit
    """
```

**Configurable Parameters:**
- `min_swap_threshold`: Minimum required swap (default: 0.15)
- `trend_filter_period`: SMA period (default: 50)
- `rsi_threshold`: Maximum RSI threshold (default: 70)
- `stop_loss_pct`: Stop loss percentage (default: 0.02)
- `take_profit_pct`: Take profit percentage (default: 0.06)
- `position_size_pct`: Position size of capital (default: 0.1)

#### 3.2.3 Required Performance Metrics

```python
class BacktestMetrics:
    """Metrics that must be calculated in each backtest."""
    
    # Return metrics
    total_return: float           # Total cumulative return
    annual_return: float          # Annualized return
    
    # Risk metrics
    max_drawdown: float           # Maximum drawdown
    sharpe_ratio: float           # Sharpe ratio (risk-free rate = 0)
    sortino_ratio: float          # Sortino ratio
    calmar_ratio: float           # Annual return / Max drawdown
    
    # Trading metrics
    total_trades: int             # Total number of trades
    win_rate: float               # Percentage of winning trades
    profit_factor: float          # Gross profit / Gross loss
    avg_win: float                # Average win
    avg_loss: float               # Average loss
    
    # Carry trade specific metrics
    avg_swap_received: float      # Average swap received per day
    total_swap_profit: float      # Total profit from swaps
    swap_contribution_pct: float  # % of profit from swaps vs price
```

#### 3.2.4 Phase 2 Success Criteria
- Backtest runs without errors on 1 year of data
- All metrics calculated correctly
- System respects position sizing rules
- Stop loss and take profit execute correctly
- Strategy performance is positive on synthetic data (baseline)
- Execution time < 5 seconds for 1 year of hourly data

---

### PHASE 3: OPTIMIZATION AND VISUALIZATION (Weeks 5-6)
**Objective:** Optimize parameters and create analysis dashboard

#### 3.3.1 Deliverables
- [ ] Parameter optimization system (grid search)
- [ ] Walk-forward analysis implementation
- [ ] Interactive dashboard with Plotly/Streamlit
- [ ] Automated reporting system
- [ ] Multiple strategy comparison
- [ ] Parameter sensitivity analysis

#### 3.3.2 Required Visualizations

**Essential Charts:**
1. Equity curve with marked drawdowns
2. Returns distribution
3. Pair correlation heatmap
4. Trade timeline (entry/exit)
5. Metrics evolution by period
6. Carry trade vs buy-and-hold comparison

**Interactive Dashboard:**
```python
# Dashboard components
- Currency pair selector
- Analysis period range slider
- Strategy parameter controls
- "Run Backtest" button
- Tabs: Performance / Trades / Risk / Optimization
```

#### 3.3.3 Optimization System

**Grid Search Configuration:**
```python
optimization_config = {
    "min_swap_threshold": [0.10, 0.15, 0.20, 0.25],
    "trend_filter_period": [20, 50, 100, 200],
    "rsi_threshold": [65, 70, 75, 80],
    "stop_loss_pct": [0.015, 0.02, 0.025, 0.03],
    "position_size_pct": [0.05, 0.10, 0.15, 0.20]
}
```

**Walk-Forward Analysis:**
- Training window: 6 months
- Testing window: 1 month
- Step size: 1 month
- Validate optimal parameter stability

#### 3.3.4 Phase 3 Success Criteria
- Grid search finds optimal parameters reproducibly
- Dashboard loads and responds in < 2 seconds
- Walk-forward analysis shows strategy consistency
- PDF reports generated automatically with all charts
- Clear identification of underperformance periods

---

### PHASE 4: REAL DATA AND REFINEMENT (Weeks 7-8)
**Objective:** Integrate real historical data without live APIs

#### 3.4.1 Deliverables
- [ ] yfinance integration for historical data
- [ ] FRED API integration for interest rates
- [ ] Data caching system
- [ ] Data quality validation
- [ ] Strategy recalibration with real data
- [ ] Documentation of synthetic vs real discrepancies

#### 3.4.2 Real Data Sources

**Forex Historical Data:**
```python
# Using yfinance
import yfinance as yf

pairs = ["USDJPY=X", "AUDJPY=X", "EURJPY=X"]
data = yf.download(pairs, start="2020-01-01", end="2024-12-31", interval="1h")
```

**Interest Rates:**
```python
# Using FRED API
from fredapi import Fred

fred = Fred(api_key='YOUR_KEY_HERE')
fed_rate = fred.get_series('FEDFUNDS')
japan_rate = fred.get_series('IRSTCI01JPM156N')
```

#### 3.4.3 Validation Pipeline

**Quality Checks:**
1. Detect data gaps (missing timestamps)
2. Validate price ranges (no extreme outliers)
3. Verify positive volume
4. Identify non-trading days (weekends)
5. Synchronize timestamps from multiple sources

**Caching System:**
```python
# data/cache/ structure
cache/
├── forex/
│   ├── USDJPY_1h_2020-2024.parquet
│   ├── AUDJPY_1h_2020-2024.parquet
│   └── metadata.json
└── rates/
    ├── FEDFUNDS_daily.parquet
    └── metadata.json
```

#### 3.4.4 Phase 4 Success Criteria
- Download 3+ years of historical data for 3 pairs
- Completeness rate > 98% (accounting for weekends)
- Backtest with real data runs without errors
- Performance metrics comparable between synthetic and real data
- Documentation of observed differences
- Cache system reduces load time by 90%

---

### PHASE 5: ADVANCED RISK MANAGEMENT (Weeks 9-10)
**Objective:** Implement robust risk management system

#### 3.5.1 Deliverables
- [ ] Dynamic position sizing (Kelly Criterion, Fixed Fractional)
- [ ] Adaptive stop loss based on volatility (ATR)
- [ ] Circuit breakers (kill switches)
- [ ] Value at Risk (VaR) analysis
- [ ] Portfolio heat map
- [ ] Alert and limits system

#### 3.5.2 Risk Management Components

**Position Sizing Methods:**
```python
class PositionSizer:
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """
        Kelly % = (Win% × Avg Win - Loss% × Avg Loss) / Avg Win
        Use 1/4 Kelly for safety.
        """
        pass
    
    def fixed_fractional(self, capital, risk_per_trade):
        """
        Size = Capital × Risk% / Stop Loss Distance
        """
        pass
    
    def volatility_based(self, capital, atr, atr_multiplier=2):
        """
        Stop distance = ATR × Multiplier
        Size inversely proportional to volatility
        """
        pass
```

**Circuit Breakers:**
```python
class RiskLimits:
    max_daily_loss_pct = 0.03      # Maximum 3% daily loss
    max_weekly_loss_pct = 0.07     # Maximum 7% weekly loss
    max_drawdown_pct = 0.20        # Maximum 20% drawdown
    max_positions = 3               # Maximum 3 simultaneous positions
    max_exposure_per_pair = 0.15   # Maximum 15% in one pair
```

#### 3.5.3 Risk Metrics

**Value at Risk (VaR):**
- VaR 95%: Maximum expected loss in 95% of cases
- VaR 99%: Maximum expected loss in 99% of cases
- Conditional VaR (CVaR): Average loss when exceeding VaR

**Portfolio Metrics:**
- Maximum Adverse Excursion (MAE)
- Maximum Favorable Excursion (MFE)
- MAE/MFE ratio per strategy

#### 3.5.4 Phase 5 Success Criteria
- Position sizing adjusts automatically based on capital
- Circuit breakers stop trading when limits are violated
- Historical backtests respect all risk limits
- VaR calculated and validated with Monte Carlo simulation
- Alert system works correctly
- Complete risk policy documentation

---

### PHASE 6: PAPER TRADING PREPARATION (Weeks 11-12)
**Objective:** Prepare system for real-time simulation

#### 3.6.1 Deliverables
- [ ] Streaming data pipeline architecture
- [ ] Broker simulator (order execution, fills, slippage)
- [ ] Robust logging system
- [ ] Real-time performance monitoring
- [ ] Preparation for broker API integration
- [ ] Paper trading transition documentation

#### 3.6.2 Broker Simulator

**Components:**
```python
class BrokerSimulator:
    def __init__(self, initial_balance, slippage_pips, commission):
        self.balance = initial_balance
        self.positions = []
        self.slippage = slippage_pips
        self.commission = commission
    
    def place_order(self, order_type, pair, size, price):
        """Simulates order execution with realistic slippage."""
        pass
    
    def get_swap(self, pair, position_type):
        """Returns swap for overnight position."""
        pass
    
    def update_positions(self, current_prices):
        """Updates P&L of open positions."""
        pass
    
    def check_margin(self):
        """Verifies sufficient margin."""
        pass
```

**Latency Simulation:**
- Order placement: 50-200ms
- Price updates: every 1 second
- Position updates: every 5 seconds

#### 3.6.3 Logging System

**Log Structure:**
```
logs/
├── trades/
│   └── 2024-01-27_trades.log
├── errors/
│   └── 2024-01-27_errors.log
├── performance/
│   └── 2024-01-27_metrics.log
└── system/
    └── 2024-01-27_system.log
```

**Information to Log:**
- Every trading signal generated
- Every order executed (price, size, timestamp)
- P&L updated every 5 minutes
- Errors and exceptions with traceback
- System metrics (CPU, memory)

#### 3.6.4 Phase 6 Success Criteria
- Simulator executes orders with realistic latency
- Slippage applied correctly (0.5-2 pips)
- Logs contain all information necessary for audit
- System can run 24hrs without errors
- Documentation ready to connect with real broker
- Smoke test of broker API integration (without executing)

---

## 4. CONFIGURATION AND STANDARDS

### 4.1 Centralized Configuration

**config/settings.py:**
```python
from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    synthetic_data_path: str = "data/synthetic"
    cache_path: str = "data/cache"
    lookback_days: int = 365
    
@dataclass
class StrategyConfig:
    name: str = "carry_trade_v1"
    pairs: List[str] = ("USD/JPY", "AUD/JPY", "EUR/USD")
    min_swap_threshold: float = 0.15
    trend_filter_period: int = 50
    
@dataclass
class RiskConfig:
    initial_capital: float = 10000.0  # Virtual USD
    max_position_size: float = 0.1    # 10% of capital
    stop_loss_pct: float = 0.02       # 2%
    max_drawdown_pct: float = 0.20    # 20%
    
@dataclass
class BacktestConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    commission_per_lot: float = 5.0
    slippage_pips: float = 1.0
```

### 4.2 Code Standards

**Linting and Formatting:**
```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 88
target-version = "py311"
```

**Mandatory Type Hints:**
```python
def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.0
) -> float:
    """Calculates Sharpe ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annualized risk-free rate
        
    Returns:
        Sharpe ratio
    """
    pass
```

**Docstrings:**
- Use Google style format
- Document all public methods
- Include examples in docstrings for key functions

### 4.3 Testing Strategy

**Testing Levels:**
1. **Unit Tests:** 80%+ minimum coverage
2. **Integration Tests:** Complete flow from data to results
3. **Performance Tests:** Backtest execution time
4. **Regression Tests:** Validate changes don't break functionality

**Test Example:**
```python
def test_carry_trade_strategy_basic():
    """Test that strategy generates correct signals."""
    # Arrange
    data = generate_synthetic_data(pair="USD/JPY", days=100)
    strategy = CarryTradeStrategy(min_swap=0.15)
    
    # Act
    signals = strategy.generate_signals(data)
    
    # Assert
    assert len(signals) > 0
    assert all(s in ['long', 'short', 'close'] for s in signals)
    assert strategy.validate_parameters()
```

---

## 5. REQUIRED DOCUMENTATION

### 5.1 CLAUDE.md (For Claude Code)

**Essential Content:**
```markdown
# Carry Trade Algorithm Project

## Project Context
Educational algorithmic trading system focused on forex carry trade strategies.
Currently in Phase [X]. Using synthetic/historical data only.

## Key Commands
- Run backtest: `uv run python -m src.backtest.engine --config config/default.yaml`
- Generate report: `uv run python -m src.visualization.dashboard`
- Run tests: `uv run pytest tests/ -v`

## Project Structure
[Explain purpose of each main module]

## Development Guidelines
[Code standards, naming conventions, etc.]

## Common Tasks
[Frequent tasks with specific commands]
```

### 5.2 README.md

**Sections:**
1. Project description
2. Requirements and installation
3. Quick start
4. Project structure
5. Completed phases
6. Roadmap
7. Contributing
8. License

### 5.3 Technical Documentation

**docs/strategies.md:**
- Mathematical description of each strategy
- Entry/exit logic
- Parameters and valid ranges
- Historical performance

**docs/architecture.md:**
- Flow diagrams
- Architectural decisions and rationale
- Design patterns used

---

## 6. GLOBAL ACCEPTANCE CRITERIA

### 6.1 Per Phase
Each phase must meet:
- [ ] All deliverables completed
- [ ] Tests pass (coverage > 80%)
- [ ] Documentation updated
- [ ] Internal code review (self-review)
- [ ] Performance benchmarks met
- [ ] No critical blockers identified

### 6.2 Code Quality
- Cyclomatic complexity < 10 per function
- No functions > 50 lines
- No files > 500 lines
- Type hints in all new code
- Docstrings in public functions

### 6.3 Performance
- 1 year backtest: < 5 seconds
- Grid search (100 combinations): < 2 minutes
- Dashboard load time: < 3 seconds
- Memory usage: < 500MB for typical dataset

---

## 7. RISKS AND MITIGATIONS

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Unrepresentative synthetic data | Medium | High | Validate with multiple statistical metrics, compare with real data in Phase 4 |
| Overfitting in optimization | High | High | Walk-forward analysis, out-of-sample testing, parameter regularization |
| Poor performance with real data | Medium | High | Use conservatism in backtests (high costs, realistic slippage) |
| Bugs in trading logic | Medium | Critical | Exhaustive testing, code reviews, prolonged dry-run |

### 7.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | High | Medium | Strictly adhere to defined phases |
| Time underestimation | Medium | Medium | 20% buffers in estimates |
| Lack of quality historical data | Low | High | Multiple data sources, cross-validation |

---

## 8. IMMEDIATE NEXT STEPS

### 8.1 Sprint 1 (Days 1-3)
1. Initialize repository with directory structure
2. Setup virtual environment and base dependencies with UV
3. Create `CLAUDE.md` with initial context
4. Implement `DataGenerator` for synthetic data
5. Write tests for generator
6. Create first exploration notebook

### 8.2 Sprint 2 (Days 4-7)
1. Implement centralized configuration
2. Develop base `Strategy` class
3. Implement basic `CarryTradeStrategy` version
4. Create backtest engine structure
5. Integration tests for complete flow

### 8.3 Startup Commands

```bash
# Initial setup with UV
mkdir -p carry-trade-algo && cd carry-trade-algo
uv init
uv python pin 3.11

# Add base dependencies
uv add pandas numpy matplotlib plotly jupyter pytest

# Add development dependencies
uv add --dev black isort mypy ruff

# Initialize git
git init
git add .
git commit -m "Initial project structure"

# Run first test
uv run pytest
```

---

## 9. PROJECT SUCCESS METRICS

### 9.1 Quantitative Objectives
- **By end of Phase 2:** Basic strategy with Sharpe ratio > 0.5 on synthetic data
- **By end of Phase 4:** Functional strategy with Sharpe ratio > 0.8 on real data
- **By end of Phase 6:** Complete system running 24hrs without errors

### 9.2 Qualitative Objectives
- Modular and maintainable code
- Complete and clear documentation
- System scalable toward production
- Deep learning of algorithmic trading

---

## 10. CONTACT AND RESOURCES

### 10.1 Learning Resources
- **Recommended book:** "Python for Algorithmic Trading" - Yves Hilpisch
- **FRED documentation:** https://fred.stlouisfed.org/docs/api/fred/
- **Backtesting.py docs:** https://kernc.github.io/backtesting.py/
- **UV documentation:** https://docs.astral.sh/uv/

### 10.2 Claude Code Integration
- This project is optimized for Claude Code
- Use commands in `.claude/commands/` for repetitive tasks
- Keep `CLAUDE.md` updated with relevant context

---

## APPENDIX A: GLOSSARY

**Carry Trade:** Strategy exploiting interest rate differentials  
**Swap:** Charge/credit for holding positions overnight  
**Sharpe Ratio:** Risk-adjusted return  
**Sortino Ratio:** Sharpe that only penalizes downside volatility  
**Drawdown:** Decline from equity peak  
**ATR:** Average True Range (volatility measure)  
**VaR:** Value at Risk (maximum probable loss)  
**Slippage:** Difference between expected and executed price  

---

**END OF DOCUMENT**

*This DFR is a living document. It will be updated upon completing each phase.*

