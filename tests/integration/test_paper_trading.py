"""Integration tests for paper trading validation."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.validation.validator import (
    PaperTradingValidator,
    ValidationMetrics,
    ValidationCriteria,
)
from src.validation.protocol import (
    TradingProtocol,
    ProtocolDay,
    ProtocolCheckpoint,
    ProtocolStatus,
    AbortCriteria,
)
from src.backtest.metrics import BacktestMetrics


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def good_backtest_metrics() -> BacktestMetrics:
    """Create good backtest metrics."""
    return BacktestMetrics(
        total_return=0.15,  # 15%
        annual_return=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.08,  # 8%
        calmar_ratio=1.875,
        total_trades=20,
        win_rate=0.60,
        profit_factor=1.8,
        avg_win=0.025,
        avg_loss=-0.015,
        avg_swap_received=50.0,
        total_swap_profit=1000.0,
        swap_contribution_pct=0.20,
    )


@pytest.fixture
def good_paper_metrics() -> BacktestMetrics:
    """Create good paper trading metrics (slightly worse than backtest)."""
    return BacktestMetrics(
        total_return=0.12,  # 12% (20% degradation from 15%)
        annual_return=0.12,
        sharpe_ratio=1.2,  # 20% degradation from 1.5
        sortino_ratio=1.6,
        max_drawdown=0.09,  # Slightly worse
        calmar_ratio=1.33,
        total_trades=18,
        win_rate=0.55,
        profit_factor=1.6,
        avg_win=0.023,
        avg_loss=-0.016,
        avg_swap_received=45.0,
        total_swap_profit=810.0,
        swap_contribution_pct=0.18,
    )


@pytest.fixture
def bad_paper_metrics() -> BacktestMetrics:
    """Create bad paper trading metrics (fails validation)."""
    return BacktestMetrics(
        total_return=0.05,  # 5% (67% degradation from 15%)
        annual_return=0.05,
        sharpe_ratio=0.5,  # 67% degradation from 1.5
        sortino_ratio=0.7,
        max_drawdown=0.15,  # Much worse
        calmar_ratio=0.33,
        total_trades=25,  # More trades (over-trading)
        win_rate=0.40,
        profit_factor=1.1,
        avg_win=0.018,
        avg_loss=-0.020,
        avg_swap_received=20.0,
        total_swap_profit=500.0,
        swap_contribution_pct=0.10,
    )


@pytest.fixture
def validator() -> PaperTradingValidator:
    """Create validator with default criteria."""
    return PaperTradingValidator()


@pytest.fixture
def protocol() -> TradingProtocol:
    """Create protocol with default settings."""
    return TradingProtocol(duration_days=30, initial_equity=100000.0)


# ============================================================================
# Validator Tests
# ============================================================================


class TestPaperTradingValidator:
    """Tests for PaperTradingValidator."""

    def test_validator_initialization(self):
        """Validator initializes with default criteria."""
        validator = PaperTradingValidator()
        assert validator.criteria.max_return_degradation == 0.30
        assert validator.criteria.max_sharpe_degradation == 0.25

    def test_validator_custom_criteria(self):
        """Validator accepts custom criteria."""
        criteria = ValidationCriteria(
            max_return_degradation=0.20,
            max_sharpe_degradation=0.15,
        )
        validator = PaperTradingValidator(criteria=criteria)

        assert validator.criteria.max_return_degradation == 0.20
        assert validator.criteria.max_sharpe_degradation == 0.15

    def test_validation_passes_good_metrics(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Validation passes when paper results are acceptable."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)

        assert metrics.is_valid is True
        assert len(metrics.failure_reasons) == 0

    def test_validation_fails_bad_metrics(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        bad_paper_metrics: BacktestMetrics,
    ):
        """Validation fails when paper results are too degraded."""
        metrics = validator.compare(good_backtest_metrics, bad_paper_metrics)

        assert metrics.is_valid is False
        assert len(metrics.failure_reasons) > 0

    def test_return_degradation_calculation(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Return degradation is calculated correctly."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)

        # Backtest: 15%, Paper: 12% -> (15-12)/15 = 20%
        assert metrics.return_degradation == pytest.approx(0.20, rel=0.1)

    def test_sharpe_degradation_calculation(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Sharpe degradation is calculated correctly."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)

        # Backtest: 1.5, Paper: 1.2 -> (1.5-1.2)/1.5 = 20%
        assert metrics.sharpe_degradation == pytest.approx(0.20, rel=0.1)

    def test_drawdown_increase_calculation(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Drawdown increase is calculated correctly."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)

        # Backtest: 8%, Paper: 9% -> (9-8)/8 = 12.5%
        assert metrics.dd_increase == pytest.approx(0.125, rel=0.1)

    def test_trade_count_match(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Trade count match is detected correctly."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)

        # 20 vs 18 trades = within 2, should match
        assert metrics.trade_count_match is True

    def test_report_generation(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Report is generated correctly."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)
        report = validator.generate_report(metrics)

        assert "PASS" in report
        assert "Total Return" in report
        assert "Sharpe Ratio" in report


class TestValidationMetrics:
    """Tests for ValidationMetrics."""

    def test_summary_property(
        self,
        validator: PaperTradingValidator,
        good_backtest_metrics: BacktestMetrics,
        good_paper_metrics: BacktestMetrics,
    ):
        """Summary property returns formatted string."""
        metrics = validator.compare(good_backtest_metrics, good_paper_metrics)

        summary = metrics.summary
        assert "PASS" in summary
        assert "%" in summary


# ============================================================================
# Protocol Tests
# ============================================================================


class TestTradingProtocol:
    """Tests for TradingProtocol."""

    def test_protocol_initialization(self):
        """Protocol initializes correctly."""
        protocol = TradingProtocol(duration_days=30)

        assert protocol.duration == 30
        assert protocol.status == ProtocolStatus.NOT_STARTED
        assert len(protocol.days) == 0

    def test_protocol_start(self, protocol: TradingProtocol):
        """Protocol starts correctly."""
        protocol.start()

        assert protocol.status == ProtocolStatus.RUNNING
        assert protocol.start_date == date.today()

    def test_record_day(self, protocol: TradingProtocol):
        """Days are recorded correctly."""
        protocol.start()

        day = ProtocolDay(
            date=date.today(),
            starting_equity=100000,
            ending_equity=101000,
            daily_pnl=1000,
            daily_return=0.01,
            trades_opened=1,
            trades_closed=1,
            max_drawdown_today=0.005,
            regime="trend_low_vol",
            circuit_breaker_triggered=False,
        )

        protocol.record_day(day)
        assert len(protocol.days) == 1
        assert protocol.days[0].daily_pnl == 1000

    def test_cannot_record_before_start(self, protocol: TradingProtocol):
        """Cannot record days before starting."""
        day = ProtocolDay(
            date=date.today(),
            starting_equity=100000,
            ending_equity=101000,
            daily_pnl=1000,
            daily_return=0.01,
            trades_opened=0,
            trades_closed=0,
            max_drawdown_today=0,
            regime="trend_low_vol",
            circuit_breaker_triggered=False,
        )

        with pytest.raises(ValueError):
            protocol.record_day(day)


class TestAbortConditions:
    """Tests for protocol abort conditions."""

    def test_abort_on_max_drawdown(self, protocol: TradingProtocol):
        """Protocol aborts on max drawdown."""
        protocol.start()

        # Record days with increasing loss
        equity = 100000
        for i in range(5):
            equity -= 4000  # 4% loss each day
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=equity + 4000,
                ending_equity=equity,
                daily_pnl=-4000,
                daily_return=-0.04,
                trades_opened=1,
                trades_closed=1,
                max_drawdown_today=0.04,
                regime="range_high_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        should_abort, reason = protocol.check_abort_conditions()
        assert should_abort is True
        assert "drawdown" in reason.lower()

    def test_abort_on_consecutive_losses(self, protocol: TradingProtocol):
        """Protocol aborts on consecutive losing days."""
        protocol.abort_criteria.consecutive_losing_days = 3
        protocol.start()

        # Record 3 losing days
        for i in range(3):
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=100000 - i * 500,
                ending_equity=100000 - (i + 1) * 500,
                daily_pnl=-500,
                daily_return=-0.005,
                trades_opened=0,
                trades_closed=1,
                max_drawdown_today=0.005,
                regime="trend_low_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        should_abort, reason = protocol.check_abort_conditions()
        assert should_abort is True
        assert "consecutive" in reason.lower()

    def test_no_abort_healthy_protocol(self, protocol: TradingProtocol):
        """Protocol doesn't abort when healthy."""
        protocol.start()

        # Record 5 good days
        for i in range(5):
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=100000 + i * 500,
                ending_equity=100000 + (i + 1) * 500,
                daily_pnl=500,
                daily_return=0.005,
                trades_opened=1,
                trades_closed=1,
                max_drawdown_today=0.002,
                regime="trend_low_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        should_abort, reason = protocol.check_abort_conditions()
        assert should_abort is False
        assert reason is None


class TestProtocolCheckpoints:
    """Tests for protocol checkpoints."""

    def test_checkpoint_day_7(self, protocol: TradingProtocol):
        """Day 7 checkpoint works correctly."""
        protocol.start()

        # Record 7 days
        for i in range(7):
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=100000 + i * 200,
                ending_equity=100000 + (i + 1) * 200,
                daily_pnl=200,
                daily_return=0.002,
                trades_opened=1,
                trades_closed=1,
                max_drawdown_today=0.001,
                regime="trend_low_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        checkpoint = protocol.run_checkpoint(7)

        assert checkpoint.day == 7
        assert checkpoint.cumulative_return > 0
        assert checkpoint.recommendation == "continue"

    def test_checkpoint_identifies_issues(self, protocol: TradingProtocol):
        """Checkpoint identifies issues correctly."""
        protocol.start()

        # Record 7 poor days with significant losses (2% daily loss = 14% cumulative)
        # This should trigger abort due to cumulative_return < -0.10
        for i in range(7):
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=100000 - i * 2000,
                ending_equity=100000 - (i + 1) * 2000,
                daily_pnl=-2000,
                daily_return=-0.02,
                trades_opened=1,
                trades_closed=1,
                max_drawdown_today=0.02,
                regime="range_high_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        checkpoint = protocol.run_checkpoint(7)

        assert len(checkpoint.issues_identified) > 0
        assert checkpoint.recommendation in ["pause", "abort"]


class TestProtocolCompletion:
    """Tests for protocol completion."""

    def test_final_report(self, protocol: TradingProtocol):
        """Final report is generated correctly."""
        protocol.start()

        # Record 30 days
        for i in range(30):
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=100000 + i * 100,
                ending_equity=100000 + (i + 1) * 100,
                daily_pnl=100,
                daily_return=0.001,
                trades_opened=1 if i % 5 == 0 else 0,
                trades_closed=1 if i % 5 == 0 else 0,
                max_drawdown_today=0.001,
                regime="trend_low_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        protocol.complete()
        report = protocol.final_report()

        assert report["status"] == "completed"
        assert report["duration_days"] == 30
        assert report["total_return"] > 0

    def test_progress_tracking(self, protocol: TradingProtocol):
        """Progress tracking works correctly."""
        protocol.start()

        # Record 10 days
        for i in range(10):
            day = ProtocolDay(
                date=date.today() + timedelta(days=i),
                starting_equity=100000,
                ending_equity=100500,
                daily_pnl=500,
                daily_return=0.005,
                trades_opened=0,
                trades_closed=0,
                max_drawdown_today=0,
                regime="trend_low_vol",
                circuit_breaker_triggered=False,
            )
            protocol.record_day(day)

        progress = protocol.get_progress()

        assert progress["days_completed"] == 10
        assert progress["days_remaining"] == 20
        assert progress["progress_pct"] == pytest.approx(33.33, rel=0.1)
        assert progress["next_checkpoint"] == 14


class TestProtocolDay:
    """Tests for ProtocolDay dataclass."""

    def test_is_profitable(self):
        """is_profitable property works correctly."""
        winning_day = ProtocolDay(
            date=date.today(),
            starting_equity=100000,
            ending_equity=101000,
            daily_pnl=1000,
            daily_return=0.01,
            trades_opened=0,
            trades_closed=0,
            max_drawdown_today=0,
            regime="trend_low_vol",
            circuit_breaker_triggered=False,
        )
        assert winning_day.is_profitable is True

        losing_day = ProtocolDay(
            date=date.today(),
            starting_equity=100000,
            ending_equity=99000,
            daily_pnl=-1000,
            daily_return=-0.01,
            trades_opened=0,
            trades_closed=0,
            max_drawdown_today=0.01,
            regime="trend_low_vol",
            circuit_breaker_triggered=False,
        )
        assert losing_day.is_profitable is False
