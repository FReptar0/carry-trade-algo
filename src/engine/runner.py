"""Main trading loop for live/paper trading.

Ties together the broker, strategy, economic calendar, circuit breaker,
performance monitor, trading protocol, state store, and autonomous
system components (alert manager, watchdog, reconciler, live calendar,
signal filter, dynamic sizer, correlation monitor, scale manager,
adaptive parameters, and model lifecycle). Uses APScheduler for
periodic execution.

Tick workflow (every hour):
1. Reconcile positions with broker
2. Check if market is open (skip if closed)
3. Check protocol status (RUNNING or DEGRADED allowed; DEGRADED
   blocks new entries and scale-in but manages existing positions)
4. Fetch latest candles from OANDA for each pair
5. Update correlation monitor
6. Check economic calendar blackout (skip new entries if active)
6b. Check ExitManager for existing positions (time, regime, support,
    trend reversal, profit-scaled trailing — closes if triggered)
7. Run V3 strategy on candle data
8. Run signal filter (bandit gate)
9. Execute signal via OANDA with dynamic sizing
10. Check scale-in/scale-out for existing positions
11. Update performance monitor
12. Check circuit breakers
12b. Check degradation conditions (RUNNING → DEGRADED)
12c. Check recovery conditions (DEGRADED → RUNNING)
13. Check protocol abort conditions (from RUNNING or DEGRADED)
13b. Update broker-side stop-losses (trailing / ratchet)
14. Save state to SQLite
15. Watchdog heartbeat
16. Log tick summary
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.broker.oanda import OandaBroker
from src.broker.orders import OrderSide
from src.engine.market_hours import ForexMarketHours
from src.data.cross_asset import CrossAssetFetcher
from src.ml.bandit import SignalBandit
from src.ml.features import FEATURE_NAMES, FeatureExtractor
from src.ml.shadow import ShadowEvaluator
from src.monitoring.performance import PerformanceMonitor, TradeRecord
from src.news.calendar import EconomicCalendar
from src.ops.alerts import AlertConfig, AlertManager
from src.ops.reconciler import Reconciler
from src.ops.watchdog import Watchdog
from src.persistence.store import StateStore
from src.risk.circuit_breakers import CircuitBreaker, PortfolioState, RiskLimits
from src.risk.correlation import CorrelationMonitor
from src.risk.dynamic_sizer import DynamicSizer
from src.risk.scaling import ScaleManager
from src.regime.detector import RegimeDetector, RegimeState
from src.strategy.base import Signal
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3
from src.strategy.exit_manager import ExitManager, ExitManagerConfig, PositionState
from src.utils.logger import TradingLogger
from src.validation.protocol import (
    ProtocolDay,
    ProtocolStatus,
    TradingProtocol,
)

UTC = ZoneInfo("UTC")

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the trading runner.

    Attributes:
        pairs: Currency pairs to trade (slash format).
        check_interval_minutes: Minutes between strategy ticks.
        candle_lookback: Number of H1 candles to fetch.
        position_size_units: Units per trade (OANDA units, not lots).
        swap_long_default: Default swap rate for long positions.
        swap_short_default: Default swap rate for short positions.
        initial_equity: Starting equity for protocol.
        log_dir: Directory for log files.
        db_path: Path to SQLite database.
        events_file: Path to economic events JSON.
        blackout_hours: Hours around events to avoid trading.
        telegram_bot_token: Telegram bot token for alerts.
        telegram_chat_id: Telegram chat ID for alerts.
        enable_signal_filter: Enable ML signal filtering.
        enable_dynamic_sizing: Enable dynamic position sizing.
        enable_scaling: Enable scale-in/scale-out.
    """

    pairs: list[str] = field(default_factory=lambda: ["USD/JPY", "AUD/JPY"])
    check_interval_minutes: int = 60
    candle_lookback: int = 300
    position_size_units: int = 10000
    swap_long_default: float = 0.00005
    swap_short_default: float = -0.00005
    initial_equity: float = 100000.0
    log_dir: str = "logs"
    db_path: str = "data/trading.db"
    events_file: str = "src/news/events_2026.json"
    blackout_hours: int = 2
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    enable_signal_filter: bool = True
    enable_dynamic_sizing: bool = True
    enable_scaling: bool = True


class TradingRunner:
    """Main trading loop that orchestrates all system components.

    Args:
        broker: OANDA broker adapter.
        config: Runner configuration.

    Example:
        >>> broker = OandaBroker(oanda_config)
        >>> runner = TradingRunner(broker, RunnerConfig())
        >>> runner.start()  # Blocks until stopped
    """

    def __init__(
        self,
        broker: OandaBroker,
        config: RunnerConfig | None = None,
    ) -> None:
        self.broker = broker
        self.config = config or RunnerConfig()
        self.running = False

        # Core components
        self.strategy = CarryTradeStrategyV3()
        self.store = StateStore(self.config.db_path)
        self.circuit_breaker = CircuitBreaker(RiskLimits())
        self.tlogger = TradingLogger(self.config.log_dir)
        self.scheduler = BlockingScheduler(timezone=UTC)

        # Phase A: Operations components
        self.alert_manager = self._init_alert_manager()
        self.watchdog = Watchdog(
            expected_interval_sec=self.config.check_interval_minutes * 60,
            alert_manager=self.alert_manager,
            market_open_check=lambda: ForexMarketHours.is_market_open(
                datetime.now(UTC)
            ),
        )
        self.reconciler = Reconciler(alert_manager=self.alert_manager)
        self._last_market_state: bool = True  # Track market open/close
        self.calendar = self._init_calendar()

        # Phase B: Adaptive parameters (loaded lazily)
        self._param_store = None
        self._adaptive_adapter = None

        # Phase C: Signal filter
        self._cross_asset_fetcher = CrossAssetFetcher(cache_ttl=300)
        self.feature_extractor = FeatureExtractor(
            cross_asset_fetcher=self._cross_asset_fetcher
        )
        self.bandit = SignalBandit(n_features=len(FEATURE_NAMES))
        self.shadow_evaluator = ShadowEvaluator(min_trades=50, min_advantage=0.1)
        self._bandit_active = False
        self._load_bandit_model()

        # Phase D: Advanced risk management
        self.dynamic_sizer = DynamicSizer(base_units=self.config.position_size_units)
        self.correlation_monitor = CorrelationMonitor(lookback=60)
        self.scale_manager = ScaleManager()

        # Cached candle data for correlation updates
        self._candle_cache: dict[str, pd.DataFrame] = {}

        # Phase 3 hardening: ExitManager per pair + regime detector
        self.regime_detector = RegimeDetector()
        self._exit_managers: dict[str, ExitManager] = {}
        self._exit_config = ExitManagerConfig(
            base_atr_mult=self.strategy.config.atr_stop_mult,
            time_exit_enabled=True,
            max_hold_days=30,
            min_hold_hours=24,
            profit_scale_enabled=True,
            regime_exit_enabled=True,
            use_sr_exits=True,
        )

        # Restore or create protocol
        self.protocol = self._restore_or_create_protocol()

        # Performance monitor
        self.monitor = PerformanceMonitor(initial_equity=self.config.initial_equity)

        # Daily tracking
        self._today: Optional[date] = None
        self._day_start_equity: float = self.config.initial_equity
        self._trades_opened_today: int = 0
        self._trades_closed_today: int = 0
        self._max_drawdown_today: float = 0.0
        self._cb_triggered_today: bool = False

        # Track open positions per pair (from strategy signals)
        self._strategy_positions: dict[str, dict] = {}

    def _init_alert_manager(self) -> Optional[AlertManager]:
        """Initialize AlertManager if Telegram credentials exist."""
        token = self.config.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = self.config.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        if token and chat_id:
            return AlertManager(
                AlertConfig(
                    telegram_bot_token=token,
                    telegram_chat_id=chat_id,
                    min_severity="WARNING",
                )
            )
        logger.info("AlertManager disabled (no Telegram credentials)")
        return None

    def _init_calendar(self) -> EconomicCalendar:
        """Initialize calendar, preferring live feed over static."""
        try:
            from src.news.live_calendar import LiveCalendar

            cal = LiveCalendar(
                fallback_file=self.config.events_file,
                blackout_hours=self.config.blackout_hours,
            )
            cal.refresh()
            logger.info("Using LiveCalendar (Forex Factory feed)")
            return cal
        except Exception as e:
            logger.warning(
                "Failed to init LiveCalendar: %s, falling back to static",
                e,
            )

        return EconomicCalendar(
            self.config.events_file,
            blackout_hours=self.config.blackout_hours,
        )

    def _load_bandit_model(self) -> None:
        """Load bandit model from registry if available."""
        try:
            from src.rl.registry import ModelRegistry

            registry = ModelRegistry(self.config.db_path)
            active = registry.get_active("bandit_v1")
            if active:
                self.bandit.load(active.artifact_path)
                # Guard: reinitialize if loaded model has wrong n_features
                if self.bandit.n_features != len(FEATURE_NAMES):
                    logger.warning(
                        "Bandit n_features mismatch (%d vs %d), reinitializing",
                        self.bandit.n_features,
                        len(FEATURE_NAMES),
                    )
                    self.bandit = SignalBandit(n_features=len(FEATURE_NAMES))
                    return
                self._bandit_active = True
                logger.info(
                    "Loaded active bandit model v%d",
                    active.version,
                )
        except Exception as e:
            logger.debug("No bandit model loaded: %s", e)

    def _restore_or_create_protocol(self) -> TradingProtocol:
        """Load protocol from database or create a new one.

        If the protocol was previously aborted, reset it to DEGRADED
        (not RUNNING) so positions are managed defensively. Recovery
        to RUNNING happens automatically when conditions clear.
        """
        protocol = self.store.load_protocol_state()
        if protocol is not None:
            if protocol.status == ProtocolStatus.ABORTED:
                logger.warning(
                    "Protocol was aborted (%s) — resuming as "
                    "DEGRADED (day %d/%d, history preserved)",
                    protocol.abort_reason,
                    len(protocol.days),
                    protocol.duration,
                )
                protocol.status = ProtocolStatus.DEGRADED
                protocol.degraded_reason = (
                    f"Resumed after abort: {protocol.abort_reason}"
                )
                protocol.degraded_since = datetime.now()
                protocol.abort_reason = None
                protocol.end_date = None
                self.store.save_protocol_state(protocol)
            else:
                logger.info(
                    "Restored protocol: day %d/%d, status=%s",
                    len(protocol.days),
                    protocol.duration,
                    protocol.status.value,
                )
            return protocol

        protocol = TradingProtocol(
            duration_days=30,
            initial_equity=self.config.initial_equity,
        )
        protocol.start()
        self.store.save_protocol_state(protocol)
        logger.info("Created new 30-day protocol")
        return protocol

    def start(self) -> None:
        """Start the trading loop. Blocks until stopped."""
        self.running = True
        self.tlogger.log_startup(
            {
                "pairs": self.config.pairs,
                "interval_min": self.config.check_interval_minutes,
                "position_units": self.config.position_size_units,
                "protocol_day": len(self.protocol.days),
                "signal_filter": self.config.enable_signal_filter,
                "dynamic_sizing": self.config.enable_dynamic_sizing,
                "scaling": self.config.enable_scaling,
            }
        )
        logger.info("TradingRunner starting")

        # Send startup alert
        if self.alert_manager:
            self.alert_manager.send(
                "WARNING",
                "System Startup",
                f"Trading runner started with pairs: {self.config.pairs}",
            )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize day tracking
        now = datetime.now(UTC)
        self._check_new_day(now)

        # Sync positions from OANDA on startup
        self._sync_positions()

        # Schedule jobs
        if self.config.check_interval_minutes >= 60:
            tick_trigger = CronTrigger(minute=0, timezone=UTC)
        else:
            tick_trigger = CronTrigger(
                minute=f"*/{self.config.check_interval_minutes}",
                timezone=UTC,
            )
        self.scheduler.add_job(
            self._tick,
            tick_trigger,
            id="strategy_tick",
            name="Strategy tick",
            max_instances=1,
            misfire_grace_time=300,
        )
        self.scheduler.add_job(
            self._health_check,
            CronTrigger(minute="*/5", timezone=UTC),
            id="health_check",
            name="Health check",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._daily_rollover,
            CronTrigger(hour=21, minute=55, timezone=UTC),
            id="daily_rollover",
            name="Daily rollover",
            max_instances=1,
        )

        # Phase B: Weekly optimization (Sunday 00:00 UTC)
        self.scheduler.add_job(
            self._weekly_optimization,
            CronTrigger(day_of_week="sun", hour=0, minute=0, timezone=UTC),
            id="weekly_optimization",
            name="Weekly param optimization",
            max_instances=1,
        )

        # Phase E: Weekly retraining (Sunday 02:00 UTC)
        self.scheduler.add_job(
            self._weekly_retraining,
            CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=UTC),
            id="weekly_retraining",
            name="Weekly model retraining",
            max_instances=1,
        )

        # Daily report (22:00 UTC, after rollover)
        self.scheduler.add_job(
            self._send_daily_report,
            CronTrigger(hour=22, minute=0, timezone=UTC),
            id="daily_report",
            name="Daily performance report",
            max_instances=1,
        )

        # Weekly report (Sunday 08:00 UTC)
        self.scheduler.add_job(
            self._send_weekly_report,
            CronTrigger(day_of_week="sun", hour=8, minute=0, timezone=UTC),
            id="weekly_report",
            name="Weekly performance report",
            max_instances=1,
        )

        # Run initial tick
        self._tick()

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop("User interrupt")

    def stop(self, reason: str = "manual") -> None:
        """Gracefully shut down the runner.

        Args:
            reason: Reason for stopping.
        """
        logger.info("Stopping runner: %s", reason)
        self.running = False

        # Save final state
        self._record_day_result()
        self.store.save_protocol_state(self.protocol)
        self.tlogger.log_shutdown(reason)

        # Send shutdown alert
        if self.alert_manager:
            self.alert_manager.send(
                "WARNING",
                "System Shutdown",
                f"Trading runner stopped: {reason}",
            )

        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle OS signals for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.info("Received %s", sig_name)
        self.stop(f"Signal {sig_name}")
        sys.exit(0)

    def _tick(self) -> None:
        """Core hourly logic — the main trading heartbeat."""
        now = datetime.now(UTC)
        self._check_new_day(now)

        # 1. Check market hours
        if not ForexMarketHours.is_market_open(now):
            logger.debug("Market closed, skipping tick")
            return

        # 2. Check protocol status
        if self.protocol.status not in (
            ProtocolStatus.RUNNING,
            ProtocolStatus.DEGRADED,
        ):
            logger.info(
                "Protocol not active (status=%s), skipping",
                self.protocol.status.value,
            )
            # Still record heartbeat so the watchdog knows the
            # system is alive even when not actively trading.
            self.watchdog.heartbeat()
            return

        # 3. Check economic calendar
        in_blackout, event = self.calendar.is_blackout_period(now)
        if in_blackout and event:
            logger.info(
                "Blackout period: %s (%s) — skipping new entries",
                event.name,
                event.currency,
            )

        # 4. Get account state
        acct = self.broker.get_account_state()
        if acct is None:
            logger.warning("Could not fetch account state")
            return

        equity = acct["equity"]
        balance = acct["balance"]
        unrealized = acct["unrealized_pnl"]

        # 5. Reconcile positions
        broker_positions = self.broker.get_all_positions()
        if broker_positions is not None:
            result = self.reconciler.reconcile(
                self._strategy_positions,
                broker_positions,
                managed_pairs=self.config.pairs,
            )
            if not result.is_clean:
                self.reconciler.apply_fixes(
                    self._strategy_positions,
                    result,
                    broker_positions,
                )

        # 6. Process each pair
        for pair in self.config.pairs:
            try:
                self._process_pair(pair, now, in_blackout, equity)
            except Exception as e:
                logger.error("Error processing %s: %s", pair, e)
                self.tlogger.log_error(e, context=f"tick/{pair}")

        # 7. Update correlation monitor
        if self._candle_cache:
            self.correlation_monitor.update(self._candle_cache)

        # 8. Update performance monitor
        daily_pnl = equity - self._day_start_equity
        positions = broker_positions
        pos_count = len(positions) if positions else 0

        self.monitor.update(
            equity=equity,
            balance=balance,
            unrealized_pnl=unrealized,
            daily_pnl=daily_pnl,
            positions_count=pos_count,
            timestamp=now,
        )

        # 9. Check circuit breakers
        portfolio_state = PortfolioState(
            equity=equity,
            high_water_mark=self.monitor.high_water_mark,
            daily_pnl=daily_pnl,
            weekly_pnl=daily_pnl,  # Simplified for now
            open_positions=pos_count,
        )
        violations = self.circuit_breaker.check_limits(portfolio_state, now)
        if violations:
            self._cb_triggered_today = True
            for v in violations:
                logger.warning("Circuit breaker: %s", v.message)
                if self.alert_manager:
                    self.alert_manager.send(
                        "HIGH",
                        "Circuit Breaker",
                        v.message,
                    )

        # 10. Track drawdown
        snapshot = self.monitor.get_snapshot()
        self._max_drawdown_today = max(self._max_drawdown_today, snapshot.drawdown)

        # 10b. Check degraded conditions (RUNNING → DEGRADED)
        if self.protocol.status == ProtocolStatus.RUNNING and self.protocol.days:
            should_degrade, degrade_reason = self.protocol.check_degraded_conditions()
            if should_degrade and degrade_reason:
                logger.warning("Protocol degrading: %s", degrade_reason)
                self.protocol.degrade(degrade_reason)
                self.store.save_protocol_state(self.protocol)
                if self.alert_manager:
                    self.alert_manager.send(
                        "HIGH",
                        "Protocol DEGRADED",
                        f"Entering defensive mode: {degrade_reason}\n"
                        "No new entries. Existing positions managed.",
                    )

        # 10c. Check recovery conditions (DEGRADED → RUNNING)
        if self.protocol.status == ProtocolStatus.DEGRADED and self.protocol.days:
            can_recover, recover_reason = self.protocol.check_recovery_conditions()
            if can_recover and recover_reason:
                logger.info("Protocol recovering: %s", recover_reason)
                self.protocol.recover()
                self.store.save_protocol_state(self.protocol)
                if self.alert_manager:
                    self.alert_manager.send(
                        "INFO",
                        "Protocol RECOVERED",
                        f"Resuming full trading: {recover_reason}",
                    )

        # 11. Check protocol abort (fires from both RUNNING and DEGRADED)
        if self.protocol.days:
            should_abort, reason = self.protocol.check_abort_conditions()
            if should_abort and reason:
                logger.critical("Protocol abort: %s", reason)
                if self.alert_manager:
                    self.alert_manager.send(
                        "CRITICAL",
                        "Protocol Abort",
                        reason,
                    )
                self._close_all_positions(reason)
                self.protocol.abort(reason)
                self.store.save_protocol_state(self.protocol)
                self.stop(f"Protocol abort: {reason}")
                return

        # 11b. Update broker-side stop-losses (trailing / ratchet)
        try:
            self._update_stop_losses()
        except Exception as e:
            logger.error("Stop-loss update sweep failed: %s", e)

        # 12. Save equity snapshot
        self.store.save_equity_snapshot(
            timestamp=now,
            equity=equity,
            balance=balance,
            unrealized_pnl=unrealized,
            positions_count=pos_count,
            drawdown=snapshot.drawdown,
        )

        # 13. Watchdog heartbeat
        self.watchdog.heartbeat()

        # 14. Log tick summary
        self.tlogger.log_pnl(
            equity=equity,
            balance=balance,
            daily_pnl=daily_pnl,
            unrealized_pnl=unrealized,
        )
        logger.info(
            "Tick complete: equity=%.2f pnl=%.2f positions=%d",
            equity,
            daily_pnl,
            pos_count,
        )

    def _process_pair(
        self,
        pair: str,
        now: datetime,
        in_blackout: bool,
        equity: float = 0.0,
    ) -> None:
        """Run strategy and execute signals for one pair.

        Args:
            pair: Currency pair.
            now: Current UTC time.
            in_blackout: Whether we're in an economic event blackout.
            equity: Current account equity.
        """
        # Fetch candles
        df = self.broker.fetch_candles(
            pair,
            count=self.config.candle_lookback,
            granularity="H1",
        )
        if df is None or len(df) < 220:
            logger.warning(
                "Insufficient data for %s (%s bars)",
                pair,
                len(df) if df is not None else 0,
            )
            return

        # Cache candles for correlation monitor
        self._candle_cache[pair] = df

        # Add swap columns — try real rates first
        swap_rates = self.broker.get_swap_rates(pair)
        if swap_rates and (swap_rates[0] != 0 or swap_rates[1] != 0):
            df["swap_long"] = swap_rates[0]
            df["swap_short"] = swap_rates[1]
        else:
            df["swap_long"] = self.config.swap_long_default
            df["swap_short"] = self.config.swap_short_default

        # Run strategy
        signals = self.strategy.generate_signals(df)
        if not signals:
            return

        # Read only the LAST signal (current state)
        last_signal = signals[-1]

        # Determine if this pair has an open position
        has_position = pair in self._strategy_positions

        # Phase 3 hardening: Run ExitManager on existing positions
        # (time-based, regime-change, support-break, profit-scaled trailing)
        # This runs BEFORE V3 signal check — if ExitManager closes, we're done.
        if has_position:
            try:
                was_closed = self._check_exit_manager(pair, df, now)
                if was_closed:
                    return
            except Exception as e:
                logger.error("ExitManager error for %s: %s", pair, e)

        # Phase D: Check scale-in/scale-out for existing positions
        # Note: scale-out (profit-taking) is allowed in DEGRADED; scale-in is blocked
        if has_position and self.config.enable_scaling:
            if self.protocol.status == ProtocolStatus.DEGRADED:
                # Only allow scale-out (reduction), skip scale-in
                self._check_scaling(pair, df, scale_in_allowed=False)
            else:
                self._check_scaling(pair, df)

        if last_signal.signal == Signal.LONG and not has_position:
            # New entry signal
            if self.protocol.status == ProtocolStatus.DEGRADED:
                logger.info(
                    "Skipping entry for %s (DEGRADED mode — no new entries)",
                    pair,
                )
                return

            if in_blackout:
                logger.info("Skipping entry for %s (blackout)", pair)
                return

            if self.circuit_breaker.is_trading_halted:
                logger.info("Skipping entry for %s (circuit breaker)", pair)
                return

            # Phase C: Signal filter
            if self.config.enable_signal_filter:
                features = self.feature_extractor.extract(df, pair)
                feat_array = self.feature_extractor.to_array(features)
                action, confidence = self.bandit.decide(feat_array)

                if self._bandit_active:
                    if action == "SKIP":
                        logger.info(
                            "Signal filtered for %s (bandit: SKIP, conf=%.3f)",
                            pair,
                            confidence,
                        )
                        return
                # Shadow mode: log but don't block

            self._open_position(pair, last_signal.reason, now, df, equity)

        elif last_signal.signal == Signal.CLOSE and has_position:
            # Exit signal
            self._close_position(pair, last_signal.reason, now)

    def _open_position(
        self,
        pair: str,
        reason: str,
        now: datetime,
        df: Optional[pd.DataFrame] = None,
        equity: float = 0.0,
    ) -> None:
        """Open a new position via OANDA with a broker-side stop-loss.

        Args:
            pair: Currency pair.
            reason: Entry reason from strategy.
            now: Current time.
            df: Candle data for dynamic sizing and stop calculation.
            equity: Current account equity.
        """
        self.tlogger.log_signal(pair, "LONG", reason)

        # Calculate ATR for stop-loss (always needed, even without
        # dynamic sizing). Falls back to a conservative default.
        from src.strategy.indicators import atr as calc_atr

        current_atr = 0.5  # Conservative fallback
        current_close = 0.0
        if df is not None and len(df) > 20:
            atr_vals = calc_atr(df["high"], df["low"], df["close"])
            last_atr = atr_vals.iloc[-1]
            if not np.isnan(last_atr):
                current_atr = float(last_atr)
            current_close = float(df["close"].iloc[-1])

        # Phase D: Dynamic position sizing
        if self.config.enable_dynamic_sizing and df is not None and equity > 0:
            price = current_close

            # Get performance stats for Kelly
            stats = self.monitor.get_snapshot()
            win_rate = getattr(stats, "win_rate", 0.5)
            avg_win = getattr(stats, "avg_win", 200.0)
            avg_loss = getattr(stats, "avg_loss", 100.0)

            # Correlation factor
            open_pairs = list(self._strategy_positions.keys()) + [pair]
            corr_factor = self.correlation_monitor.portfolio_correlation_factor(
                open_pairs
            )

            units = self.dynamic_sizer.calculate_size(
                pair=pair,
                equity=equity,
                atr=current_atr,
                price=price,
                win_rate=win_rate if win_rate > 0 else 0.5,
                avg_win=avg_win if avg_win > 0 else 200.0,
                avg_loss=avg_loss if avg_loss > 0 else 100.0,
                regime_mult=1.0,
                correlation_factor=corr_factor,
                drawdown_pct=stats.drawdown,
            )
        else:
            units = self.config.position_size_units

        # Calculate broker-side stop-loss price (3x ATR below entry)
        # Uses current close as proxy — the few pips of slippage are
        # negligible vs 3x ATR stop width (~3-4% on JPY pairs).
        atr_stop_mult = self.strategy.config.atr_stop_mult
        stop_loss_price = None
        if current_close > 0 and current_atr > 0:
            stop_loss_price = round(current_close - (current_atr * atr_stop_mult), 3)

        order = self.broker.submit_market_order(
            pair=pair,
            side=OrderSide.BUY,
            units=units,
            stop_loss_price=stop_loss_price,
        )

        if order is None or order.avg_fill_price is None:
            logger.error("Failed to open position for %s", pair)
            return

        self._strategy_positions[pair] = {
            "entry_price": order.avg_fill_price,
            "entry_time": now,
            "units": units,
            "tranche_count": 1,
            "levels_taken": 0,
            "trade_id": order.trade_id,
            "stop_price": stop_loss_price,
            "high_water_mark": order.avg_fill_price,
            "low_water_mark": order.avg_fill_price,
        }
        self._trades_opened_today += 1

        # Initialize ExitManager for this pair
        em = self._get_or_create_exit_manager(pair)
        regime = self._detect_regime_safe(df)
        em.initialize_stop(order.avg_fill_price, current_atr, regime)

        self.store.save_trade(
            {
                "timestamp": now.isoformat(),
                "pair": pair,
                "side": "BUY",
                "entry_price": order.avg_fill_price,
                "quantity": float(units),
                "status": "open",
                "metadata": {
                    "reason": reason,
                    "trade_id": order.trade_id,
                    "stop_price": stop_loss_price,
                },
            }
        )

        # Alert on trade
        if self.alert_manager:
            self.alert_manager.send(
                "INFO",
                "Trade Opened",
                f"{pair} LONG {units} units @ "
                f"{order.avg_fill_price:.5f} "
                f"(stop={stop_loss_price:.3f}) — {reason}",
            )

        self.tlogger.log_order(order.to_dict())
        logger.info(
            "Opened %s %d units @ %.5f (stop=%.3f, trade_id=%s) — %s",
            pair,
            units,
            order.avg_fill_price,
            stop_loss_price or 0.0,
            order.trade_id or "N/A",
            reason,
        )

    def _close_position(self, pair: str, reason: str, now: datetime) -> None:
        """Close an existing position via OANDA.

        Args:
            pair: Currency pair.
            reason: Exit reason from strategy.
            now: Current time.
        """
        self.tlogger.log_signal(pair, "CLOSE", reason)

        entry_info = self._strategy_positions.get(pair)
        order = self.broker.close_position(pair)

        if order is None or order.avg_fill_price is None:
            logger.error("Failed to close position for %s", pair)
            # Defensive cleanup: if the position no longer exists on
            # the broker (e.g., already closed), remove the ghost
            # entry from internal tracking to prevent repeated failures.
            broker_positions = self.broker.get_all_positions()
            if broker_positions is not None:
                broker_pairs = {p["pair"] for p in broker_positions}
                if pair not in broker_pairs:
                    logger.warning(
                        "Ghost position detected: %s not on broker, "
                        "removing from internal tracking",
                        pair,
                    )
                    self._strategy_positions.pop(pair, None)
            return

        exit_price = order.avg_fill_price
        pnl = 0.0
        duration = timedelta(0)

        if entry_info:
            entry_price = entry_info["entry_price"]
            units = entry_info["units"]
            pnl = (exit_price - entry_price) * units
            duration = now - entry_info["entry_time"]

        del self._strategy_positions[pair]
        self._exit_managers.pop(pair, None)  # Clean up ExitManager state
        self._trades_closed_today += 1

        # Phase C: Update bandit with trade outcome
        if self.config.enable_signal_filter and entry_info:
            try:
                cached_df = self._candle_cache.get(pair)
                if cached_df is not None:
                    features = self.feature_extractor.extract(cached_df, pair)
                    feat_array = self.feature_extractor.to_array(features)
                    # Normalize PnL for reward
                    reward = np.clip(pnl / 1000.0, -1.0, 1.0)
                    self.bandit.update(feat_array, "TAKE", reward)

                    # Shadow evaluation
                    if not self._bandit_active:
                        action, _ = self.bandit.decide(feat_array)
                        self.shadow_evaluator.record(feat_array, action, pnl)
                        should_activate, stats = self.shadow_evaluator.should_activate()
                        if should_activate:
                            self._bandit_active = True
                            logger.info("Bandit activated: %s", stats)
            except Exception as e:
                logger.debug("Bandit update failed: %s", e)

        # Record trade
        self.monitor.record_trade(
            TradeRecord(
                timestamp=now,
                pair=pair,
                side="BUY",
                entry_price=entry_info["entry_price"] if entry_info else 0,
                exit_price=exit_price,
                quantity=float(entry_info["units"] if entry_info else 0),
                pnl=pnl,
                pnl_pct=(
                    pnl / (entry_info["entry_price"] * entry_info["units"])
                    if entry_info
                    else 0
                ),
                duration=duration,
                swap_earned=0.0,
            )
        )

        self.store.save_trade(
            {
                "timestamp": now.isoformat(),
                "pair": pair,
                "side": "SELL",
                "entry_price": entry_info["entry_price"] if entry_info else 0,
                "exit_price": exit_price,
                "quantity": float(entry_info["units"] if entry_info else 0),
                "pnl": pnl,
                "status": "closed",
                "metadata": {"reason": reason},
            }
        )

        # Alert on trade close
        if self.alert_manager:
            self.alert_manager.send(
                "INFO",
                "Trade Closed",
                f"{pair} closed @ {exit_price:.5f}, PnL={pnl:.2f} — {reason}",
            )

        self.tlogger.log_position_close(pair, pnl, reason)
        logger.info(
            "Closed %s @ %.5f, PnL=%.2f — %s",
            pair,
            exit_price,
            pnl,
            reason,
        )

    def _get_or_create_exit_manager(self, pair: str) -> ExitManager:
        """Get the ExitManager for a pair, creating one if needed.

        Args:
            pair: Currency pair.

        Returns:
            ExitManager instance for this pair.
        """
        if pair not in self._exit_managers:
            self._exit_managers[pair] = ExitManager(self._exit_config)
        return self._exit_managers[pair]

    def _detect_regime_safe(self, df: Optional[pd.DataFrame]) -> Optional[RegimeState]:
        """Detect regime from candle data, returning None on failure.

        Args:
            df: Candle DataFrame (needs ~75+ rows for regime detection).

        Returns:
            RegimeState or None if detection fails.
        """
        if df is None or len(df) < 75:
            return None
        try:
            return self.regime_detector.detect(df)
        except Exception as e:
            logger.debug("Regime detection failed: %s", e)
            return None

    def _check_exit_manager(
        self,
        pair: str,
        df: pd.DataFrame,
        now: datetime,
    ) -> bool:
        """Run ExitManager checks on a position and close if triggered.

        Evaluates time-based, regime-change, support-break, trend-reversal,
        and profit-scaled trailing exits. Returns True if position was closed.

        Args:
            pair: Currency pair.
            df: Candle data.
            now: Current UTC time.

        Returns:
            True if position was closed, False otherwise.
        """
        pos = self._strategy_positions.get(pair)
        if pos is None:
            return False

        current_price = float(df["close"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        entry_price = pos.get("entry_price", 0)
        entry_time = pos.get("entry_time", now)

        if entry_price <= 0:
            return False

        # Update low-water mark
        lwm = pos.get("low_water_mark", entry_price)
        if current_low < lwm:
            pos["low_water_mark"] = current_low
            lwm = current_low

        # Update high-water mark (also done in _update_stop_losses,
        # but keep in sync here for ExitManager's PositionState)
        hwm = pos.get("high_water_mark", entry_price)
        if current_price > hwm:
            pos["high_water_mark"] = current_price
            hwm = current_price

        # Build PositionState for ExitManager
        position_state = PositionState(
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=current_price,
            current_time=now,
            high_water_mark=hwm,
            low_water_mark=lwm,
            side="long",
        )

        # Get or create ExitManager for this pair
        em = self._get_or_create_exit_manager(pair)

        # Detect regime (optional — ExitManager handles None gracefully)
        regime = self._detect_regime_safe(df)

        # Run all exit checks
        exit_signal = em.check_exit(
            position=position_state,
            price_data=df,
            regime=regime,
        )

        if exit_signal.should_exit:
            exit_type_name = (
                exit_signal.exit_type.value if exit_signal.exit_type else "unknown"
            )
            exit_reason = (
                f"ExitManager [{exit_type_name}]: "
                f"{exit_signal.reason} (urgency={exit_signal.urgency:.2f})"
            )
            logger.info(
                "ExitManager triggered for %s: %s",
                pair,
                exit_reason,
            )

            # Alert before closing
            if self.alert_manager:
                self.alert_manager.send(
                    "WARNING" if exit_signal.is_urgent else "INFO",
                    f"Exit Signal: {pair}",
                    f"{exit_type_name.upper()}: "
                    f"{exit_signal.reason}\n"
                    f"Urgency: {exit_signal.urgency:.0%}\n"
                    f"Profit: {position_state.return_pct:.2%}\n"
                    f"Held: {position_state.hold_days:.1f} days",
                )

            self._close_position(pair, exit_reason, now)
            return True

        return False

    def _check_scaling(
        self,
        pair: str,
        df: pd.DataFrame,
        scale_in_allowed: bool = True,
    ) -> None:
        """Check for scale-in/scale-out opportunities.

        Args:
            pair: Currency pair with open position.
            df: Current candle data.
            scale_in_allowed: If False, skip scale-in (DEGRADED mode).
        """
        position = self._strategy_positions.get(pair)
        if position is None:
            return

        from src.strategy.indicators import atr as calc_atr

        atr_vals = calc_atr(df["high"], df["low"], df["close"])
        current_atr = float(atr_vals.iloc[-1]) if not np.isnan(atr_vals.iloc[-1]) else 0
        current_price = float(df["close"].iloc[-1])

        if current_atr <= 0:
            return

        # Check scale-in
        if scale_in_allowed and self.scale_manager.should_add(
            position, current_price, current_atr
        ):
            add_units = self.scale_manager.tranche_units(position["units"])

            # Calculate broker-side stop for the new tranche
            atr_stop_mult = self.strategy.config.atr_stop_mult
            scale_stop = round(current_price - (current_atr * atr_stop_mult), 3)

            order = self.broker.submit_market_order(
                pair=pair,
                side=OrderSide.BUY,
                units=add_units,
                stop_loss_price=scale_stop,
            )
            if order and order.avg_fill_price:
                # Update average entry price
                total_units = position["units"] + add_units
                position["entry_price"] = (
                    position["entry_price"] * position["units"]
                    + order.avg_fill_price * add_units
                ) / total_units
                position["units"] = total_units
                position["tranche_count"] = position.get("tranche_count", 1) + 1

                # Track all trade_ids for the position; the first
                # remains the "primary" used by _update_stop_losses.
                trade_ids = position.get("trade_ids", [])
                if position.get("trade_id") and position["trade_id"] not in trade_ids:
                    trade_ids.append(position["trade_id"])
                if order.trade_id:
                    trade_ids.append(order.trade_id)
                position["trade_ids"] = trade_ids

                logger.info(
                    "Scaled in %s: +%d units @ %.5f (stop=%.3f, trade_id=%s, total %d)",
                    pair,
                    add_units,
                    order.avg_fill_price,
                    scale_stop,
                    order.trade_id or "N/A",
                    total_units,
                )

        # Check scale-out (partial profit)
        elif self.scale_manager.should_reduce(position, current_price, current_atr):
            level_idx = position.get("levels_taken", 0)
            reduce_units = self.scale_manager.reduction_units(
                position["units"], level_idx
            )
            if reduce_units > 0 and reduce_units < position["units"]:
                # Use partial position close (not a SELL order) to
                # reduce the position through OANDA's position endpoint.
                order = self.broker.close_position(pair=pair, units=reduce_units)
                if order and order.avg_fill_price:
                    position["units"] -= reduce_units
                    position["levels_taken"] = level_idx + 1
                    partial_pnl = (
                        order.avg_fill_price - position["entry_price"]
                    ) * reduce_units
                    logger.info(
                        "Scaled out %s: -%d units (PnL=%.2f, remaining %d)",
                        pair,
                        reduce_units,
                        partial_pnl,
                        position["units"],
                    )

    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions (emergency/abort)."""
        now = datetime.now(UTC)
        pairs = list(self._strategy_positions.keys())
        for pair in pairs:
            try:
                self._close_position(pair, reason, now)
            except Exception as e:
                logger.error("Failed to close %s during abort: %s", pair, e)

    def _sync_positions(self) -> None:
        """Sync internal state with OANDA's actual positions and trades.

        On startup, fetches both position-level data (for sizing) and
        trade-level data (for trade_ids and stop-loss state). For any
        trade that lacks a broker-side stop-loss, calculates and attaches
        one using 3x ATR from current candle data.
        """
        positions = self.broker.get_all_positions()
        if positions is None:
            logger.warning("Could not sync positions from OANDA")
            return

        # Build position dict from aggregated positions
        for pos in positions:
            pair = pos["pair"]
            if pair in [p for p in self.config.pairs]:
                self._strategy_positions[pair] = {
                    "entry_price": pos["avg_price"],
                    "entry_time": datetime.now(UTC),
                    "units": pos["units"],
                    "tranche_count": 1,
                    "levels_taken": 0,
                    "trade_id": None,
                    "stop_price": None,
                    "high_water_mark": pos["avg_price"],
                    "low_water_mark": pos["avg_price"],
                }
                logger.info(
                    "Synced position: %s %d units @ %.5f",
                    pair,
                    pos["units"],
                    pos["avg_price"],
                )

        # Fetch trade-level data for trade_ids and stop-loss state
        open_trades = self.broker.get_open_trades()
        if open_trades is None:
            logger.warning("Could not fetch open trades for sync")
            return

        # Map trades to positions (use first trade per pair for simplicity;
        # multiple trades per pair can exist from scale-in)
        for trade in open_trades:
            pair = trade["pair"]
            pos = self._strategy_positions.get(pair)
            if pos is None:
                continue

            # Store the trade_id (first trade for this pair wins)
            if pos.get("trade_id") is None:
                pos["trade_id"] = trade["trade_id"]

            # Update HWM: use current price if above entry
            current_price = trade["price"] + (
                trade["unrealized_pnl"] / max(abs(trade["units"]), 1)
            )
            if current_price > pos.get("high_water_mark", 0):
                pos["high_water_mark"] = current_price

            # Check if this trade already has a broker-side stop
            if trade["stop_loss_price"] is not None:
                pos["stop_price"] = trade["stop_loss_price"]
                logger.info(
                    "Trade %s (%s) has stop @ %.3f",
                    trade["trade_id"],
                    pair,
                    trade["stop_loss_price"],
                )
            else:
                # No broker-side stop — attach one immediately
                self._attach_initial_stop(pair, trade["trade_id"], pos)

        # Initialize ExitManagers for all synced positions so that
        # time-based and regime-based checks work from the first tick.
        for pair, pos in self._strategy_positions.items():
            em = self._get_or_create_exit_manager(pair)
            entry_price = pos.get("entry_price", 0)
            stop_price = pos.get("stop_price")
            if entry_price > 0 and stop_price and stop_price > 0:
                # Approximate original ATR from the stop distance
                atr_est = (
                    entry_price - stop_price
                ) / self.strategy.config.atr_stop_mult
                em.initialize_stop(entry_price, max(atr_est, 0.01))
            logger.debug("ExitManager initialized for %s", pair)

    def _attach_initial_stop(self, pair: str, trade_id: str, pos: dict) -> None:
        """Calculate and attach a 3x ATR stop to an unprotected trade.

        Args:
            pair: Currency pair.
            trade_id: OANDA trade ID.
            pos: Internal position dict (mutated with stop_price).
        """
        from src.strategy.indicators import atr as calc_atr

        df = self.broker.fetch_candles(pair, count=50, granularity="H1")
        if df is None or len(df) < 20:
            logger.error(
                "Cannot calc stop for %s (insufficient candle data)",
                pair,
            )
            return

        atr_vals = calc_atr(df["high"], df["low"], df["close"])
        current_atr = float(atr_vals.iloc[-1])
        if np.isnan(current_atr) or current_atr <= 0:
            logger.error("Invalid ATR for %s, cannot set stop", pair)
            return

        atr_mult = self.strategy.config.atr_stop_mult
        current_price = float(df["close"].iloc[-1])
        stop_price = round(current_price - (current_atr * atr_mult), 3)

        success = self.broker.modify_trade_stop_loss(
            trade_id=trade_id,
            stop_price=stop_price,
        )
        if success:
            pos["stop_price"] = stop_price
            logger.info(
                "Attached stop to %s (trade %s): %.3f "
                "(price=%.3f, ATR=%.3f, mult=%.1f)",
                pair,
                trade_id,
                stop_price,
                current_price,
                current_atr,
                atr_mult,
            )
            if self.alert_manager:
                self.alert_manager.send(
                    "INFO",
                    "Stop Attached",
                    f"Broker-side stop set on {pair} (trade {trade_id})"
                    f" @ {stop_price:.3f}",
                )
        else:
            logger.error(
                "Failed to attach stop to %s (trade %s)",
                pair,
                trade_id,
            )

    def _update_stop_losses(self) -> None:
        """Update broker-side stop-losses for all open positions.

        For each position:
        - Updates the high-water mark from current candle data
        - If profit >= trailing_activation_pct: trail by trailing_atr_mult × ATR
        - Otherwise: keep initial 3x ATR stop from entry
        - Stops only ratchet upward (tighten) — never move down

        Called at the end of each tick. Failures are logged but do not
        propagate, so a broker timeout on one pair doesn't crash the tick.
        """
        from src.strategy.indicators import atr as calc_atr

        trailing_activation = self.strategy.config.trailing_activation_pct
        trailing_mult = self.strategy.config.trailing_atr_mult
        initial_mult = self.strategy.config.atr_stop_mult

        for pair, pos in self._strategy_positions.items():
            try:
                trade_id = pos.get("trade_id")
                if trade_id is None:
                    logger.debug("No trade_id for %s, skipping stop update", pair)
                    continue

                # Get current candle data from cache (populated earlier in tick)
                df = self._candle_cache.get(pair)
                if df is None or len(df) < 20:
                    logger.debug("No cached candles for %s, skipping stop update", pair)
                    continue

                current_price = float(df["close"].iloc[-1])
                entry_price = pos.get("entry_price", 0)
                current_stop = pos.get("stop_price")

                # Update high-water mark
                hwm = pos.get("high_water_mark", entry_price)
                if current_price > hwm:
                    pos["high_water_mark"] = current_price
                    hwm = current_price

                # Calculate current ATR
                atr_vals = calc_atr(df["high"], df["low"], df["close"])
                current_atr = float(atr_vals.iloc[-1])
                if np.isnan(current_atr) or current_atr <= 0:
                    continue

                # Determine which stop regime applies
                if entry_price > 0:
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = 0.0

                if profit_pct >= trailing_activation:
                    # Trailing mode: 2x ATR below high-water mark
                    new_stop = round(hwm - (current_atr * trailing_mult), 3)
                else:
                    # Initial mode: 3x ATR below current price
                    # (slides up with price, but never down)
                    new_stop = round(current_price - (current_atr * initial_mult), 3)

                # Ratchet: only update if new stop is ABOVE current stop
                if current_stop is not None and new_stop <= current_stop:
                    continue

                # Update broker
                success = self.broker.modify_trade_stop_loss(
                    trade_id=trade_id,
                    stop_price=new_stop,
                )
                if success:
                    old_stop = current_stop or 0.0
                    pos["stop_price"] = new_stop
                    mode = "TRAIL" if profit_pct >= trailing_activation else "INITIAL"
                    logger.info(
                        "Stop updated %s [%s]: %.3f → %.3f "
                        "(price=%.3f, HWM=%.3f, profit=%.2f%%)",
                        pair,
                        mode,
                        old_stop,
                        new_stop,
                        current_price,
                        hwm,
                        profit_pct * 100,
                    )
                else:
                    logger.warning(
                        "Failed to update stop for %s (trade %s)",
                        pair,
                        trade_id,
                    )

            except Exception as e:
                logger.error("Error updating stop for %s: %s", pair, e)

    def _check_new_day(self, now: datetime) -> None:
        """Handle day transitions."""
        today = now.date()
        if self._today == today:
            return

        if self._today is not None:
            # Record previous day
            self._record_day_result()

        self._today = today
        acct = self.broker.get_account_state()
        if acct:
            self._day_start_equity = acct["equity"]
        self._trades_opened_today = 0
        self._trades_closed_today = 0
        self._max_drawdown_today = 0.0
        self._cb_triggered_today = False

        logger.info("New trading day: %s", today)

    def _record_day_result(self) -> None:
        """Record the current day's results to protocol and store."""
        if self._today is None:
            return

        acct = self.broker.get_account_state()
        if acct is None:
            return

        ending_equity = acct["equity"]
        daily_pnl = ending_equity - self._day_start_equity
        daily_return = (
            daily_pnl / self._day_start_equity if self._day_start_equity > 0 else 0
        )

        day = ProtocolDay(
            date=self._today,
            starting_equity=self._day_start_equity,
            ending_equity=ending_equity,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            trades_opened=self._trades_opened_today,
            trades_closed=self._trades_closed_today,
            max_drawdown_today=self._max_drawdown_today,
            regime="LIVE",
            circuit_breaker_triggered=self._cb_triggered_today,
        )

        if self.protocol.status in (
            ProtocolStatus.RUNNING,
            ProtocolStatus.DEGRADED,
        ):
            self.protocol.record_day(day)

            # Check for checkpoint days
            day_num = len(self.protocol.days)
            if day_num in TradingProtocol.CHECKPOINT_DAYS:
                checkpoint = self.protocol.run_checkpoint(day_num)
                self.store.save_checkpoint(
                    day=checkpoint.day,
                    checkpoint_date=checkpoint.date,
                    cumulative_return=checkpoint.cumulative_return,
                    max_drawdown=checkpoint.max_drawdown,
                    current_drawdown=checkpoint.current_drawdown,
                    win_rate=checkpoint.win_rate,
                    sharpe_rolling=checkpoint.sharpe_rolling,
                    total_trades=checkpoint.total_trades,
                    issues=checkpoint.issues_identified,
                    recommendation=checkpoint.recommendation,
                )
                logger.info(
                    "Checkpoint day %d: %s (return=%.2f%%, dd=%.2f%%)",
                    day_num,
                    checkpoint.recommendation,
                    checkpoint.cumulative_return * 100,
                    checkpoint.max_drawdown * 100,
                )
                if self.alert_manager:
                    self.alert_manager.send(
                        "WARNING",
                        f"Protocol Checkpoint Day {day_num}",
                        f"Return: {checkpoint.cumulative_return:.2%}\n"
                        f"Drawdown: {checkpoint.max_drawdown:.2%}\n"
                        f"Recommendation: {checkpoint.recommendation}",
                    )

            # Check for protocol completion
            if day_num >= self.protocol.duration:
                self.protocol.complete()
                report = self.protocol.final_report()
                logger.info("Protocol completed: %s", report)

        # Send daily summary
        if self.alert_manager:
            self.alert_manager.send_daily_summary(
                {
                    "equity": ending_equity,
                    "daily_pnl": daily_pnl,
                    "drawdown": self._max_drawdown_today,
                    "positions_count": len(self._strategy_positions),
                }
            )

        self.store.save_daily_result(day)
        self.store.save_protocol_state(self.protocol)

    def _daily_rollover(self) -> None:
        """End-of-day processing (runs at ~21:55 UTC)."""
        logger.info("Daily rollover starting")
        self._record_day_result()
        logger.info("Daily rollover complete")

    def _health_check(self) -> None:
        """Periodic health check (every 5 minutes)."""
        now = datetime.now(UTC)

        # Watchdog check
        if not self.watchdog.check():
            logger.warning(
                "Watchdog: tick overdue by %.0fs",
                self.watchdog.seconds_since_heartbeat,
            )

        # Check OANDA connectivity
        acct = self.broker.get_account_state()
        if acct is None:
            logger.warning("Health check: OANDA unreachable")
            return

        snapshot = self.monitor.get_snapshot()
        progress = self.protocol.get_progress()

        logger.debug(
            "Health: equity=%.2f dd=%.2f%% day=%d/%d status=%s",
            snapshot.equity,
            snapshot.drawdown * 100,
            progress["days_completed"],
            self.protocol.duration,
            progress["status"],
        )

        # Log degraded state details
        if self.protocol.status == ProtocolStatus.DEGRADED:
            logger.info(
                "DEGRADED mode active: %s (since %s)",
                self.protocol.degraded_reason or "unknown",
                self.protocol.degraded_since.isoformat()
                if self.protocol.degraded_since
                else "unknown",
            )

        # Check for market open/close transitions
        self._check_market_transition()

    def _check_market_transition(self) -> None:
        """Check if market just opened or closed and send notification."""
        now = datetime.now(UTC)
        market_open = ForexMarketHours.is_market_open(now)

        if market_open and not self._last_market_state:
            # Market just opened
            next_close = ForexMarketHours.next_close(now)
            if self.alert_manager:
                positions = self.broker.get_all_positions() or []
                acct = self.broker.get_account_state() or {}
                pos_summary = (
                    "\n".join(
                        f"  {p['pair']}: {p['units']}u, PnL: ${p['unrealized_pnl']:.2f}"
                        for p in positions
                    )
                    if positions
                    else "  No open positions"
                )

                self.alert_manager.send(
                    "INFO",
                    "Market Opened",
                    f"Forex market is now OPEN\n\n"
                    f"Equity: ${acct.get('equity', 0):,.2f}\n"
                    f"Open Positions:\n{pos_summary}\n\n"
                    f"Next close: {next_close.strftime('%a %b %d %H:%M UTC')}",
                )
            logger.info("Market opened - trading resumed")

        elif not market_open and self._last_market_state:
            # Market just closed
            next_open = ForexMarketHours.next_open(now)
            if self.alert_manager:
                positions = self.broker.get_all_positions() or []
                acct = self.broker.get_account_state() or {}
                pos_summary = (
                    "\n".join(
                        f"  {p['pair']}: {p['units']}u, PnL: ${p['unrealized_pnl']:.2f}"
                        for p in positions
                    )
                    if positions
                    else "  No open positions"
                )

                self.alert_manager.send(
                    "INFO",
                    "Market Closed",
                    f"Forex market is now CLOSED\n\n"
                    f"Weekend Equity: ${acct.get('equity', 0):,.2f}\n"
                    f"Positions held over weekend:\n{pos_summary}\n\n"
                    f"Next open: {next_open.strftime('%a %b %d %H:%M UTC')}",
                )
            logger.info("Market closed - trading paused until %s", next_open)

        self._last_market_state = market_open

    def _weekly_optimization(self) -> None:
        """Run weekly parameter optimization (Sunday 00:00 UTC)."""
        logger.info("Weekly optimization starting")
        try:
            from src.adaptive.optimizer import ParamOptimizer
            from src.adaptive.param_store import ParamStore
            from src.regime.detector import CompositeRegime

            param_store = ParamStore(self.config.db_path)
            optimizer = ParamOptimizer(param_store)

            # Fetch recent candles for all pairs
            recent_candles = {}
            for pair in self.config.pairs:
                df = self.broker.fetch_candles(pair, count=720, granularity="H1")
                if df is not None and len(df) >= 220:
                    df["swap_long"] = self.config.swap_long_default
                    df["swap_short"] = self.config.swap_short_default
                    recent_candles[pair] = df

            if not recent_candles:
                logger.warning("No candle data for optimization, skipping")
                return

            # Optimize each regime
            for regime in CompositeRegime:
                try:
                    optimizer.optimize_regime(
                        regime.value,
                        recent_candles,
                        n_trials=50,
                    )
                except Exception as e:
                    logger.error(
                        "Optimization failed for %s: %s",
                        regime.value,
                        e,
                    )

            logger.info("Weekly optimization complete")

            if self.alert_manager:
                self.alert_manager.send(
                    "INFO",
                    "Weekly Optimization",
                    "Parameter optimization completed for all regimes",
                )

        except ImportError as e:
            logger.warning("Optimization skipped (missing dependency): %s", e)
        except Exception as e:
            logger.error("Weekly optimization failed: %s", e)

    def _weekly_retraining(self) -> None:
        """Run weekly model retraining (Sunday 02:00 UTC)."""
        logger.info("Weekly retraining starting")
        try:
            from src.rl.registry import ModelRegistry
            from src.rl.retraining import RetrainingPipeline

            registry = ModelRegistry(self.config.db_path)
            pipeline = RetrainingPipeline(registry)

            # Get recent trades
            trades = self.store.get_trades()
            recent = [t for t in trades if t.get("status") == "closed"][
                -100:
            ]  # Last 100 closed trades

            if len(recent) < 10:
                logger.info(
                    "Not enough recent trades for retraining (%d < 10)",
                    len(recent),
                )
                return

            # Retrain bandit
            pipeline.retrain_bandit(recent)

            # Check auto-promotion
            promoted = pipeline.auto_promote("bandit_v1")
            if promoted:
                self._load_bandit_model()
                if self.alert_manager:
                    self.alert_manager.send(
                        "WARNING",
                        "Model Promoted",
                        f"Bandit v{promoted} promoted to active",
                    )

            logger.info("Weekly retraining complete")

        except Exception as e:
            logger.error("Weekly retraining failed: %s", e)

    def _send_daily_report(self) -> None:
        """Send daily performance report via Telegram."""
        logger.info("Generating daily report")
        try:
            from scripts.generate_report import (
                get_db_connection,
                get_daily_report,
                get_protocol_progress,
                format_daily_report,
            )

            conn = get_db_connection(self.config.db_path)
            report = get_daily_report(conn)
            progress = get_protocol_progress(conn)
            conn.close()

            if not report:
                logger.warning("No daily report data available")
                return

            positions = self.broker.get_all_positions() or []
            account = self.broker.get_account_state() or {}

            report.open_positions = len(positions)
            report.unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)

            msg = format_daily_report(report, positions, account, progress)

            if self.alert_manager:
                self.alert_manager.send("INFO", "Daily Report", msg)
                logger.info("Daily report sent")
            else:
                logger.info("Daily report (no Telegram):\n%s", msg)

        except Exception as e:
            logger.error("Failed to send daily report: %s", e)

    def _send_weekly_report(self) -> None:
        """Send weekly performance report via Telegram."""
        logger.info("Generating weekly report")
        try:
            from scripts.generate_report import (
                get_db_connection,
                get_weekly_report,
                get_protocol_progress,
                format_weekly_report,
            )

            conn = get_db_connection(self.config.db_path)
            report = get_weekly_report(conn)
            progress = get_protocol_progress(conn)
            conn.close()

            if not report:
                logger.warning("No weekly report data available")
                return

            account = self.broker.get_account_state() or {}
            msg = format_weekly_report(report, progress, account)

            if self.alert_manager:
                self.alert_manager.send("INFO", "Weekly Report", msg)
                logger.info("Weekly report sent")
            else:
                logger.info("Weekly report (no Telegram):\n%s", msg)

        except Exception as e:
            logger.error("Failed to send weekly report: %s", e)
