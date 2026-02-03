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
3. Fetch latest candles from OANDA for each pair
4. Update correlation monitor
5. Check economic calendar blackout (skip new entries if active)
6. Run V3 strategy on candle data
7. Run signal filter (bandit gate)
8. Execute signal via OANDA with dynamic sizing
9. Check scale-in/scale-out for existing positions
10. Update performance monitor
11. Check circuit breakers
12. Check protocol abort conditions
13. Save state to SQLite
14. Watchdog heartbeat
15. Log tick summary
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
from src.strategy.base import Signal
from src.strategy.carry_trade_v3 import CarryTradeStrategyV3
from src.utils.logger import TradingLogger
from src.validation.protocol import ProtocolDay, TradingProtocol

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

    pairs: list[str] = field(
        default_factory=lambda: ["USD/JPY", "AUD/JPY"]
    )
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
        )
        self.reconciler = Reconciler(
            alert_manager=self.alert_manager
        )
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
        self.shadow_evaluator = ShadowEvaluator(
            min_trades=50, min_advantage=0.1
        )
        self._bandit_active = False
        self._load_bandit_model()

        # Phase D: Advanced risk management
        self.dynamic_sizer = DynamicSizer(
            base_units=self.config.position_size_units
        )
        self.correlation_monitor = CorrelationMonitor(lookback=60)
        self.scale_manager = ScaleManager()

        # Cached candle data for correlation updates
        self._candle_cache: dict[str, pd.DataFrame] = {}

        # Restore or create protocol
        self.protocol = self._restore_or_create_protocol()

        # Performance monitor
        self.monitor = PerformanceMonitor(
            initial_equity=self.config.initial_equity
        )

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
        token = self.config.telegram_bot_token or os.getenv(
            "TELEGRAM_BOT_TOKEN", ""
        )
        chat_id = self.config.telegram_chat_id or os.getenv(
            "TELEGRAM_CHAT_ID", ""
        )
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
                "Failed to init LiveCalendar: %s, "
                "falling back to static",
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
                        "Bandit n_features mismatch (%d vs %d), "
                        "reinitializing",
                        self.bandit.n_features,
                        len(FEATURE_NAMES),
                    )
                    self.bandit = SignalBandit(
                        n_features=len(FEATURE_NAMES)
                    )
                    return
                self._bandit_active = True
                logger.info(
                    "Loaded active bandit model v%d",
                    active.version,
                )
        except Exception as e:
            logger.debug("No bandit model loaded: %s", e)

    def _restore_or_create_protocol(self) -> TradingProtocol:
        """Load protocol from database or create a new one."""
        protocol = self.store.load_protocol_state()
        if protocol is not None:
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
                f"Trading runner started with pairs: "
                f"{self.config.pairs}",
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
            CronTrigger(
                day_of_week="sun", hour=0, minute=0, timezone=UTC
            ),
            id="weekly_optimization",
            name="Weekly param optimization",
            max_instances=1,
        )

        # Phase E: Weekly retraining (Sunday 02:00 UTC)
        self.scheduler.add_job(
            self._weekly_retraining,
            CronTrigger(
                day_of_week="sun", hour=2, minute=0, timezone=UTC
            ),
            id="weekly_retraining",
            name="Weekly model retraining",
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
        if self.protocol.status.value not in ("running",):
            logger.info(
                "Protocol not running (status=%s), skipping",
                self.protocol.status.value,
            )
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
        violations = self.circuit_breaker.check_limits(
            portfolio_state, now
        )
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
        self._max_drawdown_today = max(
            self._max_drawdown_today, snapshot.drawdown
        )

        # 11. Check protocol abort
        if self.protocol.days:
            should_abort, reason = (
                self.protocol.check_abort_conditions()
            )
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

        # Phase D: Check scale-in/scale-out for existing positions
        if has_position and self.config.enable_scaling:
            self._check_scaling(pair, df)

        if last_signal.signal == Signal.LONG and not has_position:
            # New entry signal
            if in_blackout:
                logger.info(
                    "Skipping entry for %s (blackout)", pair
                )
                return

            if self.circuit_breaker.is_trading_halted:
                logger.info(
                    "Skipping entry for %s (circuit breaker)", pair
                )
                return

            # Phase C: Signal filter
            if self.config.enable_signal_filter:
                features = self.feature_extractor.extract(df, pair)
                feat_array = self.feature_extractor.to_array(features)
                action, confidence = self.bandit.decide(feat_array)

                if self._bandit_active:
                    if action == "SKIP":
                        logger.info(
                            "Signal filtered for %s (bandit: SKIP, "
                            "conf=%.3f)",
                            pair,
                            confidence,
                        )
                        return
                # Shadow mode: log but don't block

            self._open_position(
                pair, last_signal.reason, now, df, equity
            )

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
        """Open a new position via OANDA.

        Args:
            pair: Currency pair.
            reason: Entry reason from strategy.
            now: Current time.
            df: Candle data for dynamic sizing.
            equity: Current account equity.
        """
        self.tlogger.log_signal(pair, "LONG", reason)

        # Phase D: Dynamic position sizing
        if self.config.enable_dynamic_sizing and df is not None and equity > 0:
            from src.strategy.indicators import atr as calc_atr

            atr_vals = calc_atr(df["high"], df["low"], df["close"])
            current_atr = float(atr_vals.iloc[-1]) if not np.isnan(atr_vals.iloc[-1]) else 0.5
            price = float(df["close"].iloc[-1])

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

        order = self.broker.submit_market_order(
            pair=pair,
            side=OrderSide.BUY,
            units=units,
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
        }
        self._trades_opened_today += 1

        self.store.save_trade(
            {
                "timestamp": now.isoformat(),
                "pair": pair,
                "side": "BUY",
                "entry_price": order.avg_fill_price,
                "quantity": float(units),
                "status": "open",
                "metadata": {"reason": reason},
            }
        )

        # Alert on trade
        if self.alert_manager:
            self.alert_manager.send(
                "INFO",
                "Trade Opened",
                f"{pair} LONG {units} units @ "
                f"{order.avg_fill_price:.5f} — {reason}",
            )

        self.tlogger.log_order(order.to_dict())
        logger.info(
            "Opened %s %d units @ %.5f — %s",
            pair,
            units,
            order.avg_fill_price,
            reason,
        )

    def _close_position(
        self, pair: str, reason: str, now: datetime
    ) -> None:
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
        self._trades_closed_today += 1

        # Phase C: Update bandit with trade outcome
        if self.config.enable_signal_filter and entry_info:
            try:
                cached_df = self._candle_cache.get(pair)
                if cached_df is not None:
                    features = self.feature_extractor.extract(
                        cached_df, pair
                    )
                    feat_array = self.feature_extractor.to_array(
                        features
                    )
                    # Normalize PnL for reward
                    reward = np.clip(pnl / 1000.0, -1.0, 1.0)
                    self.bandit.update(feat_array, "TAKE", reward)

                    # Shadow evaluation
                    if not self._bandit_active:
                        action, _ = self.bandit.decide(feat_array)
                        self.shadow_evaluator.record(
                            feat_array, action, pnl
                        )
                        should_activate, stats = (
                            self.shadow_evaluator.should_activate()
                        )
                        if should_activate:
                            self._bandit_active = True
                            logger.info(
                                "Bandit activated: %s", stats
                            )
            except Exception as e:
                logger.debug("Bandit update failed: %s", e)

        # Record trade
        self.monitor.record_trade(
            TradeRecord(
                timestamp=now,
                pair=pair,
                side="BUY",
                entry_price=entry_info["entry_price"]
                if entry_info
                else 0,
                exit_price=exit_price,
                quantity=float(
                    entry_info["units"] if entry_info else 0
                ),
                pnl=pnl,
                pnl_pct=(
                    pnl
                    / (entry_info["entry_price"] * entry_info["units"])
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
                "entry_price": entry_info["entry_price"]
                if entry_info
                else 0,
                "exit_price": exit_price,
                "quantity": float(
                    entry_info["units"] if entry_info else 0
                ),
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
                f"{pair} closed @ {exit_price:.5f}, "
                f"PnL={pnl:.2f} — {reason}",
            )

        self.tlogger.log_position_close(pair, pnl, reason)
        logger.info(
            "Closed %s @ %.5f, PnL=%.2f — %s",
            pair,
            exit_price,
            pnl,
            reason,
        )

    def _check_scaling(
        self, pair: str, df: pd.DataFrame
    ) -> None:
        """Check for scale-in/scale-out opportunities.

        Args:
            pair: Currency pair with open position.
            df: Current candle data.
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
        if self.scale_manager.should_add(
            position, current_price, current_atr
        ):
            add_units = self.scale_manager.tranche_units(
                position["units"]
            )
            order = self.broker.submit_market_order(
                pair=pair, side=OrderSide.BUY, units=add_units
            )
            if order and order.avg_fill_price:
                # Update average entry price
                total_units = position["units"] + add_units
                position["entry_price"] = (
                    position["entry_price"] * position["units"]
                    + order.avg_fill_price * add_units
                ) / total_units
                position["units"] = total_units
                position["tranche_count"] = (
                    position.get("tranche_count", 1) + 1
                )
                logger.info(
                    "Scaled in %s: +%d units (total %d)",
                    pair,
                    add_units,
                    total_units,
                )

        # Check scale-out (partial profit)
        elif self.scale_manager.should_reduce(
            position, current_price, current_atr
        ):
            level_idx = position.get("levels_taken", 0)
            reduce_units = self.scale_manager.reduction_units(
                position["units"], level_idx
            )
            if reduce_units > 0 and reduce_units < position["units"]:
                order = self.broker.submit_market_order(
                    pair=pair, side=OrderSide.SELL, units=reduce_units
                )
                if order and order.avg_fill_price:
                    position["units"] -= reduce_units
                    position["levels_taken"] = level_idx + 1
                    partial_pnl = (
                        order.avg_fill_price - position["entry_price"]
                    ) * reduce_units
                    logger.info(
                        "Scaled out %s: -%d units (PnL=%.2f, "
                        "remaining %d)",
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
                logger.error(
                    "Failed to close %s during abort: %s", pair, e
                )

    def _sync_positions(self) -> None:
        """Sync internal state with OANDA's actual positions."""
        positions = self.broker.get_all_positions()
        if positions is None:
            logger.warning("Could not sync positions from OANDA")
            return

        for pos in positions:
            pair = pos["pair"]
            if pair in [p for p in self.config.pairs]:
                self._strategy_positions[pair] = {
                    "entry_price": pos["avg_price"],
                    "entry_time": datetime.now(UTC),
                    "units": pos["units"],
                    "tranche_count": 1,
                    "levels_taken": 0,
                }
                logger.info(
                    "Synced position: %s %d units @ %.5f",
                    pair,
                    pos["units"],
                    pos["avg_price"],
                )

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
            daily_pnl / self._day_start_equity
            if self._day_start_equity > 0
            else 0
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

        if self.protocol.status.value == "running":
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
                    "positions_count": len(
                        self._strategy_positions
                    ),
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
                df = self.broker.fetch_candles(
                    pair, count=720, granularity="H1"
                )
                if df is not None and len(df) >= 220:
                    df["swap_long"] = self.config.swap_long_default
                    df["swap_short"] = self.config.swap_short_default
                    recent_candles[pair] = df

            if not recent_candles:
                logger.warning(
                    "No candle data for optimization, skipping"
                )
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
            logger.warning(
                "Optimization skipped (missing dependency): %s", e
            )
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
            recent = [
                t for t in trades if t.get("status") == "closed"
            ][-100:]  # Last 100 closed trades

            if len(recent) < 10:
                logger.info(
                    "Not enough recent trades for retraining "
                    "(%d < 10)",
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
