"""Robust logging system for paper trading.

Log categories:
1. TRADES: Every signal, order, fill
2. ERRORS: Exceptions with full traceback
3. PERFORMANCE: P&L updates, metrics
4. SYSTEM: Application lifecycle, health checks

Log levels:
- DEBUG: Detailed diagnostic info
- INFO: Normal operational messages
- WARNING: Something unexpected but handled
- ERROR: Something failed
- CRITICAL: System must stop

File structure:
logs/
├── trades/
│   └── 2024-01-27_trades.log
├── errors/
│   └── 2024-01-27_errors.log
├── performance/
│   └── 2024-01-27_metrics.log
└── system/
    └── 2024-01-27_system.log

Each log file rotates daily. Old logs can be archived or deleted.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 30,  # Keep 30 days
) -> dict[str, logging.Logger]:
    """Set up the logging system.

    Creates separate loggers for trades, errors, performance, and system.

    Args:
        log_dir: Base directory for log files.
        level: Minimum log level.
        console: Whether to also log to console.
        max_bytes: Max size before rotation (default 10MB).
        backup_count: Number of backup files to keep.

    Returns:
        Dict of loggers by category.
    """
    log_path = Path(log_dir)

    # Create subdirectories
    categories = ["trades", "errors", "performance", "system"]
    for cat in categories:
        (log_path / cat).mkdir(parents=True, exist_ok=True)

    loggers = {}
    date_str = datetime.now().strftime("%Y-%m-%d")

    for cat in categories:
        logger = logging.getLogger(f"carry_trade.{cat}")
        logger.setLevel(level)
        logger.handlers.clear()

        # File handler with rotation
        file_path = log_path / cat / f"{date_str}_{cat}.log"
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)

        # Use JSON format for structured logging
        if cat == "trades":
            formatter = TradeLogFormatter()
        elif cat == "performance":
            formatter = PerformanceLogFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler for errors and system
        if console and cat in ("errors", "system"):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING if cat == "system" else logging.ERROR)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        loggers[cat] = logger

    return loggers


class TradeLogFormatter(logging.Formatter):
    """JSON formatter for trade logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "trade_data"):
            log_data["trade"] = record.trade_data

        return json.dumps(log_data)


class PerformanceLogFormatter(logging.Formatter):
    """JSON formatter for performance logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics

        return json.dumps(log_data)


class TradingLogger:
    """Convenience wrapper for trading-specific logging.

    Example:
        >>> logger = TradingLogger()
        >>> logger.log_signal("USD/JPY", "LONG", "High positive carry")
        >>> logger.log_fill(fill_object)
        >>> logger.log_pnl(equity=105000, daily_pnl=500)
    """

    def __init__(self, log_dir: str = "logs") -> None:
        loggers = setup_logging(log_dir)
        self.trades = loggers["trades"]
        self.errors = loggers["errors"]
        self.performance = loggers["performance"]
        self.system = loggers["system"]

    def log_signal(self, pair: str, signal: str, reason: str) -> None:
        """Log a trading signal."""
        record = self.trades.makeRecord(
            self.trades.name,
            logging.INFO,
            "",
            0,
            f"Signal: {pair} {signal}",
            (),
            None,
        )
        record.trade_data = {
            "type": "signal",
            "pair": pair,
            "signal": signal,
            "reason": reason,
        }
        self.trades.handle(record)

    def log_order(self, order_dict: dict) -> None:
        """Log an order submission."""
        record = self.trades.makeRecord(
            self.trades.name,
            logging.INFO,
            "",
            0,
            f"Order: {order_dict.get('order_id')} {order_dict.get('side')} {order_dict.get('pair')}",
            (),
            None,
        )
        record.trade_data = {"type": "order", **order_dict}
        self.trades.handle(record)

    def log_fill(self, fill_dict: dict) -> None:
        """Log an order fill."""
        record = self.trades.makeRecord(
            self.trades.name,
            logging.INFO,
            "",
            0,
            f"Fill: {fill_dict.get('order_id')} @ {fill_dict.get('price')}",
            (),
            None,
        )
        record.trade_data = {"type": "fill", **fill_dict}
        self.trades.handle(record)

    def log_position_close(self, pair: str, pnl: float, reason: str) -> None:
        """Log a position close."""
        record = self.trades.makeRecord(
            self.trades.name,
            logging.INFO,
            "",
            0,
            f"Close: {pair} PnL={pnl:+.2f} ({reason})",
            (),
            None,
        )
        record.trade_data = {
            "type": "close",
            "pair": pair,
            "pnl": pnl,
            "reason": reason,
        }
        self.trades.handle(record)

    def log_pnl(
        self,
        equity: float,
        balance: float,
        daily_pnl: float,
        unrealized_pnl: float,
    ) -> None:
        """Log P&L update."""
        record = self.performance.makeRecord(
            self.performance.name,
            logging.INFO,
            "",
            0,
            f"PnL Update: Equity={equity:.2f} Daily={daily_pnl:+.2f}",
            (),
            None,
        )
        record.metrics = {
            "type": "pnl",
            "equity": equity,
            "balance": balance,
            "daily_pnl": daily_pnl,
            "unrealized_pnl": unrealized_pnl,
        }
        self.performance.handle(record)

    def log_metrics(self, metrics: dict) -> None:
        """Log performance metrics."""
        record = self.performance.makeRecord(
            self.performance.name,
            logging.INFO,
            "",
            0,
            "Metrics update",
            (),
            None,
        )
        record.metrics = {"type": "metrics", **metrics}
        self.performance.handle(record)

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with traceback."""
        self.errors.exception(f"Error in {context}: {error}")

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.errors.warning(message)

    def log_system(self, message: str, level: int = logging.INFO) -> None:
        """Log a system message."""
        self.system.log(level, message)

    def log_startup(self, config: dict) -> None:
        """Log system startup."""
        self.system.info(f"System starting with config: {json.dumps(config, default=str)}")

    def log_shutdown(self, reason: str = "normal") -> None:
        """Log system shutdown."""
        self.system.info(f"System shutting down: {reason}")
