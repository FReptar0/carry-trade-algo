#!/usr/bin/env python3
"""Docker entry point for the carry trade live/paper trading system.

Loads configuration from environment variables, validates credentials,
creates all system components, and starts the trading runner.

Required environment variables:
    OANDA_ACCESS_TOKEN  - OANDA API access token
    OANDA_ACCOUNT_ID    - OANDA practice account ID

Optional environment variables:
    OANDA_ENVIRONMENT   - "practice" (default, only supported value)
    TRADING_PAIRS       - Comma-separated pairs (default: "USD/JPY,AUD/JPY")
    POSITION_SIZE_UNITS - Units per trade (default: 10000)
    CHECK_INTERVAL_MIN  - Minutes between ticks (default: 60)
    INITIAL_EQUITY      - Starting equity (default: 100000)
    LOG_DIR             - Log directory (default: "logs")
    DB_PATH             - SQLite path (default: "data/trading.db")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.broker.oanda import OandaBroker, OandaConfig
from src.engine.runner import RunnerConfig, TradingRunner


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_live")

    # Validate required environment variables
    access_token = os.environ.get("OANDA_ACCESS_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")

    if not access_token:
        logger.critical("OANDA_ACCESS_TOKEN not set")
        sys.exit(1)
    if not account_id:
        logger.critical("OANDA_ACCOUNT_ID not set")
        sys.exit(1)

    # Parse optional config
    environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
    pairs_str = os.environ.get("TRADING_PAIRS", "USD/JPY,AUD/JPY")
    pairs = [p.strip() for p in pairs_str.split(",")]
    position_units = int(
        os.environ.get("POSITION_SIZE_UNITS", "10000")
    )
    interval = int(os.environ.get("CHECK_INTERVAL_MIN", "60"))
    initial_equity = float(
        os.environ.get("INITIAL_EQUITY", "100000")
    )
    log_dir = os.environ.get("LOG_DIR", "logs")
    db_path = os.environ.get("DB_PATH", "data/trading.db")

    # Build OANDA instruments list
    oanda_instruments = [p.replace("/", "_") for p in pairs]

    # Create broker
    oanda_config = OandaConfig(
        access_token=access_token,
        account_id=account_id,
        environment=environment,
        instruments=oanda_instruments,
    )
    broker = OandaBroker(oanda_config)

    # Verify connectivity
    logger.info("Verifying OANDA connection...")
    acct = broker.get_account_state()
    if acct is None:
        logger.critical("Cannot connect to OANDA. Check credentials.")
        sys.exit(1)
    logger.info(
        "Connected to OANDA: balance=%.2f equity=%.2f",
        acct["balance"],
        acct["equity"],
    )

    # Create runner
    runner_config = RunnerConfig(
        pairs=pairs,
        check_interval_minutes=interval,
        position_size_units=position_units,
        initial_equity=initial_equity,
        log_dir=log_dir,
        db_path=db_path,
    )
    runner = TradingRunner(broker, runner_config)

    logger.info(
        "Starting carry trade runner: pairs=%s interval=%dmin",
        pairs,
        interval,
    )
    runner.start()


if __name__ == "__main__":
    main()
