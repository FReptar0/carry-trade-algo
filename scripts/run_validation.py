"""CLI runner for the validation & robustness suite.

Usage:
    # All tests on live DB
    uv run python scripts/run_validation.py --source live --db-path data/trading_live.db

    # HIGH priority only, fast
    uv run python scripts/run_validation.py --source live --priority high --n-sims 500

    # All tests on backtest data
    uv run python scripts/run_validation.py --source backtest

    # Specific tests
    uv run python scripts/run_validation.py --source live --tests monte_carlo,bootstrap_ci,permutation_test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import HistoricalDataLoader
from src.validation.robustness import (
    ValidationSuiteResult,
    load_from_backtest,
    load_from_multi_pair_backtest,
    load_from_live_db,
    run_validation_suite,
)

DEFAULT_PAIRS = ["USD/JPY", "AUD/JPY", "GBP/JPY", "NZD/JPY", "EUR/JPY", "CAD/JPY"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation & robustness suite for the carry trade algo."
    )
    parser.add_argument(
        "--source",
        choices=["backtest", "live"],
        default="backtest",
        help="Data source: 'backtest' runs fresh backtest, 'live' reads from SQLite DB.",
    )
    parser.add_argument(
        "--db-path",
        default="data/trading.db",
        help="Path to live trading SQLite DB (only used with --source live).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital for live DB data source.",
    )
    parser.add_argument(
        "--priority",
        choices=["high", "medium", "low", "all"],
        default="all",
        help="Which priority tier of tests to run.",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated list of specific tests to run.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5000,
        help="Number of Monte Carlo simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reports/validation",
        help="Output directory for charts and report.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (faster).",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Validation & Robustness Suite")
    print("=" * 60)

    # Parse test list
    test_list = None
    if args.tests:
        test_list = [t.strip() for t in args.tests.split(",")]

    # Hourly data limited to ~730 days; use recent cached date range
    DATA_START = "2024-06-01"
    DATA_END = "2025-12-31"

    # Load data
    data_loader = HistoricalDataLoader()
    price_data = None

    if args.source == "live":
        print(f"\nLoading live data from: {args.db_path}")
        data = load_from_live_db(args.db_path, args.initial_capital)
        print(f"  Trades: {len(data.trades_df)}")
        print(f"  Equity snapshots: {len(data.equity_df)}")
        print(f"  Daily returns: {len(data.daily_returns)}")

        # Load price data for regime/liquidity tests
        try:
            price_data = data_loader.load_pair("USD/JPY", start=DATA_START, end=DATA_END)
            print(f"  Price data loaded: {len(price_data)} bars")
        except Exception as e:
            print(f"  Warning: Could not load price data: {e}")

    else:
        print(f"\nRunning fresh backtest for {len(DEFAULT_PAIRS)} pairs...")
        from src.backtest.engine import BacktestEngine
        from src.config.settings import RiskConfig
        from src.strategy.carry_trade_v3 import CarryTradeStrategyV3, StrategyConfigV3

        config = StrategyConfigV3()
        risk_config = RiskConfig(initial_capital=100_000.0)

        # Run backtest on ALL pairs, combine into portfolio-level DataSource
        pair_results: dict[str, object] = {}
        total_trades = 0
        first_price_data = None

        for pair in DEFAULT_PAIRS:
            try:
                df = data_loader.load_pair(pair, start=DATA_START, end=DATA_END)
                if len(df) == 0:
                    df = data_loader.load_pair(pair, start="2023-01-01", end="2024-12-31", interval="1d")
                if len(df) == 0:
                    print(f"  {pair}: no data, skipping")
                    continue

                strategy = CarryTradeStrategyV3(config)
                engine = BacktestEngine(strategy, risk_config=risk_config)
                result = engine.run(df, pair)
                pair_results[pair] = result
                total_trades += result.metrics.total_trades
                print(f"  {pair}: {result.metrics.total_return:+.2%} return, "
                      f"{result.metrics.total_trades} trades")

                if first_price_data is None:
                    first_price_data = df
            except Exception as e:
                print(f"  {pair}: error - {e}")

        if not pair_results:
            print("  No pairs produced results. Exiting.")
            sys.exit(1)

        data = load_from_multi_pair_backtest(pair_results, initial_capital=100_000.0)
        price_data = first_price_data
        print(f"  Portfolio: {len(pair_results)} pairs, {total_trades} total trades, "
              f"{len(data.daily_returns)} daily returns")

    # Run suite
    print(f"\nRunning validation suite (priority={args.priority}, n_sims={args.n_sims})...")
    start = time.time()

    suite = run_validation_suite(
        data=data,
        data_loader=data_loader,
        pairs=DEFAULT_PAIRS,
        price_data=price_data,
        priority=args.priority,
        tests=test_list,
        n_sims=args.n_sims,
        seed=args.seed,
        generate_charts=not args.no_charts,
        output_dir=args.output_dir,
    )

    elapsed = time.time() - start

    # Print summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Passed: {suite.pass_count}/{suite.total_count}")
    print(f"  Time: {elapsed:.1f}s")
    print()

    for key, result in suite.results.items():
        passed = suite._check_passed(result)
        status = "PASS" if passed else "FAIL"
        icon = "[+]" if passed else "[-]"
        print(f"  {icon} {key}: {status}")

    print(f"\n  Report: {args.output_dir}/validation_report.md")
    if not args.no_charts:
        print(f"  Charts: {args.output_dir}/*.png")
    print()

    if suite.all_passed:
        print("  ALL TESTS PASSED - Strategy edge appears robust.")
    else:
        failed = [k for k, v in suite.results.items()
                  if not (all(r.passed for r in v) if isinstance(v, list) else v.passed)]
        print(f"  FAILURES: {', '.join(failed)}")
        print("  Review the report for details on failed tests.")


if __name__ == "__main__":
    main()
