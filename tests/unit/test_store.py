"""Tests for SQLite state persistence."""

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from src.persistence.store import StateStore
from src.validation.protocol import (
    AbortCriteria,
    ProtocolDay,
    ProtocolStatus,
    TradingProtocol,
)


@pytest.fixture
def store(tmp_path):
    """Create a StateStore backed by a temp database."""
    db_path = str(tmp_path / "test_trading.db")
    return StateStore(db_path)


@pytest.fixture
def sample_protocol():
    """Create a sample TradingProtocol."""
    protocol = TradingProtocol(
        duration_days=30,
        initial_equity=100000.0,
    )
    protocol.start()
    return protocol


@pytest.fixture
def sample_day():
    """Create a sample ProtocolDay."""
    return ProtocolDay(
        date=date(2026, 2, 1),
        starting_equity=100000,
        ending_equity=100500,
        daily_pnl=500,
        daily_return=0.005,
        trades_opened=1,
        trades_closed=0,
        max_drawdown_today=0.012,
        regime="LIVE",
        circuit_breaker_triggered=False,
        notes="Test day",
    )


class TestStateStoreInit:
    """Tests for StateStore initialization."""

    def test_creates_database(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = StateStore(db_path)
        assert Path(db_path).exists()

    def test_creates_parent_directory(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "test.db")
        store = StateStore(db_path)
        assert Path(db_path).exists()

    def test_tables_created(self, store):
        conn = sqlite3.connect(store.db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "protocol_state" in tables
        assert "daily_results" in tables
        assert "equity_snapshots" in tables
        assert "trade_log" in tables
        assert "checkpoints" in tables


class TestProtocolPersistence:
    """Tests for saving/loading protocol state."""

    def test_save_and_load(self, store, sample_protocol):
        store.save_protocol_state(sample_protocol)
        loaded = store.load_protocol_state()
        assert loaded is not None
        assert loaded.status == ProtocolStatus.RUNNING
        assert loaded.initial_equity == 100000.0
        assert loaded.duration == 30

    def test_load_returns_none_if_empty(self, store):
        loaded = store.load_protocol_state()
        assert loaded is None

    def test_restores_days(self, store, sample_protocol, sample_day):
        sample_protocol.record_day(sample_day)
        store.save_protocol_state(sample_protocol)
        store.save_daily_result(sample_day)

        loaded = store.load_protocol_state()
        assert loaded is not None
        assert len(loaded.days) == 1
        assert loaded.days[0].daily_pnl == 500
        assert loaded.days[0].date == date(2026, 2, 1)

    def test_restores_abort_criteria(self, store):
        criteria = AbortCriteria(
            max_drawdown=0.10,
            consecutive_losing_days=3,
            circuit_breaker_count=2,
            min_win_rate=0.30,
        )
        protocol = TradingProtocol(
            duration_days=30,
            abort_criteria=criteria,
            initial_equity=50000.0,
        )
        protocol.start()
        store.save_protocol_state(protocol)

        loaded = store.load_protocol_state()
        assert loaded is not None
        assert loaded.abort_criteria.max_drawdown == 0.10
        assert loaded.abort_criteria.consecutive_losing_days == 3
        assert loaded.abort_criteria.circuit_breaker_count == 2
        assert loaded.abort_criteria.min_win_rate == 0.30

    def test_save_overwrites(self, store, sample_protocol):
        store.save_protocol_state(sample_protocol)
        sample_protocol.abort("test reason")
        store.save_protocol_state(sample_protocol)

        loaded = store.load_protocol_state()
        assert loaded is not None
        assert loaded.status == ProtocolStatus.ABORTED
        assert loaded.abort_reason == "test reason"


class TestDailyResults:
    """Tests for daily result persistence."""

    def test_save_and_retrieve(self, store, sample_day):
        store.save_daily_result(sample_day)
        # Verify via direct query
        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM daily_results WHERE date = ?",
            (sample_day.date.isoformat(),),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["daily_pnl"] == 500
        assert row["trades_opened"] == 1

    def test_upsert_on_duplicate(self, store, sample_day):
        store.save_daily_result(sample_day)
        # Modify and save again
        updated = ProtocolDay(
            date=sample_day.date,
            starting_equity=100000,
            ending_equity=101000,
            daily_pnl=1000,
            daily_return=0.01,
            trades_opened=2,
            trades_closed=1,
            max_drawdown_today=0.005,
            regime="LIVE",
            circuit_breaker_triggered=False,
        )
        store.save_daily_result(updated)

        conn = sqlite3.connect(store.db_path)
        count = conn.execute("SELECT COUNT(*) FROM daily_results").fetchone()[0]
        conn.close()
        assert count == 1


class TestEquitySnapshots:
    """Tests for equity snapshot persistence."""

    def test_save_snapshot(self, store):
        now = datetime(2026, 2, 1, 12, 0)
        store.save_equity_snapshot(
            timestamp=now,
            equity=101000,
            balance=100500,
            unrealized_pnl=500,
            positions_count=1,
            drawdown=0.005,
        )
        latest = store.get_latest_equity()
        assert latest == 101000

    def test_get_latest_with_multiple(self, store):
        for i in range(3):
            store.save_equity_snapshot(
                timestamp=datetime(2026, 2, 1, i, 0),
                equity=100000 + i * 100,
                balance=100000,
                unrealized_pnl=0,
                positions_count=0,
                drawdown=0,
            )
        latest = store.get_latest_equity()
        assert latest == 100200

    def test_get_latest_returns_none_if_empty(self, store):
        assert store.get_latest_equity() is None


class TestTradeLog:
    """Tests for trade log persistence."""

    def test_save_trade(self, store):
        trade = {
            "timestamp": "2026-02-01T12:00:00",
            "pair": "USD/JPY",
            "side": "BUY",
            "entry_price": 155.50,
            "quantity": 10000,
            "status": "open",
        }
        store.save_trade(trade)
        trades = store.get_trades()
        assert len(trades) == 1
        assert trades[0]["pair"] == "USD/JPY"

    def test_get_trades_filtered_by_pair(self, store):
        for pair in ["USD/JPY", "AUD/JPY", "USD/JPY"]:
            store.save_trade(
                {
                    "timestamp": "2026-02-01T12:00:00",
                    "pair": pair,
                    "side": "BUY",
                    "entry_price": 155.0,
                    "quantity": 10000,
                    "status": "open",
                }
            )
        trades = store.get_trades(pair="USD/JPY")
        assert len(trades) == 2

    def test_trade_with_metadata(self, store):
        trade = {
            "timestamp": "2026-02-01T12:00:00",
            "pair": "USD/JPY",
            "side": "BUY",
            "entry_price": 155.50,
            "quantity": 10000,
            "status": "open",
            "metadata": {"reason": "Golden Cross"},
        }
        store.save_trade(trade)
        trades = store.get_trades()
        assert len(trades) == 1


class TestCheckpoints:
    """Tests for checkpoint persistence."""

    def test_save_checkpoint(self, store):
        store.save_checkpoint(
            day=7,
            checkpoint_date=date(2026, 2, 7),
            cumulative_return=0.02,
            max_drawdown=0.05,
            current_drawdown=0.01,
            win_rate=0.60,
            sharpe_rolling=1.5,
            total_trades=5,
            issues=["Low win rate: 60%"],
            recommendation="continue",
        )
        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM checkpoints WHERE day = 7").fetchone()
        conn.close()
        assert row is not None
        assert row["recommendation"] == "continue"
        assert row["cumulative_return"] == 0.02


class TestFinancingMigration:
    """Tests for financing/swap column migrations."""

    def test_equity_snapshots_has_financing_column(self, store):
        """Migration should add financing column to equity_snapshots."""
        conn = sqlite3.connect(store.db_path)
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(equity_snapshots)").fetchall()
        ]
        conn.close()
        assert "financing" in cols

    def test_daily_results_has_financing_column(self, store):
        """Migration should add financing column to daily_results."""
        conn = sqlite3.connect(store.db_path)
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(daily_results)").fetchall()
        ]
        conn.close()
        assert "financing" in cols

    def test_save_equity_snapshot_with_financing(self, store):
        """Equity snapshot should persist the financing value."""
        now = datetime(2026, 2, 8, 12, 0)
        store.save_equity_snapshot(
            timestamp=now,
            equity=100500,
            balance=100200,
            unrealized_pnl=300,
            positions_count=3,
            drawdown=0.005,
            financing=12.45,
        )
        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["financing"] == pytest.approx(12.45)

    def test_save_equity_snapshot_defaults_financing_zero(self, store):
        """Financing should default to 0 if not provided."""
        now = datetime(2026, 2, 8, 13, 0)
        store.save_equity_snapshot(
            timestamp=now,
            equity=100500,
            balance=100200,
            unrealized_pnl=300,
            positions_count=3,
            drawdown=0.005,
        )
        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row["financing"] == pytest.approx(0.0)

    def test_save_daily_result_with_financing(self, store):
        """Daily result should persist financing via getattr fallback."""
        day = ProtocolDay(
            date=date(2026, 2, 8),
            starting_equity=100000,
            ending_equity=100500,
            daily_pnl=500,
            daily_return=0.005,
            trades_opened=1,
            trades_closed=0,
            max_drawdown_today=0.01,
            regime="LIVE",
            circuit_breaker_triggered=False,
        )
        # ProtocolDay doesn't have financing attr, so getattr defaults to 0
        store.save_daily_result(day)
        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM daily_results WHERE date = '2026-02-08'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["financing"] == pytest.approx(0.0)

    def test_get_total_financing(self, store):
        """get_total_financing should sum swap_earned from closed trades."""
        for i, swap in enumerate([5.50, 3.25, -1.00]):
            store.save_trade(
                {
                    "timestamp": f"2026-02-0{i + 1}T12:00:00",
                    "pair": "USD/JPY",
                    "side": "SELL",
                    "entry_price": 155.0,
                    "exit_price": 156.0,
                    "quantity": 10000,
                    "pnl": 100.0,
                    "swap_earned": swap,
                    "status": "closed",
                }
            )
        # Also add an open trade (should be excluded)
        store.save_trade(
            {
                "timestamp": "2026-02-04T12:00:00",
                "pair": "AUD/JPY",
                "side": "BUY",
                "entry_price": 98.0,
                "quantity": 10000,
                "swap_earned": 99.99,
                "status": "open",
            }
        )
        total = store.get_total_financing()
        assert total == pytest.approx(7.75)

    def test_get_total_financing_empty(self, store):
        """get_total_financing should return 0 with no closed trades."""
        total = store.get_total_financing()
        assert total == pytest.approx(0.0)


class TestPositionStatePersistence:
    """Tests for save_positions / load_positions / delete_position."""

    def test_save_and_load_positions(self, store):
        """Round-trip: save two positions, load them back, verify all fields."""
        positions = {
            "USD/JPY": {
                "entry_price": 155.0,
                "units": 10000,
                "high_water_mark": 156.5,
                "low_water_mark": 154.0,
                "entry_atr": 0.65,
                "original_units": 10000,
                "tranche_count": 2,
                "levels_taken": 1,
                "financing": 5.50,
                "last_vol_trim_time": None,
            },
            "AUD/JPY": {
                "entry_price": 98.0,
                "units": 5000,
                "high_water_mark": 99.0,
                "low_water_mark": 97.5,
                "entry_atr": 0.40,
                "original_units": 8000,
                "tranche_count": 1,
                "levels_taken": 0,
                "financing": 3.25,
                "last_vol_trim_time": None,
            },
        }
        store.save_positions(positions)
        loaded = store.load_positions()

        assert set(loaded.keys()) == {"USD/JPY", "AUD/JPY"}
        assert loaded["USD/JPY"]["high_water_mark"] == 156.5
        assert loaded["USD/JPY"]["original_units"] == 10000
        assert loaded["USD/JPY"]["tranche_count"] == 2
        assert loaded["AUD/JPY"]["entry_atr"] == pytest.approx(0.40)
        assert loaded["AUD/JPY"]["financing"] == pytest.approx(3.25)

    def test_load_positions_empty(self, store):
        """load_positions returns empty dict when no data."""
        loaded = store.load_positions()
        assert loaded == {}

    def test_delete_position(self, store):
        """delete_position removes a specific pair."""
        positions = {
            "USD/JPY": {"entry_price": 155.0, "units": 10000},
            "AUD/JPY": {"entry_price": 98.0, "units": 5000},
        }
        store.save_positions(positions)
        store.delete_position("USD/JPY")

        loaded = store.load_positions()
        assert "USD/JPY" not in loaded
        assert "AUD/JPY" in loaded

    def test_save_positions_handles_datetime_serialization(self, store):
        """Datetimes survive round-trip through JSON serialization."""
        entry_time = datetime(2026, 2, 3, 4, 14, 56, 582484)
        trim_time = datetime(2026, 2, 7, 10, 30, 0)

        positions = {
            "USD/JPY": {
                "entry_price": 155.0,
                "units": 10000,
                "entry_time": entry_time,
                "last_vol_trim_time": trim_time,
            },
        }
        store.save_positions(positions)
        loaded = store.load_positions()

        assert loaded["USD/JPY"]["entry_time"] == entry_time
        assert loaded["USD/JPY"]["last_vol_trim_time"] == trim_time

    def test_save_positions_cleans_stale_pairs(self, store):
        """Pairs removed from the dict are deleted from the DB."""
        # Save two pairs
        store.save_positions(
            {
                "USD/JPY": {"entry_price": 155.0, "units": 10000},
                "AUD/JPY": {"entry_price": 98.0, "units": 5000},
            }
        )

        # Save only one pair — AUD/JPY should be deleted
        store.save_positions(
            {
                "USD/JPY": {"entry_price": 155.5, "units": 10000},
            }
        )

        loaded = store.load_positions()
        assert "USD/JPY" in loaded
        assert "AUD/JPY" not in loaded

    def test_save_positions_empty_clears_all(self, store):
        """Saving an empty dict clears all rows."""
        store.save_positions({"USD/JPY": {"entry_price": 155.0, "units": 10000}})
        store.save_positions({})
        loaded = store.load_positions()
        assert loaded == {}

    def test_save_positions_upsert_updates_existing(self, store):
        """Saving the same pair twice updates (not duplicates) the row."""
        store.save_positions({"USD/JPY": {"high_water_mark": 155.0}})
        store.save_positions({"USD/JPY": {"high_water_mark": 157.0}})
        loaded = store.load_positions()
        assert loaded["USD/JPY"]["high_water_mark"] == 157.0

    def test_delete_position_nonexistent_pair(self, store):
        """Deleting a pair that doesn't exist should not raise."""
        store.delete_position("NZD/JPY")  # Should be a no-op

    def test_datetime_none_survives_roundtrip(self, store):
        """last_vol_trim_time=None should not become a string."""
        positions = {
            "USD/JPY": {
                "entry_price": 155.0,
                "last_vol_trim_time": None,
                "entry_time": datetime(2026, 2, 1, 10, 0, 0),
            },
        }
        store.save_positions(positions)
        loaded = store.load_positions()
        assert loaded["USD/JPY"]["last_vol_trim_time"] is None
