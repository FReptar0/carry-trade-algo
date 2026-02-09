"""SQLite-based state persistence for the trading system.

Stores protocol state, daily results, equity snapshots, trade logs,
checkpoints, and live position metadata so the system can resume
after container restarts without losing position state.

Tables:
    protocol_state  - Current protocol status and configuration
    daily_results   - One row per trading day
    equity_snapshots - Periodic equity/balance snapshots
    trade_log       - Completed trade records
    checkpoints     - Protocol checkpoint results
    position_states - Live position metadata (survives restarts)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from src.validation.protocol import (
    AbortCriteria,
    ProtocolDay,
    ProtocolStatus,
    TradingProtocol,
)


class StateStore:
    """SQLite persistence layer for the trading system.

    Args:
        db_path: Path to the SQLite database file.

    Example:
        >>> store = StateStore("data/trading.db")
        >>> store.save_protocol_state(protocol)
        >>> restored = store.load_protocol_state()
    """

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS protocol_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    status TEXT NOT NULL,
                    start_date TEXT,
                    duration_days INTEGER NOT NULL,
                    initial_equity REAL NOT NULL,
                    abort_reason TEXT,
                    max_drawdown_limit REAL NOT NULL,
                    consecutive_losing_limit INTEGER NOT NULL,
                    circuit_breaker_limit INTEGER NOT NULL,
                    min_win_rate_limit REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    starting_equity REAL NOT NULL,
                    ending_equity REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    daily_return REAL NOT NULL,
                    trades_opened INTEGER NOT NULL,
                    trades_closed INTEGER NOT NULL,
                    max_drawdown_today REAL NOT NULL,
                    regime TEXT NOT NULL,
                    circuit_breaker_triggered INTEGER NOT NULL,
                    notes TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    balance REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    positions_count INTEGER NOT NULL,
                    drawdown REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL,
                    swap_earned REAL DEFAULT 0,
                    status TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    cumulative_return REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    current_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    sharpe_rolling REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    issues TEXT NOT NULL,
                    recommendation TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS position_states (
                    pair TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _migrate_db(self) -> None:
        """Apply incremental schema migrations."""
        conn = self._get_conn()
        try:
            # Add degraded_reason column if missing (Phase 2 hardening)
            proto_cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(protocol_state)").fetchall()
            ]
            if "degraded_reason" not in proto_cols:
                conn.execute(
                    "ALTER TABLE protocol_state ADD COLUMN degraded_reason TEXT"
                )

            # Add financing columns (swap PnL tracking)
            equity_cols = [
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(equity_snapshots)"
                ).fetchall()
            ]
            if "financing" not in equity_cols:
                conn.execute(
                    "ALTER TABLE equity_snapshots ADD COLUMN financing REAL DEFAULT 0"
                )

            daily_cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(daily_results)").fetchall()
            ]
            if "financing" not in daily_cols:
                conn.execute(
                    "ALTER TABLE daily_results ADD COLUMN financing REAL DEFAULT 0"
                )

            conn.commit()
        finally:
            conn.close()

    def save_protocol_state(self, protocol: TradingProtocol) -> None:
        """Persist the current protocol state.

        Args:
            protocol: The TradingProtocol instance to save.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO protocol_state
                (id, status, start_date, duration_days, initial_equity,
                 abort_reason, degraded_reason,
                 max_drawdown_limit, consecutive_losing_limit,
                 circuit_breaker_limit, min_win_rate_limit, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    protocol.status.value,
                    protocol.start_date.isoformat() if protocol.start_date else None,
                    protocol.duration,
                    protocol.initial_equity,
                    protocol.abort_reason,
                    protocol.degraded_reason,
                    protocol.abort_criteria.max_drawdown,
                    protocol.abort_criteria.consecutive_losing_days,
                    protocol.abort_criteria.circuit_breaker_count,
                    protocol.abort_criteria.min_win_rate,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_protocol_state(self) -> Optional[TradingProtocol]:
        """Load protocol state from database.

        Returns:
            Restored TradingProtocol, or None if no saved state.
        """
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM protocol_state WHERE id = 1").fetchone()
            if row is None:
                return None

            criteria = AbortCriteria(
                max_drawdown=row["max_drawdown_limit"],
                consecutive_losing_days=row["consecutive_losing_limit"],
                circuit_breaker_count=row["circuit_breaker_limit"],
                min_win_rate=row["min_win_rate_limit"],
            )

            protocol = TradingProtocol(
                duration_days=row["duration_days"],
                abort_criteria=criteria,
                initial_equity=row["initial_equity"],
            )

            protocol.status = ProtocolStatus(row["status"])
            if row["start_date"]:
                protocol.start_date = date.fromisoformat(row["start_date"])
            protocol.abort_reason = row["abort_reason"]
            protocol.degraded_reason = row["degraded_reason"]

            # Restore daily results
            days = conn.execute("SELECT * FROM daily_results ORDER BY date").fetchall()
            for d in days:
                protocol.days.append(
                    ProtocolDay(
                        date=date.fromisoformat(d["date"]),
                        starting_equity=d["starting_equity"],
                        ending_equity=d["ending_equity"],
                        daily_pnl=d["daily_pnl"],
                        daily_return=d["daily_return"],
                        trades_opened=d["trades_opened"],
                        trades_closed=d["trades_closed"],
                        max_drawdown_today=d["max_drawdown_today"],
                        regime=d["regime"],
                        circuit_breaker_triggered=bool(d["circuit_breaker_triggered"]),
                        notes=d["notes"] or "",
                    )
                )

            return protocol
        finally:
            conn.close()

    def save_daily_result(self, day: ProtocolDay) -> None:
        """Save a single day's results.

        Args:
            day: The ProtocolDay to persist.
        """
        conn = self._get_conn()
        try:
            financing = getattr(day, "financing", 0.0)
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_results
                (date, starting_equity, ending_equity, daily_pnl,
                 daily_return, trades_opened, trades_closed,
                 max_drawdown_today, regime, circuit_breaker_triggered,
                 notes, financing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    day.date.isoformat(),
                    day.starting_equity,
                    day.ending_equity,
                    day.daily_pnl,
                    day.daily_return,
                    day.trades_opened,
                    day.trades_closed,
                    day.max_drawdown_today,
                    day.regime,
                    int(day.circuit_breaker_triggered),
                    day.notes,
                    financing,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_equity_snapshot(
        self,
        timestamp: datetime,
        equity: float,
        balance: float,
        unrealized_pnl: float,
        positions_count: int,
        drawdown: float,
        financing: float = 0.0,
    ) -> None:
        """Save a periodic equity snapshot.

        Args:
            timestamp: Snapshot time.
            equity: Current equity.
            balance: Current balance.
            unrealized_pnl: Unrealized P&L.
            positions_count: Number of open positions.
            drawdown: Current drawdown percentage.
            financing: Cumulative financing/swap across all positions.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO equity_snapshots
                (timestamp, equity, balance, unrealized_pnl,
                 positions_count, drawdown, financing)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(),
                    equity,
                    balance,
                    unrealized_pnl,
                    positions_count,
                    drawdown,
                    financing,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_trade(self, trade: dict) -> None:
        """Save a trade record.

        Args:
            trade: Dictionary with trade details. Expected keys:
                timestamp, pair, side, entry_price, quantity, status.
                Optional: exit_price, pnl, swap_earned, metadata.
        """
        conn = self._get_conn()
        try:
            metadata = trade.get("metadata", {})
            conn.execute(
                """
                INSERT INTO trade_log
                (timestamp, pair, side, entry_price, exit_price,
                 quantity, pnl, swap_earned, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["timestamp"],
                    trade["pair"],
                    trade["side"],
                    trade["entry_price"],
                    trade.get("exit_price"),
                    trade["quantity"],
                    trade.get("pnl"),
                    trade.get("swap_earned", 0.0),
                    trade["status"],
                    json.dumps(metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_checkpoint(
        self,
        day: int,
        checkpoint_date: date,
        cumulative_return: float,
        max_drawdown: float,
        current_drawdown: float,
        win_rate: float,
        sharpe_rolling: float,
        total_trades: int,
        issues: list[str],
        recommendation: str,
    ) -> None:
        """Save a protocol checkpoint.

        Args:
            day: Checkpoint day number (7, 14, 21, 30).
            checkpoint_date: Date of checkpoint.
            cumulative_return: Return since start.
            max_drawdown: Maximum drawdown observed.
            current_drawdown: Current drawdown.
            win_rate: Win rate percentage.
            sharpe_rolling: Rolling Sharpe ratio.
            total_trades: Total trades completed.
            issues: List of identified issues.
            recommendation: "continue", "pause", or "abort".
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO checkpoints
                (day, date, cumulative_return, max_drawdown,
                 current_drawdown, win_rate, sharpe_rolling,
                 total_trades, issues, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    day,
                    checkpoint_date.isoformat(),
                    cumulative_return,
                    max_drawdown,
                    current_drawdown,
                    win_rate,
                    sharpe_rolling,
                    total_trades,
                    json.dumps(issues),
                    recommendation,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_latest_equity(self) -> Optional[float]:
        """Get the most recent equity value.

        Returns:
            Latest equity, or None if no snapshots exist.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return row["equity"] if row else None
        finally:
            conn.close()

    def get_trades(self, pair: Optional[str] = None) -> list[dict]:
        """Get trade log entries.

        Args:
            pair: Optional filter by currency pair.

        Returns:
            List of trade dictionaries.
        """
        conn = self._get_conn()
        try:
            if pair:
                rows = conn.execute(
                    "SELECT * FROM trade_log WHERE pair = ? ORDER BY timestamp",
                    (pair,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trade_log ORDER BY timestamp"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_total_financing(self) -> float:
        """Get total financing/swap earned from all closed trades.

        Returns:
            Sum of swap_earned from all closed trades.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(swap_earned), 0) as total "
                "FROM trade_log WHERE status = 'closed'"
            ).fetchone()
            return float(row["total"]) if row else 0.0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Position state persistence (survives container restarts)
    # ------------------------------------------------------------------

    def save_positions(self, positions: dict[str, dict]) -> None:
        """Persist all open position dicts to SQLite.

        For each pair, serializes the position dict to JSON (converting
        datetime objects to ISO strings) and upserts into position_states.
        Rows for pairs no longer in *positions* are deleted (stale cleanup).

        Args:
            positions: Mapping of pair -> position dict.
        """
        conn = self._get_conn()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()

            for pair, pos in positions.items():
                serializable = self._serialize_position(pos)
                state_json = json.dumps(serializable)
                conn.execute(
                    "INSERT INTO position_states (pair, state_json, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(pair) DO UPDATE SET "
                    "state_json = excluded.state_json, "
                    "updated_at = excluded.updated_at",
                    (pair, state_json, now_iso),
                )

            # Delete stale rows for pairs no longer held
            if positions:
                placeholders = ",".join("?" for _ in positions)
                conn.execute(
                    f"DELETE FROM position_states WHERE pair NOT IN ({placeholders})",
                    list(positions.keys()),
                )
            else:
                conn.execute("DELETE FROM position_states")

            conn.commit()
        finally:
            conn.close()

    def load_positions(self) -> dict[str, dict]:
        """Load all persisted position states from SQLite.

        Deserializes JSON back into dicts, converting ISO datetime
        strings back to ``datetime`` objects for known datetime fields.

        Returns:
            Mapping of pair -> position dict. Empty dict if no data.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT pair, state_json FROM position_states"
            ).fetchall()

            result: dict[str, dict] = {}
            for row in rows:
                pos = json.loads(row["state_json"])
                self._deserialize_datetimes(pos)
                result[row["pair"]] = pos
            return result
        finally:
            conn.close()

    def delete_position(self, pair: str) -> None:
        """Remove a single position from the position_states table.

        Called when a position is closed so stale data is not restored
        on the next restart.

        Args:
            pair: Currency pair to remove (e.g. "USD/JPY").
        """
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM position_states WHERE pair = ?", (pair,))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers for position serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_position(pos: dict) -> dict:
        """Convert a position dict to a JSON-safe dict.

        Converts ``datetime`` values to ISO 8601 strings.
        """
        result = {}
        for key, value in pos.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    _DATETIME_FIELDS = {"entry_time", "last_vol_trim_time"}

    @classmethod
    def _deserialize_datetimes(cls, pos: dict) -> None:
        """Convert known ISO datetime strings back to ``datetime`` objects.

        Modifies *pos* in place.
        """
        for field in cls._DATETIME_FIELDS:
            value = pos.get(field)
            if isinstance(value, str):
                try:
                    pos[field] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    pass  # Leave as-is if unparseable
