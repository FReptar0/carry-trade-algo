"""SQLite-backed parameter store for regime-adapted parameters.

Versions and tracks parameter sets with their measured performance,
enabling data-driven parameter selection and historical comparison.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.regime.adapters import RegimeAdaptedParams

logger = logging.getLogger(__name__)


@dataclass
class ParamSet:
    """A versioned set of regime parameters with performance data.

    Attributes:
        id: Auto-incremented ID.
        regime: CompositeRegime string value.
        params: The adapted parameters.
        created_at: When this set was created.
        sharpe: Measured Sharpe ratio (None if not yet evaluated).
        win_rate: Measured win rate (None if not yet evaluated).
        sample_size: Number of trades evaluated on.
        is_active: Whether this is the currently active set.
    """

    id: int
    regime: str
    params: RegimeAdaptedParams
    created_at: datetime
    sharpe: Optional[float] = None
    win_rate: Optional[float] = None
    sample_size: int = 0
    is_active: bool = False


class ParamStore:
    """SQLite storage for versioned regime parameter sets.

    Args:
        db_path: Path to the SQLite database.

    Example:
        >>> store = ParamStore("data/trading.db")
        >>> pid = store.save_params("TREND_LOW_VOL", params)
        >>> active = store.get_active_params("TREND_LOW_VOL")
    """

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_table(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS param_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime TEXT NOT NULL,
                    stop_loss_mult REAL NOT NULL,
                    position_size_mult REAL NOT NULL,
                    entry_threshold REAL NOT NULL,
                    should_trade INTEGER NOT NULL,
                    should_close_on_weakness INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    sharpe REAL,
                    win_rate REAL,
                    sample_size INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 0
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_params(
        self, regime: str, params: RegimeAdaptedParams
    ) -> int:
        """Save a new parameter set.

        Args:
            regime: CompositeRegime string value.
            params: The parameters to store.

        Returns:
            ID of the new parameter set.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                INSERT INTO param_sets
                (regime, stop_loss_mult, position_size_mult,
                 entry_threshold, should_trade,
                 should_close_on_weakness, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    regime,
                    params.stop_loss_mult,
                    params.position_size_mult,
                    params.entry_threshold,
                    int(params.should_trade),
                    int(params.should_close_on_weakness),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_active_params(
        self, regime: str
    ) -> Optional[ParamSet]:
        """Get the currently active parameter set for a regime.

        Args:
            regime: CompositeRegime string value.

        Returns:
            Active ParamSet, or None if no active set exists.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM param_sets "
                "WHERE regime = ? AND is_active = 1 "
                "ORDER BY id DESC LIMIT 1",
                (regime,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_paramset(row)
        finally:
            conn.close()

    def record_performance(
        self,
        param_id: int,
        sharpe: float,
        win_rate: float,
        n: int,
    ) -> None:
        """Record measured performance for a parameter set.

        Args:
            param_id: ID of the parameter set.
            sharpe: Measured Sharpe ratio.
            win_rate: Measured win rate.
            n: Number of trades in the sample.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE param_sets SET sharpe=?, win_rate=?, "
                "sample_size=? WHERE id=?",
                (sharpe, win_rate, n, param_id),
            )
            conn.commit()
        finally:
            conn.close()

    def activate(self, param_id: int, regime: str) -> None:
        """Set a parameter set as active, deactivating others.

        Args:
            param_id: ID to activate.
            regime: Regime to scope deactivation.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE param_sets SET is_active=0 WHERE regime=?",
                (regime,),
            )
            conn.execute(
                "UPDATE param_sets SET is_active=1 WHERE id=?",
                (param_id,),
            )
            conn.commit()
            logger.info(
                "Activated param_set %d for regime %s",
                param_id,
                regime,
            )
        finally:
            conn.close()

    def get_history(
        self, regime: str, limit: int = 20
    ) -> list[ParamSet]:
        """Get recent parameter sets for a regime.

        Args:
            regime: CompositeRegime string value.
            limit: Maximum records to return.

        Returns:
            List of ParamSet ordered by newest first.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM param_sets "
                "WHERE regime = ? ORDER BY id DESC LIMIT ?",
                (regime, limit),
            ).fetchall()
            return [self._row_to_paramset(r) for r in rows]
        finally:
            conn.close()

    @staticmethod
    def _row_to_paramset(row: sqlite3.Row) -> ParamSet:
        return ParamSet(
            id=row["id"],
            regime=row["regime"],
            params=RegimeAdaptedParams(
                stop_loss_mult=row["stop_loss_mult"],
                position_size_mult=row["position_size_mult"],
                entry_threshold=row["entry_threshold"],
                should_trade=bool(row["should_trade"]),
                should_close_on_weakness=bool(
                    row["should_close_on_weakness"]
                ),
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            sharpe=row["sharpe"],
            win_rate=row["win_rate"],
            sample_size=row["sample_size"],
            is_active=bool(row["is_active"]),
        )
