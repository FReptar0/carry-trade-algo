"""Model registry for versioned ML model storage and lifecycle.

Tracks model versions with their performance metrics and manages
the promotion pipeline: training -> shadow -> active -> retired.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A versioned model with performance tracking.

    Attributes:
        version: Auto-incremented version number.
        model_type: Model identifier (e.g., "bandit_v1", "rl_ppo_v1").
        created_at: When the model was created.
        train_sharpe: Sharpe ratio on training data.
        val_sharpe: Sharpe ratio on validation data.
        shadow_sharpe: Sharpe ratio during shadow evaluation.
        status: One of "training", "shadow", "active", "retired".
        artifact_path: Path to the serialized model file.
    """

    version: int
    model_type: str
    created_at: datetime
    train_sharpe: float
    val_sharpe: float
    shadow_sharpe: Optional[float]
    status: str
    artifact_path: str


class ModelRegistry:
    """Version-controlled model storage with performance tracking.

    Args:
        db_path: Path to the SQLite database.
        artifacts_dir: Directory for model artifacts.

    Example:
        >>> registry = ModelRegistry("data/trading.db")
        >>> vid = registry.register("bandit_v1", "data/models/b1.json",
        ...     {"train_sharpe": 1.5, "val_sharpe": 1.2})
        >>> registry.promote(vid, "shadow")
        >>> active = registry.get_active("bandit_v1")
    """

    def __init__(
        self,
        db_path: str = "data/trading.db",
        artifacts_dir: str = "data/models",
    ) -> None:
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS model_versions (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    train_sharpe REAL NOT NULL,
                    val_sharpe REAL NOT NULL,
                    shadow_sharpe REAL,
                    status TEXT NOT NULL DEFAULT 'training',
                    artifact_path TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def register(
        self,
        model_type: str,
        artifact_path: str,
        metrics: dict,
    ) -> int:
        """Register a new model version.

        Args:
            model_type: Model identifier.
            artifact_path: Path to serialized model.
            metrics: Dict with train_sharpe and val_sharpe.

        Returns:
            Version number of the new model.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                INSERT INTO model_versions
                (model_type, created_at, train_sharpe, val_sharpe,
                 status, artifact_path)
                VALUES (?, ?, ?, ?, 'training', ?)
                """,
                (
                    model_type,
                    datetime.now().isoformat(),
                    metrics.get("train_sharpe", 0.0),
                    metrics.get("val_sharpe", 0.0),
                    artifact_path,
                ),
            )
            conn.commit()
            version = cursor.lastrowid
            logger.info(
                "Registered model %s v%d (train_sharpe=%.3f)",
                model_type,
                version,
                metrics.get("train_sharpe", 0.0),
            )
            return version
        finally:
            conn.close()

    def promote(self, version: int, to_status: str) -> None:
        """Change a model's status.

        If promoting to 'active', retires the current active model.

        Args:
            version: Model version to promote.
            to_status: New status.
        """
        conn = self._get_conn()
        try:
            if to_status == "active":
                # Get model type for this version
                row = conn.execute(
                    "SELECT model_type FROM model_versions "
                    "WHERE version = ?",
                    (version,),
                ).fetchone()
                if row:
                    # Retire current active
                    conn.execute(
                        "UPDATE model_versions SET status='retired' "
                        "WHERE model_type=? AND status='active'",
                        (row["model_type"],),
                    )

            conn.execute(
                "UPDATE model_versions SET status=? WHERE version=?",
                (to_status, version),
            )
            conn.commit()
            logger.info(
                "Model v%d promoted to %s", version, to_status
            )
        finally:
            conn.close()

    def get_active(
        self, model_type: str
    ) -> Optional[ModelVersion]:
        """Get the currently active model of a given type.

        Args:
            model_type: Model identifier.

        Returns:
            Active ModelVersion, or None.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM model_versions "
                "WHERE model_type=? AND status='active' "
                "ORDER BY version DESC LIMIT 1",
                (model_type,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_model(row)
        finally:
            conn.close()

    def get_shadow(
        self, model_type: str
    ) -> Optional[ModelVersion]:
        """Get the current shadow model of a given type.

        Args:
            model_type: Model identifier.

        Returns:
            Shadow ModelVersion, or None.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM model_versions "
                "WHERE model_type=? AND status='shadow' "
                "ORDER BY version DESC LIMIT 1",
                (model_type,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_model(row)
        finally:
            conn.close()

    def get_history(
        self, model_type: str
    ) -> list[ModelVersion]:
        """Get all versions of a model type.

        Args:
            model_type: Model identifier.

        Returns:
            List of ModelVersion ordered by version descending.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM model_versions "
                "WHERE model_type=? ORDER BY version DESC",
                (model_type,),
            ).fetchall()
            return [self._row_to_model(r) for r in rows]
        finally:
            conn.close()

    def update_shadow_sharpe(
        self, version: int, shadow_sharpe: float
    ) -> None:
        """Update the shadow Sharpe for a model.

        Args:
            version: Model version.
            shadow_sharpe: Measured Sharpe during shadow evaluation.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE model_versions SET shadow_sharpe=? "
                "WHERE version=?",
                (shadow_sharpe, version),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            version=row["version"],
            model_type=row["model_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            train_sharpe=row["train_sharpe"],
            val_sharpe=row["val_sharpe"],
            shadow_sharpe=row["shadow_sharpe"],
            status=row["status"],
            artifact_path=row["artifact_path"],
        )
