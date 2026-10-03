"""
Local-first SQLite storage for the off-grid digital twin.

Every write goes local first. The `synced` flags on each row are the
cloud-sync queue — the sync module (Phase 3) drains them when
connectivity is restored. The dashboard reads from here, so nothing
in the UI depends on the network being available.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS telemetry_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    t             REAL    NOT NULL,
    regime        TEXT    NOT NULL,
    pressure_pa   REAL    NOT NULL,
    boom_angle    REAL    NOT NULL,
    vibration_g   REAL    NOT NULL,
    oil_temp_c    REAL    NOT NULL,
    synced        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stress_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    t             REAL    NOT NULL,
    load_vector   TEXT    NOT NULL,
    sigma_max_pa  REAL    NOT NULL,
    sigma_mean_pa REAL    NOT NULL,
    synced        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fatigue_state (
    id       INTEGER PRIMARY KEY CHECK (id = 1),
    t_last   REAL    NOT NULL,
    damage   REAL    NOT NULL,
    n_cycles INTEGER NOT NULL,
    status   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_t      ON telemetry_samples(t);
CREATE INDEX IF NOT EXISTS idx_stress_t         ON stress_snapshots(t);
CREATE INDEX IF NOT EXISTS idx_telemetry_synced ON telemetry_samples(synced);
CREATE INDEX IF NOT EXISTS idx_stress_synced    ON stress_snapshots(synced);
"""

class LocalDB:
    """Thin SQLite wrapper with offline-first semantics."""

    def __init__(self, path: Path | str = "output/offgrid.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --------------------------------------------------------------
    # Connection helpers
    # --------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor]:
        conn = self._connect()
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.executescript(SCHEMA)
    
    # --------------------------------------------------------------
    # Writes
    # --------------------------------------------------------------

    def insert_telemetry(self, sample: dict) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO telemetry_samples
                    (t, regime, pressure_pa, boom_angle, vibration_g, oil_temp_c)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample["t"],
                    sample["regime"],
                    sample["hydraulic_pressure_pa"],
                    sample["boom_angle_rad"],
                    sample["vibration_g"],
                    sample["oil_temp_c"]
                )
            )
            return int(cur.lastrowid) # type: ignore

    def insert_stress_snapshot(self, t: float, load_vector: list[float], sigma_max_pa: float, sigma_mean_pa: float) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO stress_snapshots
                    (t, load_vector, sigma_max_pa, sigma_mean_pa)
                VALUES (?, ?, ?, ?)
                """,
                (t, json.dumps(load_vector), sigma_max_pa, sigma_mean_pa)
            )
            return int(cur.lastrowid) # type: ignore

    def upsert_fatigue_state(self, t_last: float, damage: float, n_cycles: int, status: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO fatigue_state (id, t_last, damage, n_cycles, status)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    t_last   = excluded.t_last,
                    damage   = excluded.damage,
                    n_cycles = excluded.n_cycles,
                    status   = excluded.status
                """,
                (t_last, damage, n_cycles, status)
            )
    # --------------------------------------------------------------
    # Reads
    # --------------------------------------------------------------

    def latest_telemetry(self, limit: int = 100) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM telemetry_samples ORDER BY t DESC LIMIT ?",
                (limit,)
            )
            rows = [dict(r) for r in cur.fetchall()]
        return list(reversed(rows))

    def latest_stress(self, limit: int = 100) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM stress_snapshots ORDER BY t DESC LIMIT ?",
                (limit,)
            )
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["load_vector"] = json.loads(r["load_vector"])
        return list(reversed(rows))

    def fatigue_state(self) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM fatigue_state WHERE id = 1")
            row = cur.fetchone()
        return dict(row) if row else None
    
    # --------------------------------------------------------------
    # Sync support (used by src/edge/sync.py)
    # --------------------------------------------------------------

    def pending_telemetry(self, limit: int = 1000) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM telemetry_samples WHERE synced = 0 "
                "ORDER BY t ASC LIMIT ?",
                (limit,)
            )
            return [dict(r) for r in cur.fetchall()]

    def pending_stress(self, limit: int = 1000) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM stress_snapshots WHERE synced = 0 "
                "ORDER BY t ASC LIMIT ?",
                (limit,)
            )
            return [dict(r) for r in cur.fetchall()]

    def mark_synced(self, table: str, ids: list[int]) -> None:
        if not ids:
            return
        if table not in {"telemetry_samples", "stress_snapshots"}:
            raise ValueError(f"unknown table: {table}")
        placeholders = ",".join("?" * len(ids))
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE {table} SET synced = 1 WHERE id IN ({placeholders})",
                ids
            )

    def stats(self) -> dict:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM telemetry_samples")
            n_tel = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM stress_snapshots")
            n_str = cur.fetchone()["n"]
            cur.execute(
                "SELECT COUNT(*) AS n FROM telemetry_samples WHERE synced = 0"
            )
            n_pend_tel = cur.fetchone()["n"]
            cur.execute(
                "SELECT COUNT(*) AS n FROM stress_snapshots WHERE synced = 0"
            )
            n_pend_str = cur.fetchone()["n"]
        
        return {
            "telemetry_total": n_tel,
            "stress_total": n_str,
            "telemetry_pending_sync": n_pend_tel,
            "stress_pending_sync": n_pend_str,
        }