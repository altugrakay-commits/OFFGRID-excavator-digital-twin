"""
Offline-first cloud sync for the digital twin.

The LocalDB accumulates rows with `synced = 0`. This module drains
them when connectivity is available, marking each row `synced = 1`
on success. Failures leave the row pending so the next attempt
retries it — an at-least-once delivery model.

A real deployment would POST batches to an HTTPS endpoint. For the
hackathon demo we use a pluggable `CloudTransport` interface with two
implementations:

  * FileTransport   — writes JSONL batches to output/cloud_spool/
  * MemoryTransport — keeps counters in RAM (useful for tests)

Swapping to a real cloud is a ~20-line subclass of CloudTransport.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.edge.local_db import LocalDB


# ------------------------------------------------------------------ #
# Result type
# ------------------------------------------------------------------ #

@dataclass
class SyncResult:
    n_telemetry: int
    n_stress: int
    elapsed_s: float
    ok: bool
    error: Optional[str] = None


# ------------------------------------------------------------------ #
# Cloud transport abstraction
# ------------------------------------------------------------------ #

class CloudTransport(ABC):
    """Interface for cloud endpoints. Subclass to hit real APIs."""

    @abstractmethod
    def send_telemetry(self, batch: list[dict]) -> None: ...

    @abstractmethod
    def send_stress(self, batch: list[dict]) -> None: ...


class FileTransport(CloudTransport):
    """
    Writes each batch to output/cloud_spool/<prefix>_<ts>_<n>.jsonl.

    Useful for offline demos: open the spool directory and see exactly
    what would have been sent to the cloud.
    """

    def __init__(self, spool_dir: Path | str = "output/cloud_spool") -> None:
        self.spool = Path(spool_dir)
        self.spool.mkdir(parents=True, exist_ok=True)

    def _write(self, prefix: str, batch: list[dict]) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.spool / f"{prefix}_{ts}_{len(batch):05d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in batch:
                f.write(json.dumps(row, default=str) + "\n")

    def send_telemetry(self, batch: list[dict]) -> None:
        self._write("telemetry", batch)

    def send_stress(self, batch: list[dict]) -> None:
        self._write("stress", batch)


class MemoryTransport(CloudTransport):
    """In-memory transport, useful for tests and dashboard sessions."""

    def __init__(self) -> None:
        self.sent_telemetry: int = 0
        self.sent_stress: int = 0
        self.batches: list[tuple[str, int]] = []

    def send_telemetry(self, batch: list[dict]) -> None:
        self.sent_telemetry += len(batch)
        self.batches.append(("telemetry", len(batch)))

    def send_stress(self, batch: list[dict]) -> None:
        self.sent_stress += len(batch)
        self.batches.append(("stress", len(batch)))

    def reset(self) -> None:
        self.sent_telemetry = 0
        self.sent_stress = 0
        self.batches.clear()


# ------------------------------------------------------------------ #
# Sync engine
# ------------------------------------------------------------------ #

@dataclass
class SyncEngine:
    """
    Drains pending rows from LocalDB and ships them via a CloudTransport.

    Usage:
        engine = SyncEngine(db, FileTransport())
        result = engine.sync_once()
    """

    db: LocalDB
    transport: CloudTransport
    batch_size: int = 500
    max_batches: int = 10                     # cap one sync pass
    simulate_failure_rate: float = 0.0        # for testing
    _rng_state: int = field(default=0xC0FFEE)

    def _should_fail(self) -> bool:
        """Deterministic pseudo-random failure injection for testing."""
        if self.simulate_failure_rate <= 0.0:
            return False
        self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
        return (self._rng_state / 0x7FFFFFFF) < self.simulate_failure_rate

    def sync_once(self) -> SyncResult:
        t0 = time.perf_counter()
        n_tel = 0
        n_str = 0

        try:
            for _ in range(self.max_batches):
                tel_batch = self.db.pending_telemetry(self.batch_size)
                str_batch = self.db.pending_stress(self.batch_size)
                if not tel_batch and not str_batch:
                    break

                if self._should_fail():
                    raise RuntimeError("simulated network failure")

                if tel_batch:
                    self.transport.send_telemetry(tel_batch)
                    self.db.mark_synced(
                        "telemetry_samples",
                        [r["id"] for r in tel_batch],
                    )
                    n_tel += len(tel_batch)

                if str_batch:
                    self.transport.send_stress(str_batch)
                    self.db.mark_synced(
                        "stress_snapshots",
                        [r["id"] for r in str_batch],
                    )
                    n_str += len(str_batch)

            elapsed = time.perf_counter() - t0
            return SyncResult(n_tel, n_str, elapsed, ok=True)

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return SyncResult(n_tel, n_str, elapsed, ok=False, error=str(exc))