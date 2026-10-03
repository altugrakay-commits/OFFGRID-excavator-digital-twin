"""
Demonstrate the offline -> online transition.

Runs 500 telemetry samples in "offline" mode (nothing synced), then flips
the switch and drains the local queue via the FileTransport.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.telemetry import TelemetrySimulator
from src.core.geometry import create_boom_mesh, boom_geometry_params
from src.core.stress_solver import compute_stress
from src.edge.local_db import LocalDB
from src.edge.sync import FileTransport, SyncEngine


def main() -> None:
    db_path = Path("output/sync_demo.db")
    db_path.unlink(missing_ok=True)
    db = LocalDB(db_path)

    mesh = create_boom_mesh(n_segments=30, subdivide=1)
    points = np.asarray(mesh.points, dtype=np.float64)
    geom = boom_geometry_params()
    sim = TelemetrySimulator()

    print("=" * 60)
    print("Phase 3 — Offline -> Online Sync Demo")
    print("=" * 60)

    # ---- Phase A: OFFLINE ----
    print("\n[OFFLINE] Generating 500 samples, no sync...")
    for _ in range(500):
        s = sim.sample()
        db.insert_telemetry(asdict(s))
        loads = sim.loads_from_sample(s)
        stress = compute_stress(points, loads, geom)
        db.insert_stress_snapshot(
            t=s.t,
            load_vector=loads.tolist(),
            sigma_max_pa=float(stress.max()),
            sigma_mean_pa=float(stress.mean()),
        )

    stats = db.stats()
    print(f"  telemetry_total          {stats['telemetry_total']}")
    print(f"  stress_total             {stats['stress_total']}")
    print(f"  telemetry_pending_sync   {stats['telemetry_pending_sync']}")
    print(f"  stress_pending_sync      {stats['stress_pending_sync']}")

    # ---- Phase B: ONLINE ----
    print("\n[ONLINE] Signal restored. Draining queue...")
    engine = SyncEngine(db, FileTransport("output/cloud_spool"))
    result = engine.sync_once()

    print(f"  telemetry sent  {result.n_telemetry}")
    print(f"  stress sent     {result.n_stress}")
    print(f"  elapsed         {result.elapsed_s * 1000:.1f} ms")
    print(f"  ok              {result.ok}")

    stats = db.stats()
    print(f"\n  telemetry_pending_sync   {stats['telemetry_pending_sync']}")
    print(f"  stress_pending_sync      {stats['stress_pending_sync']}")

    spool = Path("output/cloud_spool")
    files = sorted(spool.glob("*.jsonl"))
    print(f"\nSpool directory: {spool} ({len(files)} files)")
    for p in files[-4:]:
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")

    print("=" * 60)


if __name__ == "__main__":
    main()