"""
Smoke test for Phase 2: telemetry → stress → fatigue → SQLite.

Run 200 simulated seconds at 10 Hz = 2,000 samples.
Writes everything to output/offgrid.db, then prints DB stats.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.telemetry import TelemetrySimulator
from src.core.geometry import create_boom_mesh, boom_geometry_params
from src.core.stress_solver import compute_stress, backend_name
from src.core.fatigue_demo import simulate_load_history
from src.core.stress_solver import compute_fatigue_damage
from src.edge.local_db import LocalDB


N_SAMPLES = 2000   # 200 seconds at 10 Hz


def main() -> None:
    print("=" * 60)
    print("Phase 2 — Telemetry + SQLite Demo")
    print("=" * 60)
    print(f"Backend: {backend_name()}")

    mesh = create_boom_mesh(n_segments=40, subdivide=1)
    points = np.asarray(mesh.points, dtype=np.float64)
    geom = boom_geometry_params()

    db_path = Path("output/offgrid.db")
    db_path.unlink(missing_ok=True)   # clean slate for the demo
    db = LocalDB(db_path)

    sim = TelemetrySimulator()

    peak_stresses: list[float] = []
    n_cycles_total = 0
    damage_total = 0.0

    for i in range(N_SAMPLES):
        s = sim.sample()
        db.insert_telemetry(asdict(s))

        loads = sim.loads_from_sample(s)
        stress = compute_stress(points, loads, geom)
        sigma_max = float(stress.max())
        sigma_mean = float(stress.mean())

        db.insert_stress_snapshot(
            t=s.t, load_vector=loads.tolist(),
            sigma_max_pa=sigma_max, sigma_mean_pa=sigma_mean,
        )

        # Accumulate fatigue on every cycle above a meaningful threshold.
        if sigma_max > 150e6:      # 150 MPa — below this, Basquin life is
                                   # astronomically long and contributes nothing
            hist = simulate_load_history(sigma_max, n_cycles=10)
            d = compute_fatigue_damage(hist, fatigue_strength=900e6)
            damage_total += d
            n_cycles_total += 10

        peak_stresses.append(sigma_max)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1:4d}/{N_SAMPLES}]  "
                  f"regime={s.regime:8s}  "
                  f"p={s.hydraulic_pressure_pa/1e6:5.1f} MPa  "
                  f"σ_max={sigma_max/1e6:6.1f} MPa")

    # Final fatigue state
    status = "FAILURE" if damage_total >= 1.0 else "OK"
    db.upsert_fatigue_state(
        t_last=sim.t0 + sim.t,
        damage=damage_total,
        n_cycles=n_cycles_total,
        status=status,
    )

    print("\nFinal state:")
    print(f"  Peak stress across run:   {max(peak_stresses)/1e6:.1f} MPa")
    print(f"  Total (high-load)cycles:  {n_cycles_total}")
    print(f"  Cumulative damage D:      {damage_total:.4e}")
    print(f"  Remaining life:           {(1-damage_total)*100:.3f}%")
    print(f"  Status:                   {status}")

    print("\nDB stats:")
    for k, v in db.stats().items():
        print(f"  {k:26s} {v}")

    print("\nMaterial-profile comparison (2000 samples each):")
    print(f"  {'profile':8s}  {'peak MPa':>9s}  {'p95 MPa':>8s}  "
          f"{'damage':>10s}  {'remaining %':>12s}")
    for mat in ("soil", "mixed", "rock"):
        sim2 = TelemetrySimulator(seed=1234, material=mat)
        peaks = []
        dmg = 0.0
        for _ in range(2000):
            s2 = sim2.sample()
            loads2 = sim2.loads_from_sample(s2)
            st2 = compute_stress(points, loads2, geom)
            smax = float(st2.max())
            peaks.append(smax)
            if smax > 150e6:
                h = simulate_load_history(smax, n_cycles=10)
                dmg += compute_fatigue_damage(h, fatigue_strength=900e6)
        peaks_arr = np.asarray(peaks) / 1e6
        print(f"  {mat:8s}  {peaks_arr.max():9.1f}  "
              f"{np.percentile(peaks_arr, 95):8.1f}  "
              f"{dmg:10.3e}  {(1-dmg)*100:12.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()