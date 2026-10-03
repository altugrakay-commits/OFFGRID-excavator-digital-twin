"""
Quick smoke-test / simulator for the off-grid digital twin.

Usage:
    python scripts/run_simulator.py

This will:
  1. Build the boom mesh
  2. Compute the stress field (C backend if available)
  3. Render a PyVista screenshot to output/boom_stress.png
  4. Print summary statistics and fatigue damage
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv

from src.core.geometry import create_boom_mesh, boom_geometry_params, default_loads
from src.core.stress_solver import (compute_stress, backend_name)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("Off-Grid Excavator Digital Twin — Simulator")
    print("=" * 60)
    print(f"Stress solver backend: {backend_name()}")

    # ---- Geometry ----
    mesh = create_boom_mesh(
        length=3.0,
        root_width=0.25,
        root_height=0.35,
        tip_width=0.15,
        tip_height=0.22,
        n_segments=60,
        subdivide=1,
    )
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    geom = boom_geometry_params()

    # ---- Loads ----
    loads = default_loads(Fy=2.0e4, Fz=2.0e5, Mx=1.0e4)
    print(f"Loads: Fx={loads[0]:.0f} N, Fy={loads[1]:.0f} N, "
          f"Fz={loads[2]:.0f} N, Mx={loads[3]:.0f} Nm")

    # ---- Stress ----
    points = np.asarray(mesh.points, dtype=np.float64)
    stress_pa = compute_stress(points, loads, geom)
    stress_mpa = stress_pa / 1e6

    mesh.point_data["Von_Mises_Stress_MPa"] = stress_mpa

    print(f"Stress range: {stress_mpa.min():.1f} – {stress_mpa.max():.1f} MPa")
    print(f"Mean stress:  {stress_mpa.mean():.1f} MPa")

    # ---- Fatigue (time-series based, at the critical location) ----
    from src.core.fatigue_demo import report_fatigue

    print("\nFatigue analysis at critical location (peak-stress node):")
    fatigue = report_fatigue(
        peak_stress_pa=float(stress_pa.max()),
        fatigue_strength_pa=900e6,
        n_cycles=5_000,
    )

    # ---- Render ----
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 800])
    plotter.add_mesh(
        mesh,
        scalars="Von_Mises_Stress_MPa",
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.3,
        scalar_bar_args={
            "title": "Von Mises Stress (MPa)",
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.15,
            "width": 0.08,
            "height": 0.7,
            "fmt": "%.0f",
        },
    )
    plotter.add_title(
        "Off-Grid Excavator Boom — Digital Twin Stress Field",
        font_size=14,
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.2)

    out_path = OUTPUT_DIR / "boom_stress.png"
    plotter.screenshot(str(out_path))
    plotter.close()

    print(f"Screenshot saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()