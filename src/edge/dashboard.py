"""
Streamlit dashboard for the off-grid excavator digital twin.

Demonstrates the complete offline-first workflow:
  * Live 3D stress field on the boom (PyVista off-screen render)
  * Rolling hydraulic pressure trace
  * Fatigue bar and status indicator
  * Material profile + machine size selectors
  * Offline / Online toggle that drives the sync engine

Run with:
    streamlit run src/edge/dashboard.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st

from src.core.telemetry import TelemetrySimulator
from src.core.geometry import create_boom_mesh, boom_geometry_params
from src.core.stress_solver import (
    compute_stress, compute_fatigue_damage, backend_name,
)
from src.core.fatigue_demo import simulate_load_history
from src.edge.local_db import LocalDB
from src.edge.sync import FileTransport, SyncEngine


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

MACHINE_PRESETS = {
    "Mini (0.006 m²)":      0.006,
    "Compact (0.008 m²)":   0.008,
    "Mid-size (0.012 m²)":  0.012,
    "Heavy (0.020 m²)":     0.020
}

MATERIAL_CHOICES = ["soil", "mixed", "rock"]

DB_PATH = Path("output/dashboard.db")
SPOOL_DIR = Path("output/cloud_spool")

SAMPLES_PER_TICK = 5        # simulation steps per Streamlit rerun
RENDER_EVERY_N_TICKS = 3    # refresh 3D view every N ticks
CHART_POINTS = 400          # ~40 s of telemetry at 10 Hz

# ------------------------------------------------------------------ #
# Cached resources
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner=False)
def _load_boom_mesh():
    mesh = create_boom_mesh(n_segments=40, subdivide=1)
    points = np.asarray(mesh.points, dtype=np.float64)
    geom = boom_geometry_params()
    return mesh, points, geom

# ------------------------------------------------------------------ #
# Rendering
# ------------------------------------------------------------------ #

def render_stress_image(mesh: pv.PolyData, stress_mpa: np.ndarray) -> np.ndarray:
    """Off-screen render. Returns an RGB numpy for st.image()."""
    m = mesh.copy()
    m.point_data["Von Mises (MPa)"] = stress_mpa

    plotter = pv.Plotter(off_screen=True, window_size=[900, 500])
    plotter.add_mesh(
        m,
        scalars="Von Mises (MPa)",
        cmap="plasma",
        show_edges=True,
        edge_color="black",
        line_width=0.2,
        clim=[0.0, 350.0],
        scalar_bar_args={
            "title": "Von Mises (MPa)",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.12,
            "width": 0.07,
            "height": 0.75,
            "fmt": "%.0f"
        }
    )

    plotter.add_title("Boom - Live Stress Field", font_size=12)
    plotter.view_isometric()  # type: ignore
    plotter.camera.zoom(1.15)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img  # type: ignore

# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #

DEFAULTS = {
    "running": False,
    "tick": 0,
    "damage": 0.0,
    "n_cycles": 0,
    "last_sigma_max_mpa": None,
    "last_stress_pa": None,
    "last_png": None,
    "online": False,
    "machine_label": "Compact (0.008 m²)",
    "material": "mixed",
    "session_peak_mpa": 0.0,
    "synced_telemetry": 0,
    "synced_stress": 0,
    "_sim_key": None
}

def _init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "db" not in st.session_state:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        st.session_state.db = LocalDB(DB_PATH)

def _reset_state() -> None:
    """Deletes DB files and clears simulation state. Keeps the mesh cache."""
    for suffix in ("", "-wal", "-shm"):
        Path(str(DB_PATH) + suffix).unlink(missing_ok=True)

    st.session_state.db = LocalDB(DB_PATH)
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ------------------------------------------------------------------ #
# Simulator management
# ------------------------------------------------------------------ #

def _get_simulator() -> TelemetrySimulator:
    """
    Return a simulator bound to the current machine + material config.
    Recreate only when the config changes, so regime state and RNG
    stream persist across Streamlit reruns.
    """
    key = (st.session_state.machine_label, st.session_state.material)
    if st.session_state["_sim_key"] != key:
        st.session_state.sim = TelemetrySimulator(
            cylinder_area_m2=MACHINE_PRESETS[st.session_state.machine_label],
            material=st.session_state.material,
            seed=1234
        )
        st.session_state["_sim_key"] = key
    return st.session_state.sim

# ------------------------------------------------------------------ #
# Tick
# ------------------------------------------------------------------ #

def _tick(mesh, points, geom) -> None:
    """Advance the simulation by SAMPLES_PER_TICK and persist results."""
    db: LocalDB = st.session_state.db
    sim = _get_simulator()

    for _ in range(SAMPLES_PER_TICK):
        s = sim.sample()
        db.insert_telemetry(asdict(s))

        loads = sim.loads_from_sample(s)
        stress_pa = compute_stress(points, loads, geom)
        sigma_max = float(stress_pa.max())
        sigma_mean = float(stress_pa.mean())

        db.insert_stress_snapshot(
            t=s.t,
            load_vector=loads.tolist(),
            sigma_max_pa=sigma_max,
            sigma_mean_pa=sigma_mean
        )

        if sigma_max > 150e6:
            hist = simulate_load_history(sigma_max, n_cycles=10)
            st.session_state.damage += compute_fatigue_damage(
                hist, fatigue_strength=900e6
            )
            st.session_state.n_cycles += 10

        st.session_state.last_sigma_max_mpa = sigma_max / 1e6
        st.session_state.session_peak_mpa = max(
            st.session_state.session_peak_mpa, sigma_max / 1e6
        )
        st.session_state.last_stress_pa = stress_pa

    st.session_state.tick += 1

    db.upsert_fatigue_state(
        t_last=s.t,
        damage=st.session_state.damage,
        n_cycles=st.session_state.n_cycles,
        status="FAILURE" if st.session_state.damage >= 1.0 else "OK"
    )

    if st.session_state.tick % RENDER_EVERY_N_TICKS == 0:
        st.session_state.last_png = render_stress_image(
            mesh, st.session_state.last_stress_pa / 1e6
        )

    if st.session_state.online:
        engine = SyncEngine(db, FileTransport(SPOOL_DIR))
        r = engine.sync_once()
        st.session_state.synced_telemetry += r.n_telemetry
        st.session_state.synced_stress += r.n_stress

# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #

def main() -> None:
    st.set_page_config(
        page_title="Off-Grid Excavator Digital Twin",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    _init_state()

    db: LocalDB = st.session_state.db
    mesh, points, geom = _load_boom_mesh()

    # ---------------- Sidebar ----------------
    st.sidebar.title("⚙️ Twin Controls")
    st.sidebar.caption(f"Solver backend: **{backend_name()}**")

    st.session_state.machine_label = st.sidebar.selectbox(
        "Machine size",
        list(MACHINE_PRESETS),
        index=list(MACHINE_PRESETS).index(st.session_state.machine_label)
    )
    st.session_state.material = st.sidebar.selectbox(
        "Material profile",
        MATERIAL_CHOICES,
        index=MATERIAL_CHOICES.index(st.session_state.material),
        help=(
            "Geology the machine is working in. "
            "'rock' = hard igneous formations (abuse mode on), "
            "'soil' = soft ground or dredging."
        )
    )

    st.sidebar.divider()

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button(
        "⏸️ Pause" if st.session_state.running else "▶️ Start",
        use_container_width=True
    ):
        st.session_state.running = not st.session_state.running

    if col_b.button("🔄️ Reset", use_container_width=True):
        _reset_state()
        st.rerun()

    st.sidebar.divider()

    st.session_state.online = st.sidebar.toggle(
        "🌐 Cloud link active",
        value=st.session_state.online,
        help="Toggle off to simulate an off-grid deployment."
    )

    if st.session_state.online:
        st.sidebar.success("Link active - queue draining.")
    else:
        st.sidebar.warning("Off-grid - data buffered locally.")

    if st.sidebar.button("⬆️ Force sync now", use_container_width=True):
        engine = SyncEngine(db, FileTransport(SPOOL_DIR))
        r = engine.sync_once()
        st.session_state.synced_telemetry += r.n_telemetry
        st.session_state.synced_stress += r.n_stress
        st.sidebar.info(
            f"Sent {r.n_telemetry} tel / {r.n_stress} stress in "
            f"{r.elapsed_s * 1000:.0f} ms"
        )

    st.sidebar.divider()
    stats = db.stats()
    st.sidebar.metric("Telemetry rows", f"{stats['telemetry_total']:,}")
    st.sidebar.metric("Stress rows", f"{stats['stress_total']:,}")
    pending = (stats["telemetry_pending_sync"] + stats["stress_pending_sync"])
    st.sidebar.metric("Pending sync", f"{pending:,}")
    st.sidebar.caption(
        f"Synced this session: {st.session_state.synced_telemetry:,} tel, "
        f"{st.session_state.synced_stress:,} stress"
    )

    # ---------------- Header ----------------
    st.title("Off-Grid Excavator Digital Twin")
    st.caption(
        "Physics-informed structural monitoring for remote construction equipment. "
        "All computation and storage run locally — no cloud dependency."
    )

    # ---------------- Advance simulation ----------------
    if st.session_state.running:
        _tick(mesh, points, geom)

    # ---------------- Main layout ----------------
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Live Stress Field")
        if st.session_state.last_png is not None:
            st.image(st.session_state.last_png, use_container_width=True)
        else:
            st.info("Press **Start** to begin the simulation.")

    with right:
        st.subheader("Machine State")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Last σ",
            f"{st.session_state.last_sigma_max_mpa:.0f} MPa"
            if st.session_state.last_sigma_max_mpa is not None else "—",
        )
        m2.metric("Session peak", f"{st.session_state.session_peak_mpa:.0f} MPa")
        m3.metric("Cycles", f"{st.session_state.n_cycles:,}")

        st.subheader("Fatigue Life")
        d = st.session_state.damage
        remaining = max(0.0, 1.0 - d)
        st.progress(min(remaining, 1.0),
                    text=f"{remaining * 100:.5f}% remaining")

        if d < 0.1:
            st.success(f"Status: OK   |   D = {d:.3e}")
        elif d < 0.5:
            st.warning(f"Status: Elevated   |   D = {d:.3e}")
        else:
            st.error(f"Status: Critical   |   D = {d:.3e}")

        last = st.session_state.last_sigma_max_mpa
        if last is not None:
            if last > 320:
                st.error(f"⚠ Last σ {last:.0f} MPa exceeds yield (~300 MPa)")
            elif last > 260:
                st.warning(f"Last σ {last:.0f} MPa — elevated loading")
            else:
                st.success(f"Last σ {last:.0f} MPa — within design envelope")

    # ---------------- Rolling chart ----------------
    st.subheader("Hydraulic Pressure — rolling window")
    tel = db.latest_telemetry(CHART_POINTS)
    if tel:
        df = pd.DataFrame(tel)
        df["t_rel"] = df["t"] - df["t"].iloc[0]
        chart_df = df.set_index("t_rel")[["pressure_pa"]].copy()
        chart_df["pressure_pa"] = chart_df["pressure_pa"] / 1e6
        chart_df.columns = ["Pressure (MPa)"]
        st.line_chart(chart_df, height=220, use_container_width=True)
    else:
        st.info("No telemetry yet.")

    # ---------------- Footer ----------------
    st.divider()
    st.caption(
        f"Ticks: {st.session_state.tick}  |  "
        f"Material: **{st.session_state.material}**  |  "
        f"Machine: **{st.session_state.machine_label}**  |  "
        f"DB: `{DB_PATH}`  |  Spool: `{SPOOL_DIR}`"
    )

    # ---------------- Auto-rerun while running ----------------
    if st.session_state.running:
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()