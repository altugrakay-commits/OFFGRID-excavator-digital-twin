# Off-Grid Excavator Digital Twin

> **Physics-informed structural health monitoring for remote construction equipment — designed to run entirely offline on a Raspberry Pi in the machine's cab.**

[![Backend](https://img.shields.io/badge/solver-C%20%2B%20NumPy-blue)](#)
[![Edge](https://img.shields.io/badge/edge--ready-Raspberry%20Pi-green)](#)
[![Offline](https://img.shields.io/badge/network-optional-success)](#)

---

## The Problem

Excavators work in places with no reliable internet: hard-rock mines, disaster-recovery sites, forest clear-cuts, deep pits. Two things follow from that:

1. **Cloud-based predictive maintenance is unavailable.** You cannot upload telemetry you cannot transmit.
2. **Operators cannot see cumulative damage.** A boom arm can be 60% through its fatigue life and still look fine on the outside — until it cracks.

The result is unplanned downtime at 3 AM, 40 km from the nearest road, with a machine that costs €2,000/hour to keep idle.

## The Solution

A **physics-informed digital twin** that runs entirely on an edge device in the cab. It:

- Reads hydraulic pressure, boom angle, and vibration from the machine's existing control system
- Computes real-time Von Mises stress across a CAD-derived boom mesh
- Accumulates fatigue damage using Basquin's law with Goodman mean-stress correction
- Warns the operator locally when stress exceeds design limits
- Buffers everything in a local SQLite database
- Syncs to the cloud opportunistically when a link becomes available

No new sensors. No cloud dependency for core functionality. Same code runs on a laptop, a Raspberry Pi 4, or an NVIDIA Jetson.

## Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│  SENSOR LAYER                                                     │
│  hydraulic pressure │ boom angle │ vibration │ oil temp           │
└────────────────────────┬──────────────────────────────────────────┘
                         │  10 Hz
                         ▼
┌───────────────────────────────────────────────────────────────────┐
│  PHYSICS CORE  (C shared library + NumPy fallback)                │
│                                                                   │
│  geometry.py ──▶ stress_solver (libstress) ──▶ fatigue (Miner)   │
│   tapered boom     Von Mises σ_vm(x,y,z)        Basquin + Goodman │
└────────────────────────┬──────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────────┐
│  LOCAL-FIRST DATA LAYER  (SQLite, WAL mode)                       │
│  telemetry_samples │ stress_snapshots │ fatigue_state             │
│  all rows carry a `synced` flag — the offline queue               │
└────────────────────────┬──────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Streamlit   │ │  Sync engine │ │  Spool files │
│  dashboard   │ │  (drains     │ │  (JSONL,     │
│  (in-cab UI) │ │   synced=0)  │ │   auditable) │
└──────────────┘ └──────────────┘ └──────────────┘
                         │
                         │  (when online)
                         ▼
                  ┌──────────────┐
                  │  Cloud API   │
                  │  (fleet      │
                  │   analytics) │
                  └──────────────┘
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compile the C solver (Windows/Linux/macOS)
python build.py

# 3. Run the physics smoke test
python scripts/run_simulator.py

# 4. Run the telemetry + SQLite demo
python scripts/run_telemetry_demo.py

# 5. Launch the dashboard
streamlit run src/edge/dashboard.py
```

The dashboard opens at http://localhost:8501.

## Technical Highlights

### C + Python hybrid solver
The numerical core `c_src/stress_solver.c` computes Von Mises stress across the boom mesh in a tight C loop. Python handles orchestration, geometry generation, and UI. The wrapper `src/core/stress_solver.py` uses `ctypes` with `c_void_p` pointers for cross-platform reliability, and falls back to a pure-NumPy implementation if the shared library is unavailable.

### Physics that respects the material
 - **Basquin's S-N curve** with exponent 10 (standard for structural steel per Eurocode 3)
 - **Goodman mean-stress correction** accounts for the non-zero operating mean
 - **Endurance-limit behavior** emerges naturally: cycles below ~150 MPa contribute essentially zero damage
 - **Yield-surface warning** in the UI when peak stress exceeds ~300 MPa

### Offline-first by construction
The `synced` flag on every telemetry and stress row is the cloud queue. Nothing in the UI reads from the network. The sync engine drains batches opportunistically; failures leave rows pending for the next attempt (at-least-once delivery).

### Material profiles
Three built-in geology profiles tune the load model:

`soil` — dredging, soft ground, low pressure

`mixed` — general excavation

`rock` — hard igneous formations, sustained high-pressure digging

Switching profiles at runtime rescales the entire simulation.

## Repository Layout

```
DSL_OFFGRID/
├── c_src/
│   ├── stress_solver.c        # Von Mises + fatigue in C
│   └── stress_solver.h
├── src/
│   ├── core/
│   │   ├── geometry.py        # Tapered boom mesh
│   │   ├── stress_solver.py   # ctypes wrapper + NumPy fallback
│   │   ├── fatigue_demo.py    # Load-history simulator
│   │   └── telemetry.py       # Sensor stream + material profiles
│   └── edge/
│       ├── local_db.py        # SQLite, offline-first
│       ├── sync.py            # Cloud transport + engine
│       └── dashboard.py       # Streamlit UI
├── scripts/
│   ├── run_simulator.py       # Physics smoke test
│   ├── run_telemetry_demo.py  # Data-layer smoke test
│   └── run_sync_demo.py       # Offline→online demo
├── tests/                     # pytest suite
├── data/                      # Sample meshes and reference data
├── Videos/                    # Demo recordings
├── build.py                   # Cross-platform C build
├── config.yaml                # Tunable parameters
├── conftest.py                # Pytest path configuration
├── requirements.txt
└── README.md
```

## What Makes This Off-Grid

Three concrete properties, not marketing:

1. **No network call is required for any core function.** The solver, DB, and dashboard run entirely on localhost.

2. **The C solver is compiled once and runs at native speed on an ARM Cortex-A72** (Raspberry Pi 4). Benchmarks on the dev machine show ~2 ms per stress field; the Pi runs the same workload in ~15 ms, comfortably inside the 100 ms telemetry tick.

3. **The sync queue is durable across restarts.** SQLite's WAL mode means a power cut cannot lose more than one in-flight transaction.

## Deployment on Real Hardware

```bash
# On a Raspberry Pi 4 running Raspberry Pi OS 64-bit:
sudo apt install gcc python3-pip
git clone https://github.com/altugrakay-commits/OFFGRID-excavator-digital-twin.git
cd OFFGRID-excavator-digital-twin
pip install -r requirements.txt
python build.py
streamlit run src/edge/dashboard.py --server.address=0.0.0.0 --server.port=8501
```

### Docker (optional)

A `Dockerfile` and `dockerignore`is included for reproducibility. On systems where hardware virtualization cannot be enabled (common on locked-down corporate laptops), the same stack runs natively with the steps above — no container required. The container adds no functionality; it only packages the runtime.

The operator's tablet connects to the same local Wi-Fi hotspot and opens `http://<pi-ip>:8501`. The Pi's cellular modem (or manual sync at the site office) drains the queue when a link appears.

## Testing
```bash
pytest tests/ -v
```
Covers:

 - Backend parity (C vs NumPy produce identical stress fields)
 - Fatigue monotonicity (higher stress → more damage)
 - Sync idempotence (rows marked synced=1 are not re-sent)

## Roadmap
☑ C stress solver with NumPy fallback

☑ Physics-informed fatigue model (Basquin + Goodman)

☑ Offline-first SQLite storage

☑ Material profiles (soil / mixed / rock)

☑ Streamlit in-cab dashboard

☑ Offline→online sync engine

□ Replace synthetic telemetry with OPC-UA / CAN-bus reader

□ Ship as a systemd service on Raspberry Pi OS

□ Add stpyvista WebGL viewer for smoother 3D rendering

## License
MIT — see `LICENSE` file.

## Author
Altug Remzi Akay, Ph.D.

Bergsskolan (Metallurgy & Materials Engineering) · KTH Royal Institute of Technology

Email: altug.akay@bergsskolan.se

github.com/altugrakay-commits
