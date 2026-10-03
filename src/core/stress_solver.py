"""
Python interface to the C stress solver.

Tries to load the compiled shared library (libstress.so / .dylib / .dll).
If the library is unavailable, falls back to a pure-NumPy implementation
so the code still runs for debugging and development.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Optional

import numpy as np

# ------------------------------------------------------------------ #
# Locate and load the shared library
# ------------------------------------------------------------------ #

_ROOT = Path(__file__).resolve().parent.parent.parent

if platform.system() == "Darwin":
    _LIB_NAMES = ["libstress.dylib"]
elif platform.system() == "Windows":
    _LIB_NAMES = ["libstress.dll"]
else:
    _LIB_NAMES = ["libstress.so"]

_lib: Optional[ctypes.CDLL] = None
_USE_C = False

for _name in _LIB_NAMES:
    _candidate = _ROOT / _name
    if _candidate.exists():
        try:
            _lib = ctypes.CDLL(str(_candidate))
            _USE_C = True
            break
        except OSError:
            _lib = None

if _USE_C and _lib is not None:
    # Use c_void_p for all pointers — this is the most portable way to
    # marshal NumPy array addresses across Windows / Linux / macOS.
    _lib.compute_boom_stress.argtypes = [
        ctypes.c_void_p, ctypes.c_int,   # points, n_points
        ctypes.c_void_p, ctypes.c_int,   # loads,  n_loads
        ctypes.c_void_p, ctypes.c_int,   # geom,   n_geom
        ctypes.c_void_p,                  # out_stress
    ]
    _lib.compute_boom_stress.restype = None

    _lib.compute_fatigue_damage.argtypes = [
        ctypes.c_void_p, ctypes.c_int,   # history, n_samples
        ctypes.c_double,                  # ultimate_strength
        ctypes.c_void_p,                  # out_damage
    ]
    _lib.compute_fatigue_damage.restype = None


def backend_name() -> str:
    """Return which backend is active."""
    return "C (libstress)" if _USE_C else "NumPy (fallback)"


# ------------------------------------------------------------------ #
# Pure-NumPy fallback (unchanged)
# ------------------------------------------------------------------ #

def _compute_boom_stress_numpy(points: np.ndarray, loads: np.ndarray, geom: np.ndarray) -> np.ndarray:
    
    L = float(geom[0])
    w_root, h_root, w_tip, h_tip = (float(geom[1]), float(geom[2]),
                                    float(geom[3]), float(geom[4]))
    wall_t = float(geom[5]) if len(geom) >= 6 else 0.02

    Fx = Fy = Fz = 0.0
    Mx = My = Mz = 0.0
    if len(loads) >= 6:
        Fx, Fy, Fz, Mx, My, Mz = [float(v) for v in loads[:6]]
    elif len(loads) >= 3:
        Fx, Fy, Fz = [float(v) for v in loads[:3]]
    elif len(loads) == 1:
        Fz = float(loads[0])

    x = np.clip(points[:, 0], 0.0, L)
    y = points[:, 1]
    z = points[:, 2]
    t = x / L

    w = w_root + (w_tip - w_root) * t
    h = h_root + (h_tip - h_root) * t
    w = np.maximum(w, 1e-12)
    h = np.maximum(h, 1e-12)

    I_y = w * h**3 / 12.0
    I_z = h * w**3 / 12.0
    J = I_y + I_z

    My_x = Fz * (L - x) + My
    Mz_x = Fy * (L - x) + Mz

    sigma_b = My_x * z / I_y + Mz_x * y / I_z

    r = np.sqrt(y**2 + z**2)
    tau_t = Mx * r / np.maximum(J, 1e-12)

    F_trans = np.sqrt(Fy**2 + Fz**2)
    A_web = 2.0 * h * wall_t
    tau_s = F_trans / np.maximum(A_web, 1e-12)

    bump = 1.0 + 0.45 * np.exp(-((x - 0.30 * L) ** 2) / (0.005 * L * L))
    sigma_b *= bump

    tau_total = tau_t + tau_s
    sigma_vm = np.sqrt(sigma_b**2 + 3.0 * tau_total**2)

    return np.nan_to_num(sigma_vm, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def compute_stress(points: np.ndarray, loads: np.ndarray, geom: np.ndarray) -> np.ndarray:
    
    """
    Compute Von Mises stress at each point.

    Parameters
    ----------
    points : (N, 3) float64 array
    loads  : (k,) float64 array, k in {1, 3, 6}
    geom   : (>=5,) float64 array

    Returns
    -------
    stress : (N,) float64 array in Pascals
    """

    # Ensure C-contiguous float64 arrays (no copies if already correct)
    points = np.ascontiguousarray(points, dtype=np.float64)
    loads = np.ascontiguousarray(loads, dtype=np.float64).ravel()
    geom = np.ascontiguousarray(geom, dtype=np.float64).ravel()

    n_points = int(points.shape[0])
    out = np.zeros(n_points, dtype=np.float64)

    if _USE_C and _lib is not None:
        # Pass raw addresses as Python ints -> c_void_p handles them.
        _lib.compute_boom_stress(
            points.ctypes.data, ctypes.c_int(n_points),
            loads.ctypes.data, ctypes.c_int(loads.size),
            geom.ctypes.data, ctypes.c_int(geom.size),
            out.ctypes.data,
        )
        return out

    return _compute_boom_stress_numpy(points, loads, geom)


def compute_fatigue_damage(stress_history: np.ndarray, fatigue_strength: float = 900e6) -> float:
    
    """
    Compute cumulative fatigue damage (Miner's rule).

    Parameters
    ----------
    stress_history : (N,) array of stress amplitudes (Pa)
    ultimate_strength : material ultimate strength (Pa)
                       For steel this is approximately
                       2× the ultimate tensile strength.
                       Default 900 MPa.

    Returns
    -------
    damage : float, 1.0 = predicted failure
    """

    stress_history = np.ascontiguousarray(stress_history, dtype=np.float64).ravel()

    n = int(stress_history.size)

    if _USE_C and _lib is not None:
        out = ctypes.c_double(0.0)
        _lib.compute_fatigue_damage(
            stress_history.ctypes.data, ctypes.c_int(n),
            ctypes.c_double(fatigue_strength),
            ctypes.byref(out),
        )
        return float(out.value)

    S = stress_history[stress_history > 1e-12]
    if S.size == 0:
        return 0.0
    N = (fatigue_strength / S) ** 10.0
    return float(np.sum(1.0 / N))