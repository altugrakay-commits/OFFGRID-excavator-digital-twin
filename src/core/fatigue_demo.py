"""
Demonstration of fatigue accumulation at the critical stress location.

Real deployments feed this from a time-series buffer in the local DB.
For the smoke test, we synthesise a realistic load history.
"""

from __future__ import annotations

import numpy as np

from src.core.stress_solver import compute_fatigue_damage


def simulate_load_history(
    peak_stress_pa: float,
    n_cycles: int = 5_000,
    mean_ratio: float = 0.20,
    max_amp_ratio: float = 0.50,
    uts_pa: float = 450e6,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate equivalent fully-reversed stress amplitudes for Miner's rule.

    A real excavator duty cycle looks like:
      * A slowly varying operating mean stress (boom self-weight,
        hydraulic preload, dig resistance baseline).
      * A distribution of alternating amplitudes superimposed on
        that mean (positioning, scooping, break-out, releasing).
      * Most cycles are small; a few are severe.

    We model that as:
      sigma_m ~ mean_ratio * peak_stress
      sigma_a ~ lognormal(median = 0.15 * peak, tail up to max_amp_ratio * peak)
      sigma_a_eq = sigma_a / (1 - sigma_m / UTS)   [Goodman]

    The equivalent fully-reversed amplitude is what Basquin's law
    consumes. It is bounded so the result is physically meaningful.

    Parameters
    ----------
    peak_stress_pa : float
        Peak von Mises stress in the current mesh (Pa).
    n_cycles : int
        Number of representative load cycles to sample.
    mean_ratio : float
        Operating mean stress as a fraction of peak.
    max_amp_ratio : float
        Upper bound of raw amplitude as a fraction of peak.
    uts_pa : float
        Ultimate tensile strength of the cast steel (Pa).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    eq_amplitudes : (n_cycles,) array of equivalent fully-reversed
                    stress amplitudes (Pa).
    """
    rng = np.random.default_rng(seed)

    # ---- Operating mean stress ----
    mean = mean_ratio * peak_stress_pa

    # ---- Amplitude distribution ----
    # Median amplitude ≈ 15% of peak. Long tail toward 50% of peak.
    # lognormal(mean=log(0.15), sigma=0.6)
    median_frac = 0.15
    sigma_ln = 0.60
    mag = rng.lognormal(
        mean=np.log(median_frac), sigma=sigma_ln, size=n_cycles
    )
    mag = np.clip(mag, 0.02, max_amp_ratio)
    raw_amplitudes = mag * peak_stress_pa

    # ---- Goodman mean-stress correction ----
    denom = max(1.0 - mean / uts_pa, 0.05)   # guard against UTS approach
    eq_amplitudes = raw_amplitudes / denom

    # ---- Floor at 1 MPa to avoid log-of-zero ----
    return np.maximum(eq_amplitudes, 1e6)


def report_fatigue(
    peak_stress_pa: float,
    fatigue_strength_pa: float = 900e6,
    n_cycles: int = 5_000,
) -> dict:
    """Compute and print a fatigue summary. Returns a dict for the DB."""
    history = simulate_load_history(peak_stress_pa, n_cycles=n_cycles)
    damage = compute_fatigue_damage(
        history, fatigue_strength=fatigue_strength_pa
    )

    remaining_pct = max(0.0, (1.0 - damage) * 100.0)
    status = "FAILURE" if damage >= 1.0 else "OK"

    print(f"  Cycles simulated:     {n_cycles}")
    print(f"  Eq. amplitude (max):  {history.max() / 1e6:.1f} MPa")
    print(f"  Eq. amplitude (med):  {np.median(history) / 1e6:.1f} MPa")
    print(f"  Cumulative damage D:  {damage:.4e}")
    print(f"  Remaining life:       {remaining_pct:.3f}%")
    print(f"  Status:               {status}")

    return {
        "n_cycles": n_cycles,
        "damage": damage,
        "remaining_pct": remaining_pct,
        "status": status,
        "peak_cycle_mpa": history.max() / 1e6,
        "median_cycle_mpa": float(np.median(history) / 1e6),
    }