"""
Synthetic telemetry generator.

Produces a 10 Hz stream of hydraulic pressure, boom angle, vibration
and oil temperature. Converts each sample into the load vector the
stress solver consumes.

In a real deployment an OPC-UA or CAN-bus reader would replace 
this class. The public interface (sample() -> TelemetrySample,
loads_from_sample(sample) -> np.ndarray) stays the same.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

@dataclass
class TelemetrySample:
    t: float                        # seconds since epoch
    hydraulic_pressure_pa: float
    boom_angle_rad: float
    vibration_g: float
    oil_temp_c: float
    regime: str                     # "idle" | "digging" | "lifting"

def to_dict(self) -> dict:
    return asdict(self)

class TelemetrySimulator:
    """
    Regime-driven synthetic telemetry.

    State machine: IDLE → DIGGING → LIFTING → IDLE
    Regime transitions are driven by a seeded PRNG for reproducibility. 
    Shock events (rock strikes, hard slams) are superimposed on any regime.
    """
    REGIME_IDLE = "idle"
    REGIME_DIGGING = "digging"
    REGIME_LIFTING = "lifting"

    # p_mean / p_std are in Pa; vib_mean / vib_std in g;
    # angle_rate is the (min, max) rate in rad/s.
    REGIME_PROFILES = {
        REGIME_IDLE: {
            "p_mean": 3.0e6, "p_std": 0.3e6,
            "vib_mean": 0.05, "vib_std": 0.02,
            "angle_rate": (-0.05, 0.05),
        },
        REGIME_DIGGING: {
            "p_mean": 18.0e6, "p_std": 3.0e6,
            "vib_mean": 0.60, "vib_std": 0.15,
            "angle_rate": (-0.15, 0.15),
        },
        REGIME_LIFTING: {
            "p_mean": 10.0e6, "p_std": 1.5e6,
            "vib_mean": 0.25, "vib_std": 0.08,
            "angle_rate": (0.0, 0.08),
        },
    }

    RELIEF_VALVE_PA = 32.0e6
    SHOCK_PROBABILITY = 0.004        # per sample, ~1 shock per 25 s
    SHOCK_DURATION_RANGE = (2, 8)    # samples
    SHOCK_PRESSURE_MULT = (1.4, 1.8)
    SHOCK_VIBRATION_MIN = 1.5
    SHOCK_VIBRATION_MAX = 3.0

    # Material profiles: the geology the machine is working in.
    # 'rock' operates at sustained high pressure (abuse-like).
    # 'soil' is light duty. 'mixed' is the baseline used so far.

    MATERIAL_PROFILES = {
        "soil":     {"digging_p_mean": 10.0e6, "abuse": False, "shock_mult": 0.8},
        "mixed":    {"digging_p_mean": 16.0e6, "abuse": False, "shock_mult": 1.0},
        "rock":     {"digging_p_mean": 22.0e6, "abuse": True, "shock_mult": 1.2},
    }

    def __init__(
            self,
            cylinder_area_m2: float = 0.008,     # ~100 mm bore, mini-excavator
            boom_length: float = 3.0,
            seed: int = 1234,
            start_time: Optional[float] = None,
            material: str = "mixed"
    ) -> None:
        if material not in self.MATERIAL_PROFILES:
            raise ValueError(f"Unknown material: {material}")
        
        self.cylinder_area = cylinder_area_m2
        self.boom_length = boom_length
        self.rng = np.random.default_rng(seed)
        self.t0 = start_time if start_time is not None else time.time()
        self.t = 0.0
        self.dt = 0.1       # 10 Hz
        self.regime = self.REGIME_IDLE
        self.boom_angle = math.radians(15.0)
        self.regime_timer = 0.0
        self.shock_remaining = 0    # samples remaining in a shock event

        self.material = material
        self._apply_material_profile()

    def _apply_material_profile(self) -> None:
        """Load the pressure scaling and abuse flag for the current material."""
        prof = self.MATERIAL_PROFILES[self.material]
        self._digging_p_mean = float(prof["digging_p_mean"])
        self.abuse_mode = bool(prof["abuse"])
        self._shock_mult = float(prof["shock_mult"])

    # --------------------------------------------------------------
    # Internal regime state machine
    # --------------------------------------------------------------

    def _step_regime(self) -> None:
        self.regime_timer -= self.dt
        if self.regime_timer > 0:
            return

        r = self.rng.random()
        if self.regime == self.REGIME_IDLE:
            if r < 0.30:
                self.regime = self.REGIME_DIGGING 
                self.regime_timer = self.rng.uniform(3.0, 8.0)
            else:
                self.regime_timer = self.rng.uniform(1.0, 3.0)
        elif self.regime == self.REGIME_DIGGING:
            if r < 0.40:
                self.regime = self.REGIME_LIFTING 
                self.regime_timer = self.rng.uniform(2.0, 5.0)
            else:
                self.regime_timer = self.rng.uniform(2.0, 5.0)
        else:   # lifting
            if r < 0.50:
                self.regime = self.REGIME_IDLE
                self.regime_timer = self.rng.uniform(2.0, 6.0)
            else:
                self.regime_timer = self.rng.uniform(1.0, 3.0)

    def _step_shock(self) -> None:
        """Advance the shock state machine by one sample."""
        if self.shock_remaining > 0:
            self.shock_remaining -= 1
            return

        if self.rng.random() < self.SHOCK_PROBABILITY:
            lo, hi = self.SHOCK_DURATION_RANGE
            self.shock_remaining = int(self.rng.integers(lo, hi))
    
    # --------------------------------------------------------------
    # Sample generation
    # --------------------------------------------------------------
    def sample(self) -> TelemetrySample:
        self._step_regime()
        self._step_shock()
        self.t += self.dt

        profile = dict(self.REGIME_PROFILES[self.regime])
        # Sustained overload: hard-rock digging holds the boom near relief
        if self.regime == self.REGIME_DIGGING:
            profile["p_mean"] = self._digging_p_mean

        # --- Base pressure and vibration for the current regime ---
        pressure = float(np.clip(self.rng.normal(profile["p_mean"], profile["p_std"]), 0.0, self.RELIEF_VALVE_PA))
        vibration = max(0.0, float(self.rng.normal(profile["vib_mean"], profile["vib_std"])))

        # --- Superimpose shock if one is active ---
        if self.shock_remaining > 0:
            lo, hi = self.SHOCK_PRESSURE_MULT
            mult = self.rng.uniform(lo, hi) * self._shock_mult
            pressure = min(pressure * mult, self.RELIEF_VALVE_PA)
            vibration = max(vibration, self.rng.uniform(self.SHOCK_VIBRATION_MIN, self.SHOCK_VIBRATION_MAX))

        # --- Boom angle integrates the regime's angular rate ---
        angle_lo, angle_hi = profile["angle_rate"]
        angle_rate = self.rng.uniform(angle_lo, angle_hi)
        self.boom_angle = float(np.clip(self.boom_angle + angle_rate * self.dt, math.radians(-10.0), math.radians(60.0)))

        # --- Oil temperature: base + digging heat + noise ---
        base_temp = 55.0
        dig_heat = 15.0 if self.regime == self.REGIME_DIGGING else 0.0
        oil_temp = base_temp + dig_heat + float(self.rng.normal(0.0, 0.5))

        return TelemetrySample(
            t=self.t0 + self.t,
            hydraulic_pressure_pa=pressure,
            boom_angle_rad=self.boom_angle,
            vibration_g=vibration,
            oil_temp_c=oil_temp,
            regime=self.regime,
        )

    # --------------------------------------------------------------
    # Load-vector derivation
    # --------------------------------------------------------------
    def loads_from_sample(self, s: TelemetrySample) -> np.ndarray:
        """
        Map a telemetry sample to the [Fx, Fy, Fz, Mx, My, Mz] load
        vector that compute_stress() expects.

        Mechanical advantage peaks near 35 degrees and tapers off at
        extreme angles. The C solver integrates the bending moment
        from Fz along the boom, so My here is only the small tip
        moment from bucket reaction.
        """
        cylinder_force = s.hydraulic_pressure_pa * self.cylinder_area
        arm = self.boom_length

        angle_deg = math.degrees(s.boom_angle_rad)
        gain = math.exp(-((angle_deg - 35.0) ** 2) / (2 * 25.0 ** 2))

        Fz = cylinder_force * gain
        Fy = 0.08 * Fz
        Fx = 0.04 * Fz

        Mx = 0.02 * Fy * arm
        My = 0.05 * Fz * arm
        Mz = 0.0

        return np.array([Fx, Fy, Fz, Mx, My, Mz], dtype=np.float64)