# tests/test_fatigue.py
"""Fatigue model sanity checks."""
import numpy as np

from src.core.stress_solver import compute_fatigue_damage


def test_damage_amplifies_with_stress():
    # Doubling the amplitude should multiply damage by 2^10 = 1024,
    # since the Basquin exponent is 10.
    d1 = compute_fatigue_damage(np.full(1000, 100e6), fatigue_strength=900e6)
    d2 = compute_fatigue_damage(np.full(1000, 200e6), fatigue_strength=900e6)
    ratio = d2 / d1
    assert 900 < ratio < 1100


def test_damage_zero_below_floor():
    d = compute_fatigue_damage(np.array([0.0, 0.0, 0.0]), fatigue_strength=900e6)
    assert d == 0.0


def test_damage_scales_linearly_with_cycles():
    one = compute_fatigue_damage(np.full(100, 200e6), fatigue_strength=900e6)
    ten = compute_fatigue_damage(np.full(1000, 200e6), fatigue_strength=900e6)
    np.testing.assert_allclose(ten / one, 10.0, rtol=1e-9)