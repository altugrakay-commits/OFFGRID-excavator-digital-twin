"""Tests for the stress solver public API and backend behavior."""

import numpy as np
import pytest

from src.core.geometry import (
    create_boom_mesh,
    boom_geometry_params,
    default_loads,
)
from src.core.stress_solver import (
    backend_name,
    compute_stress,
    compute_fatigue_damage,
)


@pytest.fixture(scope="module")
def mesh_and_geom():
    mesh = create_boom_mesh(n_segments=15, subdivide=0)
    points = np.asarray(mesh.points, dtype=np.float64)
    geom = boom_geometry_params()
    return points, geom


# ------------------------------------------------------------------ #
# Backend identity
# ------------------------------------------------------------------ #

def test_backend_name_is_string():
    name = backend_name()
    assert isinstance(name, str)
    assert name  # non-empty


# ------------------------------------------------------------------ #
# compute_stress
# ------------------------------------------------------------------ #

def test_stress_shape_matches_points(mesh_and_geom):
    points, geom = mesh_and_geom
    s = compute_stress(points, default_loads(), geom)
    assert s.shape == (points.shape[0],)
    assert s.dtype == np.float64


def test_stress_non_negative_and_finite(mesh_and_geom):
    points, geom = mesh_and_geom
    s = compute_stress(points, default_loads(Fz=2e5), geom)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)


def test_stress_increases_with_load(mesh_and_geom):
    points, geom = mesh_and_geom
    low = compute_stress(points, default_loads(Fz=1e5), geom)
    high = compute_stress(points, default_loads(Fz=4e5), geom)
    assert high.max() > low.max()
    # Bending stress scales with Fz, but the peak node can shift between
    # the root and the cylinder-mount bump. Allow a generous window.
    ratio = high.max() / max(low.max(), 1.0)
    assert 3.0 < ratio < 4.5


def test_zero_load_produces_low_stress(mesh_and_geom):
    points, geom = mesh_and_geom
    s = compute_stress(points, np.zeros(6), geom)
    assert s.max() < 1e3


def test_invalid_geometry_does_not_crash(mesh_and_geom):
    points, _ = mesh_and_geom
    # length = 0 -> solver should return zeros, not divide by zero
    geom = np.array([0.0, 0.2, 0.3, 0.1, 0.2, 0.02])
    s = compute_stress(points, default_loads(), geom)
    assert np.all(np.isfinite(s))


# ------------------------------------------------------------------ #
# compute_fatigue_damage
# ------------------------------------------------------------------ #

def test_fatigue_returns_float():
    hist = np.full(100, 200e6)
    d = compute_fatigue_damage(hist, fatigue_strength=900e6)
    assert isinstance(d, float)


def test_fatigue_zero_for_empty_history():
    d = compute_fatigue_damage(np.array([]), fatigue_strength=900e6)
    assert d == 0.0


def test_fatigue_monotonic_with_stress():
    low = compute_fatigue_damage(np.full(1000, 100e6), fatigue_strength=900e6)
    high = compute_fatigue_damage(np.full(1000, 250e6), fatigue_strength=900e6)
    # 2.5x amplitude jump -> 2.5^10 ≈ 9.5e3 damage increase
    assert high > low * 5_000
    assert high < low * 20_000


def test_fatigue_scales_linearly_with_cycle_count():
    one = compute_fatigue_damage(np.full(100, 200e6), fatigue_strength=900e6)
    ten = compute_fatigue_damage(np.full(1000, 200e6), fatigue_strength=900e6)
    np.testing.assert_allclose(ten / one, 10.0, rtol=1e-9)


def test_fatigue_near_zero_below_endurance_limit():
    # 50 MPa cycles are effectively infinite life: (50/900)^10 * 1000 ≈ 2.8e-10
    d = compute_fatigue_damage(np.full(1000, 50e6), fatigue_strength=900e6)
    assert d < 1e-9