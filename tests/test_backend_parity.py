# tests/test_backend_parity.py
"""C solver must produce identical output to the NumPy fallback."""
import numpy as np
import pytest

from src.core.geometry import create_boom_mesh, boom_geometry_params, default_loads
from src.core.stress_solver import compute_stress, _compute_boom_stress_numpy


@pytest.fixture(scope="module")
def setup():
    mesh = create_boom_mesh(n_segments=10, subdivide=0)
    points = np.asarray(mesh.points, dtype=np.float64)
    loads = default_loads(Fy=2e4, Fz=2e5, Mx=1e4)
    geom = boom_geometry_params()
    return points, loads, geom


def test_c_matches_numpy(setup):
    points, loads, geom = setup
    c_result = compute_stress(points, loads, geom)
    py_result = _compute_boom_stress_numpy(points, loads, geom)
    np.testing.assert_allclose(c_result, py_result, rtol=1e-10, atol=1e-3)


def test_stress_non_negative(setup):
    points, loads, geom = setup
    s = compute_stress(points, loads, geom)
    assert np.all(s >= 0.0)
    assert np.all(np.isfinite(s))


def test_zero_load_low_stress(setup):
    points, _, geom = setup
    s = compute_stress(points, np.zeros(6), geom)
    assert s.max() < 1e3   # essentially zero