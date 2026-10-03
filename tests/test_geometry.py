"""Tests for the boom mesh generator and load-vector helpers."""

import numpy as np
import pyvista as pv

from src.core.geometry import (
    create_boom_mesh,
    boom_geometry_params,
    default_loads,
)


def test_mesh_returns_polydata():
    mesh = create_boom_mesh(n_segments=10, subdivide=0)
    assert isinstance(mesh, pv.PolyData)


def test_mesh_has_points_and_cells():
    mesh = create_boom_mesh(n_segments=10, subdivide=0)
    assert mesh.n_points > 0
    assert mesh.n_cells > 0


def test_mesh_bounding_box_matches_length():
    L = 3.0
    mesh = create_boom_mesh(length=L, n_segments=20, subdivide=0)
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    assert bounds[0] == 0.0
    assert abs(bounds[1] - L) < 1e-6


def test_mesh_root_wider_than_tip():
    mesh = create_boom_mesh(
        root_width=0.30, tip_width=0.10,
        root_height=0.40, tip_height=0.15,
        n_segments=20, subdivide=0,
    )
    pts = np.asarray(mesh.points)
    root_pts = pts[pts[:, 0] < 0.1]
    tip_pts = pts[pts[:, 0] > 2.9]

    root_span_y = root_pts[:, 1].max() - root_pts[:, 1].min()
    tip_span_y = tip_pts[:, 1].max() - tip_pts[:, 1].min()
    assert root_span_y > tip_span_y


def test_subdivide_increases_resolution():
    coarse = create_boom_mesh(n_segments=10, subdivide=0)
    fine = create_boom_mesh(n_segments=10, subdivide=1)
    assert fine.n_points > coarse.n_points


def test_geometry_params_shape():
    geom = boom_geometry_params()
    assert geom.shape == (6,)
    assert geom.dtype == np.float64
    assert geom[0] > 0.0      # length positive
    assert geom[1] > geom[3]  # root wider than tip


def test_default_loads_shape():
    loads = default_loads()
    assert loads.shape == (6,)
    assert loads.dtype == np.float64


def test_default_loads_accepts_overrides():
    loads = default_loads(Fz=1e6, Mx=5e4)
    assert loads[2] == 1e6
    assert loads[3] == 5e4