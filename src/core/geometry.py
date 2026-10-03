"""
Geometry generation for the excavator boom arm.

The boom is modelled as a tapered, hollow box beam extending along +X.
It includes:
  - A pivot boss at the root (thicker section)
  - A hydraulic cylinder mount (stress riser) at ~30% span
  - A bucket linkage boss at the tip
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


def create_boom_mesh(
    length: float = 3.0,
    root_width: float = 0.25,
    root_height: float = 0.35,
    tip_width: float = 0.15,
    tip_height: float = 0.22,
    n_segments: int = 60,
    subdivide: int = 1,
) -> pv.PolyData:
    
    """
    Build a tapered box-beam mesh for the excavator boom.

    Parameters
    ----------
    length : float
        Boom length along +X (meters).
    root_width, root_height : float
        Cross-section dimensions at the root (meters).
    tip_width, tip_height : float
        Cross-section dimensions at the tip (meters).
    n_segments : int
        Number of segments along the boom length.
    subdivide : int
        Number of subdivision passes for smoother stress fields.

    Returns
    -------
    mesh : pv.PolyData
        Triangulated surface mesh with point coordinates.
    """

    points = []

    for i in range(n_segments + 1):
        t = i / n_segments
        x = t * length

        # Interpolated cross-section
        w = root_width + (tip_width - root_width) * t
        h = root_height + (tip_height - root_height) * t

        # Pivot boss: slightly thicker at the root
        if t < 0.05:
            w *= 1.15
            h *= 1.15

        # Cylinder mount: slight bulge at 30% span
        if 0.27 < t < 0.33:
            bulge = 1.0 + 0.25 * np.exp(-((t - 0.30) ** 2) / 0.0005)
            w *= bulge
            h *= bulge

        # Bucket boss at the tip
        if t > 0.95:
            w *= 1.10
            h *= 1.10

        # Four corners of the rectangular cross-section
        points.append([x, -w / 2, -h / 2])
        points.append([x,  w / 2, -h / 2])
        points.append([x,  w / 2,  h / 2])
        points.append([x, -w / 2,  h / 2])

    points = np.array(points, dtype=np.float64)

    # ---- Build faces ----
    faces = []

    # Side faces (4 per segment)
    for i in range(n_segments):
        for j in range(4):
            j_next = (j + 1) % 4
            p0 = i * 4 + j
            p1 = i * 4 + j_next
            p2 = (i + 1) * 4 + j_next
            p3 = (i + 1) * 4 + j
            faces.append([4, p0, p1, p2, p3])

    # Root cap
    faces.append([4, 0, 1, 2, 3])

    # Tip cap
    base = n_segments * 4
    faces.append([4, base + 0, base + 3, base + 2, base + 1])

    faces = np.hstack(faces)

    mesh = pv.PolyData(points, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()

    if subdivide > 0:
        mesh = mesh.subdivide(subdivide, 'linear')

    mesh.compute_normals(inplace=True)

    return mesh # type: ignore

def boom_geometry_params(
    length: float = 3.0,
    root_width: float = 0.25,
    root_height: float = 0.35,
    tip_width: float = 0.15,
    tip_height: float = 0.22,
    wall_thickness: float = 0.02,
) -> np.ndarray:
    
    """
    Return the geometry parameter vector expected by the C solver.

    Order: [L, w_root, h_root, w_tip, h_tip, wall_t]
    """

    return np.array([
        length,
        root_width,
        root_height,
        tip_width,
        tip_height,
        wall_thickness,
    ], dtype=np.float64)


def default_loads(
    Fx: float = 0.0,
    Fy: float = 2.0e4,     # 20 kN side force
    Fz: float = 2.0e5,     # 200 kN digging force
    Mx: float = 1.0e4,     # 10 kNm torsion
    My: float = 0.0,
    Mz: float = 0.0,
) -> np.ndarray:
    
    """Return a load vector [Fx, Fy, Fz, Mx, My, Mz]."""
    
    return np.array([Fx, Fy, Fz, Mx, My, Mz], dtype=np.float64)