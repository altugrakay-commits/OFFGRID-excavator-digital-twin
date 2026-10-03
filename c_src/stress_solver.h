#ifndef STRESS_SOLVER_H
#define STRESS_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute Von Mises stress at each mesh point of a tapered box-beam boom arm.
 *
 * The boom extends along +X from x=0 (root/pivot) to x=L (tip).
 * Cross-section is a rectangle (width along Y, height along Z) that
 * tapers linearly from root to tip.
 *
 * @param points     Flat array of N*3 doubles: [x0,y0,z0, x1,y1,z1, ...]
 * @param n_points   Number of mesh points
 * @param loads      Array of load values:
 *                   [Fx, Fy, Fz, Mx, My, Mz]  (forces in N, moments in N*m)
 *                   If n_loads < 6, missing values default to 0.
 * @param n_loads    Number of load entries (0..6)
 * @param geom       Array of geometry parameters:
 *                   [L, w_root, h_root, w_tip, h_tip, wall_t]
 *                   (lengths in meters)
 * @param n_geom     Number of geometry entries (must be >= 5)
 * @param out_stress Output array of N doubles (Von Mises stress in Pa)
 */
void compute_boom_stress(
    const double* points,
    int n_points,
    const double* loads,
    int n_loads,
    const double* geom,
    int n_geom,
    double* out_stress 
);

/**
 * Compute cumulative fatigue damage using Miner's rule.
 *
 * @param stress_history  Array of stress amplitudes (Pa)
 * @param n_samples       Number of samples
 * @param ultimate_strength  Material ultimate strength (Pa)
 * @param out_damage      Output: cumulative damage (1.0 = failure)
 */
void compute_fatigue_damage(
    const double* stress_history,
    int n_samples,
    double ultimate_strength,
    double* out_damange
);
#ifdef __cplusplus

}
#endif

#endif /* STRESS_SOLVER_H */