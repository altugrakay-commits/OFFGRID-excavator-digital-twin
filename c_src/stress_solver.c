/* Heavy math: computes Von Mises stress from nodal loads.*/

#include "stress_solver.h"
#include <math.h>

#ifdef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Small epsilon to avoid division by zero */
#define EPS 1e-12

/* Stress concentration bump location (fraction of boom length) */
#define BUMP_LOCATION 0.30

/* Stress concentration bump amplitude and width */
#define BUMP_AMPLITUDE 0.45
#define BUMP_WIDTH_FRAC 0.005

void compute_boom_stress(
    const double* points,
    int n_points,
    const double* loads,
    int n_loads,
    const double* geom,
    int n_geom,
    double* out_stress
) {
    if (n_points <= 0 || n_geom < 5) return;

    /* ---- Unpack geometry ----*/
    const double L          = geom[0];
    const double w_root     = geom[1];
    const double h_root     = geom[2];
    const double w_tip      = geom[3];
    const double h_tip      = geom[4];
    const double wall_t     = (n_geom >= 6) ? geom[5] : 0.02;

    if (L <= EPS) return;

    /* ---- Unpack Loads (default 0) ---- */
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double Mx = 0.0, My = 0.0, Mz = 0.0;

    if (n_loads >= 6) {
        Fx = loads[0]; Fy = loads[1]; Fz = loads[2];
        Mx = loads[3]; My = loads[4]; Mz = loads[5];
    } else if (n_loads >= 3) {Fx = loads[0]; Fy = loads[1]; Fz = loads[2];
    } else if (n_loads >= 1) {Fz = loads[0];}

    /* Resultant transverse force magnitude (for shear) */
    const double F_trans = sqrt(Fy * Fy + Fz * Fz);

    /* ---- Loop over mesh points ---- */
    for (int i = 0; i < n_points; ++i) {
        double x = points[3 * i + 0];
        double y = points[3 * i + 1];
        double z = points[3 * i + 2];

        /* Clamp x to [0, L] for safety */
        if (x < 0.0) x = 0.0;
        if (x > L) x = L;

        const double t = x / L; /* normalized position */

        /* Interpolated cross section dimensions */
        double w = w_root + (w_tip - w_root) * t;
        double h = h_root + (h_tip - h_root) * t;
        
        if (w < EPS) w = EPS;
        if (h < EPS) h = EPS;

        /* ---- Section parameters ---- */
        /* Moment of inertia aobut y-axis (bending in the z-direction) */
        const double I_y = w * h * h * h / 12.0;

        /* Moment of inertia aobut z-axis (bending in the y-direction) */
        const double I_z = h * w * w * w / 12.0;
        
        /* Approximate polar moment (for torison) */
        const double J = I_y + I_z;

        /* ---- Bending moments at this x (cantilever model) ---- */
        /* Moment about y from Fz: M_y(x) = Fz * (L - x) + applied My */
        const double My_x = Fz * (L - x) + My;

        /* Moment about z from Fy: M_z(x) = Fy * (L - x) + applied Mz */
        const double Mz_x = Fy * (L - x) + Mz;

        /* ---- Bending stress (beam theory) ---- */
        /* sigma = M_y * z / I_y + M_z * y / I_z */
        double sigma_b = 0.0;
        sigma_b += (My_x * z) / I_y;
        sigma_b += (Mz_x * y) / I_z;

        /* ---- Torisonal shear stress ---- */
        const double r = sqrt(y * y + z * z);
        const double tau_t = (J > EPS) ? (Mx * r / J) : 0.0;
        
        /* ---- Transverse shear stress (two webs) ---- */
        const double A_web = 2.0 * h * wall_t;
        const double tau_s = (A_web > EPS) ? (F_trans / A_web) : 0.0;

        /* ---- Stress concentration factor (bump near 30% span) ----*/
        const double dx = x - BUMP_LOCATION * L;
        const double bump = 1.0 + BUMP_AMPLITUDE * exp(-(dx * dx) / (BUMP_WIDTH_FRAC * L * L));

        /* Apply concentration to bending stress */
        sigma_b *= bump;

        /* ---- Von Mises stress ---- */
        const double tau_total = tau_t + tau_s;
        double sigma_vm = sqrt(sigma_b * sigma_b + 3.0 * tau_total * tau_total);

        /* Guard against NaN */
        if (isnan(sigma_vm) || sigma_vm < 0.0) sigma_vm = 0.0;

        out_stress[i] = sigma_vm;
    }
}

/* ------------------------------------------------------------------ */
/* Fatigue: Miner's rule with a simplified S-N curve                  */
/* ------------------------------------------------------------------ */

void compute_fatigue_damage(
    const double* stress_history, 
    int n_samples, 
    double ultimate_strength, 
    double* out_damage
) {
    if (n_samples <= 0 || ultimate_strength <= EPS) {
        *out_damage = 0.0;
        return;
    }

    /*
     * Basquin-style S-N curve for high-cycle fatigue of steel:
     *
     *   N_f(S) = (reference_strength / S)^8
     *
     * The exponent 8 corresponds to |b| ≈ 0.125 in the
     * Basquin relation σ_a = σ'f (2N_f)^b, which is typical
     * for structural steels in the high-cycle regime.
     *
     * The reference_strength argument must be the fatigue
     * strength coefficient σ'f (≈ 2× ultimate tensile strength
     * for steel), NOT the ultimate tensile strength itself.
     *
     * Damage per cycle: D_i = 1 / N_f(S_i)
     * Cumulative damage: D = Σ D_i.   D ≥ 1 → predicted failure.
     */

     double damage = 0.0;
     const double exponent = 10.0;

     for (int i = 0; i < n_samples; ++i) {
        double S = stress_history[i];
        if (S <= EPS) continue;

        double ratio = ultimate_strength / S;
        double N = pow(ratio, exponent);

        if (N > EPS) {
            damage += 1.0 / N;
        }
     }

     *out_damage = damage;
    }