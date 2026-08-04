#!/usr/bin/env python3
"""
perigee_boundary.py
Numerical verification of the J2-corrected perigee-constraint boundary.
Reproduces Tables III, IV, V and Figures 1, 2 (Sec. IV-V).
"""

import numpy as np
from numpy import sqrt, sin, cos, pi
from scipy.integrate import solve_ivp

# Physical constants
MU = 398600.4418
RE = 6378.137
J2 = 1.08262668e-3

def j2_acceleration(t, state):
    """J2-perturbed equations of motion (Cartesian)."""
    x, y, z, vx, vy, vz = state
    r = sqrt(x**2 + y**2 + z**2)
    r2 = r*r
    r3 = r2*r

    # Two-body
    ax = -MU*x/r3
    ay = -MU*y/r3
    az = -MU*z/r3

    # J2 perturbation
    z2_r2 = (z/r)**2
    factor = 1.5 * MU * J2 * (RE/r)**2 / r3
    ax += factor * x * (5*z2_r2 - 1)
    ay += factor * y * (5*z2_r2 - 1)
    az += factor * z * (5*z2_r2 - 3)

    return [vx, vy, vz, ax, ay, az]

def rv2coe(r, v):
    """Convert Cartesian state to classical orbital elements."""
    rvec = np.array(r)
    vvec = np.array(v)

    hvec = np.cross(rvec, vvec)
    h = np.linalg.norm(hvec)

    evec = np.cross(vvec, hvec)/MU - rvec/np.linalg.norm(rvec)
    e = np.linalg.norm(evec)

    v2 = np.dot(vvec, vvec)
    r_mag = np.linalg.norm(rvec)
    E = 0.5*v2 - MU/r_mag
    a = -MU/(2*E) if E != 0 else np.inf

    # Inclination
    i = np.arccos(hvec[2]/h)

    # Node
    nvec = np.cross([0,0,1], hvec)
    n = np.linalg.norm(nvec)

    # Argument of perigee
    if n == 0:
        omega = 0.0
    else:
        omega = np.arccos(np.clip(np.dot(nvec, evec)/(n*e), -1, 1))
        if evec[2] < 0:
            omega = 2*pi - omega

    # True anomaly
    if e < 1e-12:
        nu = 0.0
    else:
        nu = np.arccos(np.clip(np.dot(evec, rvec)/(e*r_mag), -1, 1))
        if np.dot(rvec, vvec) < 0:
            nu = 2*pi - nu

    return a, e, i, omega, nu

def coe2rv(a, e, i, Omega, omega, nu):
    """Convert classical elements to Cartesian state."""
    p = a*(1 - e**2)
    r = p / (1 + e*cos(nu))

    # Position in PQW frame
    r_pqw = np.array([r*cos(nu), r*sin(nu), 0.0])
    v_pqw = np.array([-sqrt(MU/p)*sin(nu), sqrt(MU/p)*(e + cos(nu)), 0.0])

    # Rotation matrix PQW -> IJK
    cO, sO = cos(Omega), sin(Omega)
    co, so = cos(omega), sin(omega)
    ci, si = cos(i), sin(i)

    R = np.array([
        [cO*co - sO*so*ci, -cO*so - sO*co*ci, sO*si],
        [sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
        [si*so, si*co, ci]
    ])

    rvec = R @ r_pqw
    vvec = R @ v_pqw
    return np.concatenate([rvec, vvec])

def integrate_and_extract(a0, e0, i0, Omega0=0.0, omega0=0.0, nu0=0.0, n_periods=20):
    """Integrate J2-perturbed orbit and extract osculating elements."""
    T0 = 2*pi * sqrt(a0**3 / MU)
    state0 = coe2rv(a0, e0, i0, Omega0, omega0, nu0)

    t_span = (0, n_periods * T0)
    t_eval = np.linspace(0, n_periods * T0, 2000)

    sol = solve_ivp(j2_acceleration, t_span, state0, t_eval=t_eval,
                    method='RK45', rtol=1e-12, atol=1e-12, dense_output=True)

    a_arr, e_arr, rp_arr = [], [], []
    for t, state in zip(sol.t, sol.y.T):
        a, e, i, omega, nu = rv2coe(state[:3], state[3:])
        a_arr.append(a)
        e_arr.append(e)
        rp_arr.append(a*(1-e))

    return sol.t, np.array(a_arr), np.array(e_arr), np.array(rp_arr), T0

# ---------------------------------------------------------------------------
# Analytic formulas (Eqs. 22-25)
# ---------------------------------------------------------------------------
def analytic_delta_rp(a, e, i_deg, u_deg):
    """Compute delta_r_p(u) from Eqs. 22-25 (omega=0)."""
    i = np.radians(i_deg)
    u = np.radians(u_deg)
    si2 = sin(i)**2

    A = 1.5 * J2 * (RE**2/a) * si2 * (e - 1)
    B = (3.0/8.0) * J2 * (RE**2/a) * (5*si2 - 4)
    C = -(7.0/8.0) * J2 * (RE**2/a) * si2

    return A*cos(2*u) + B*cos(u) + C*cos(3*u)

def analytic_peak_to_peak(a, e, i_deg):
    """Peak-to-peak amplitude of delta_r_p by exact cubic extremization."""
    i = np.radians(i_deg)
    si2 = sin(i)**2

    A = 1.5 * J2 * (RE**2/a) * si2 * (e - 1)
    B = (3.0/8.0) * J2 * (RE**2/a) * (5*si2 - 4)
    C = -(7.0/8.0) * J2 * (RE**2/a) * si2

    # f(x) = 4Cx^3 + 2Ax^2 + (B-3C)x - A, x = cos(u)
    # Extrema at x = +/-1 or f'(x) = 12Cx^2 + 4Ax + (B-3C) = 0
    coeffs = [12*C, 4*A, B - 3*C]
    roots = np.roots(coeffs)

    candidates = [-1.0, 1.0]
    for r in roots:
        if np.isreal(r) and -1 <= np.real(r) <= 1:
            candidates.append(np.real(r))

    vals = []
    for x in candidates:
        val = A*(2*x**2 - 1) + B*x + C*(4*x**3 - 3*x)
        vals.append(val)

    return max(vals) - min(vals)

# ---------------------------------------------------------------------------
# Table III: Verification at e >= 0.01
# ---------------------------------------------------------------------------
def table_verify():
    print("Table III: Analytic vs numerical peak-to-peak (omega=0)")
    print("e\t\ti (deg)\tanalytic (km)\tnumeric (km)\t% error")

    cases = [(0.010, 45.0), (0.050, 63.4), (0.001, 90.0)]
    for e0, i0 in cases:
        t, a_arr, e_arr, rp_arr, T0 = integrate_and_extract(7000, e0, np.radians(i0), n_periods=20)

        # Extract one orbit in the middle (avoid initial transients)
        mid = len(t)//2
        idx = slice(mid, mid + int(2*T0/(t[1]-t[0])))
        rp_single = rp_arr[idx]
        numeric_pp = max(rp_single) - min(rp_single)

        analytic_pp = analytic_peak_to_peak(7000, e0, i0)
        err = 100*(analytic_pp - numeric_pp)/numeric_pp
        print("%.3f\t%.1f\t\t%.2f\t\t\t%.2f\t\t\t%.1f" % (e0, i0, analytic_pp, numeric_pp, err))

# ---------------------------------------------------------------------------
# Table IV: Altitude generality (MEO/GEO)
# ---------------------------------------------------------------------------
def table_altitude():
    print("\nTable IV: Altitude generality")
    print("a (km)\t\te\t\ti (deg)\tanalytic (km)\tnumeric (km)\t% error")

    cases = [
        (26560, 0.010, 55.0),
        (26560, 0.020, 55.0),
        (20000, 0.010, 63.4),
        (42164, 0.010, 5.0),
        (42164, 0.050, 10.0),
    ]
    for a0, e0, i0 in cases:
        t, a_arr, e_arr, rp_arr, T0 = integrate_and_extract(a0, e0, np.radians(i0), n_periods=20)
        mid = len(t)//2
        idx = slice(mid, mid + int(2*T0/(t[1]-t[0])))
        rp_single = rp_arr[idx]
        numeric_pp = max(rp_single) - min(rp_single)

        analytic_pp = analytic_peak_to_peak(a0, e0, i0)
        err = 100*(analytic_pp - numeric_pp)/numeric_pp
        print("%d\t%.3f\t%.1f\t\t%.2f\t\t\t%.2f\t\t\t%.1f" % (a0, e0, i0, analytic_pp, numeric_pp, err))

if __name__ == "__main__":
    table_verify()
    table_altitude()
