#!/usr/bin/env python3
"""
zonal_validation.py
Validation against J2-J6 zonal harmonics.
Reproduces Table V (Sec. V.E).
"""

import numpy as np
from numpy import sqrt, sin, cos, pi
from scipy.integrate import solve_ivp

MU = 398600.4418
RE = 6378.137

# Standard zonal coefficients
J = {
    2: 1.08262668e-3,
    3: -2.53265648e-6,
    4: -1.61962159e-6,
    5: -2.27296082e-7,
    6: 5.40681239e-7,
}

def zonal_acceleration(t, state, n_max=6):
    """Zonal-only equations of motion through J_n."""
    x, y, z, vx, vy, vz = state
    r = sqrt(x**2 + y**2 + z**2)
    r2 = r*r
    r3 = r2*r

    # Two-body
    ax = -MU*x/r3
    ay = -MU*y/r3
    az = -MU*z/r3

    # Zonal perturbations
    sin_phi = z / r

    # Potential derivative terms
    dV_dr = 0.0
    dV_dphi = 0.0

    for n in range(2, n_max+1):
        # Legendre polynomial P_n and derivative
        if n == 2:
            Pn = 0.5*(3*sin_phi**2 - 1)
            dPn = 3*sin_phi
        elif n == 3:
            Pn = 0.5*(5*sin_phi**3 - 3*sin_phi)
            dPn = 0.5*(15*sin_phi**2 - 3)
        elif n == 4:
            Pn = (1/8)*(35*sin_phi**4 - 30*sin_phi**2 + 3)
            dPn = (1/8)*(140*sin_phi**3 - 60*sin_phi)
        elif n == 5:
            Pn = (1/8)*(63*sin_phi**5 - 70*sin_phi**3 + 15*sin_phi)
            dPn = (1/8)*(315*sin_phi**4 - 210*sin_phi**2 + 15)
        elif n == 6:
            Pn = (1/16)*(231*sin_phi**6 - 315*sin_phi**4 + 105*sin_phi**2 - 5)
            dPn = (1/16)*(1386*sin_phi**5 - 1260*sin_phi**3 + 210*sin_phi)
        else:
            continue

        factor = -MU/r2 * J[n] * (RE/r)**n
        dV_dr += factor * (-(n+1)*Pn)
        dV_dphi += factor * dPn

    # Convert to Cartesian accelerations
    if r > 0:
        dphi_dx = -x*z/(r2*sqrt(x**2+y**2)) if sqrt(x**2+y**2)>0 else 0
        dphi_dy = -y*z/(r2*sqrt(x**2+y**2)) if sqrt(x**2+y**2)>0 else 0
        dphi_dz = sqrt(x**2+y**2)/r2

        ax += dV_dr * x/r + dV_dphi * dphi_dx
        ay += dV_dr * y/r + dV_dphi * dphi_dy
        az += dV_dr * z/r + dV_dphi * dphi_dz

    return [vx, vy, vz, ax, ay, az]

def rv2coe(r, v):
    """Convert Cartesian to classical elements (simplified)."""
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
    return a, e

def coe2rv(a, e, i, Omega, omega, nu):
    """Convert classical elements to Cartesian state."""
    p = a*(1 - e**2)
    r = p / (1 + e*cos(nu))
    r_pqw = np.array([r*cos(nu), r*sin(nu), 0.0])
    v_pqw = np.array([-sqrt(MU/p)*sin(nu), sqrt(MU/p)*(e + cos(nu)), 0.0])

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

def measure_peak_to_peak(a0, e0, i0, n_max=6):
    """Measure peak-to-peak r_p oscillation under zonal field through J_n."""
    T0 = 2*pi * sqrt(a0**3 / MU)
    state0 = coe2rv(a0, e0, i0, 0.0, 0.0, 0.0)

    t_span = (0, 20*T0)
    t_eval = np.linspace(0, 20*T0, 2000)

    def accel(t, y):
        return zonal_acceleration(t, y, n_max=n_max)

    sol = solve_ivp(accel, t_span, state0, t_eval=t_eval,
                    method='RK45', rtol=1e-12, atol=1e-12)

    rp_arr = []
    for state in sol.y.T:
        a, e = rv2coe(state[:3], state[3:])
        rp_arr.append(a*(1-e))

    # Middle orbit
    mid = len(sol.t)//2
    idx = slice(mid, mid + int(2*T0/(sol.t[1]-sol.t[0])))
    rp_single = np.array(rp_arr)[idx]
    return max(rp_single) - min(rp_single)

# Analytic peak-to-peak (J2-only formula)
def analytic_pp(a, e, i_deg):
    i = np.radians(i_deg)
    si2 = sin(i)**2
    J2 = J[2]
    A = 1.5 * J2 * (RE**2/a) * si2 * (e - 1)
    B = (3.0/8.0) * J2 * (RE**2/a) * (5*si2 - 4)
    C = -(7.0/8.0) * J2 * (RE**2/a) * si2

    coeffs = [12*C, 4*A, B - 3*C]
    roots = np.roots(coeffs)
    candidates = [-1.0, 1.0]
    for r in roots:
        if np.isreal(r) and -1 <= np.real(r) <= 1:
            candidates.append(np.real(r))

    vals = [A*(2*x**2 - 1) + B*x + C*(4*x**3 - 3*x) for x in candidates]
    return max(vals) - min(vals)

def table_zonal():
    print("Table V: Zonal validation (J2-only vs J2-J6)")
    print("e\t\ti (deg)\tanalytic (km)\tJ2-only % err\tJ2-J6 % err")

    cases = [
        (0.010, 45.0, 7000),
        (0.050, 63.4, 7000),
        (0.001, 90.0, 7000),
        (0.010, 55.0, 26560),
        (0.010, 5.0, 42164),
        (0.050, 10.0, 42164),
        (0.0005, 90.0, 7000),
        (0.0010, 90.0, 7000),
        (0.0100, 90.0, 7000),
    ]

    for e0, i0, a0 in cases:
        an = analytic_pp(a0, e0, i0)
        num_j2 = measure_peak_to_peak(a0, e0, np.radians(i0), n_max=2)
        num_j6 = measure_peak_to_peak(a0, e0, np.radians(i0), n_max=6)

        err_j2 = 100*(an - num_j2)/num_j2
        err_j6 = 100*(an - num_j6)/num_j6
        print("%.3f\t%.1f\t\t%.2f\t\t\t%.1f\t\t\t%.1f" % (e0, i0, an, err_j2, err_j6))

if __name__ == "__main__":
    table_zonal()
