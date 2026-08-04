#!/usr/bin/env python3
"""
energy_boundary.py
Numerical verification of the J2-corrected energy-constraint boundary.
Reproduces Tables I and II (Sec. III.C).
"""

import numpy as np
from numpy import sqrt, sin, cos

# Physical constants (km, s)
MU = 398600.4418       # km^3/s^2
RE = 6378.137          # km
J2 = 1.08262668e-3

# Synthetic LEO attributable (Sec. III.C)
q = np.array([RE, 0.0, 0.0])
qdot = np.array([0.0, 0.4651, 0.0])
alpha, delta = 0.30, 0.50
alphadot, deltadot = 8e-4, 4e-4

# Line-of-sight and partials
u = np.array([cos(delta)*cos(alpha), cos(delta)*sin(alpha), sin(delta)])
u_alpha = np.array([-cos(delta)*sin(alpha), cos(delta)*cos(alpha), 0.0])
u_delta = np.array([-sin(delta)*cos(alpha), -sin(delta)*sin(alpha), cos(delta)])
w = alphadot*u_alpha + deltadot*u_delta
eta2 = alphadot**2 * cos(delta)**2 + deltadot**2
B = np.dot(qdot, nu)

# Position and speed^2 as functions of (rho, rhodot)
def r_of_rho(rho):
    return q + rho*nu

def r_mag(rho):
    return np.linalg.norm(r_of_rho(rho))

def speed2(rho, rhodot):
    return rhodot**2 + 2*B*rhodot + rho**2*eta2 + 2*rho*np.dot(qdot, w) + np.dot(qdot, qdot)

# Unperturbed energy and discriminant
def E0(rho, rhodot):
    return 0.5*speed2(rho, rhodot) - MU/r_mag(rho)

def Delta0(rho):
    # B^2 - 2*C(rho) where C is the rho-dependent part of E0
    C = 0.5*rho**2*eta2 + rho*np.dot(qdot, w) + 0.5*np.dot(qdot, qdot) - MU/r_mag(rho)
    return B**2 - 2*C

# J2 potential correction
def deltaE(rho):
    rvec = r_of_rho(rho)
    r = np.linalg.norm(rvec)
    rz = rvec[2]
    return 0.5*MU*J2*RE**2 / r**3 * (3*(rz/r)**2 - 1)

# Exact perturbed boundary
def rhodot_exact(rho, branch=+1):
    D = Delta0(rho) - 2*deltaE(rho)
    if D < 0:
        return np.nan
    return -B + branch*sqrt(D)

# First-order correction
def rhodot_corrected(rho, branch=+1):
    D0 = Delta0(rho)
    if D0 <= 0:
        return np.nan
    dE = deltaE(rho)
    return -B + branch*sqrt(D0) - branch*dE/sqrt(D0)

# Unperturbed boundary
def rhodot_0(rho, branch=+1):
    D0 = Delta0(rho)
    if D0 < 0:
        return np.nan
    return -B + branch*sqrt(D0)

# ---------------------------------------------------------------------------
# Table I: Convergence order (interior point, rho = 5000 km)
# ---------------------------------------------------------------------------
def table_convergence():
    rho_test = 5000.0
    print("Table I: Convergence at interior point rho = %.1f km" % rho_test)
    print("lambda\t\tresidual (km/s)\t\tratio")

    # Compute exact with full J2 strength
    dE_full = deltaE(rho_test)

    # Scale lambda down
    lambdas = [1.0, 0.5, 0.25, 0.125]
    resids = []
    for lam in lambdas:
        dE = lam * dE_full
        D0 = Delta0(rho_test)
        exact = -B + np.sqrt(D0 - 2*dE)
        approx = -B + np.sqrt(D0) - dE/np.sqrt(D0)
        res = abs(exact - approx)
        resids.append(res)
        ratio = resids[-1]/resids[-2] if len(resids)>1 else np.nan
        print("%.3f\t\t%.3e\t\t%.3f" % (lam, res, ratio))

# ---------------------------------------------------------------------------
# Table II: Edge breakdown
# ---------------------------------------------------------------------------
def table_edge():
    print("\nTable II: Breakdown near AR edge")
    print("rho (km)\tDelta0\t\tdeltaE\t\tresid (km/s)\tresid/Delta0^(-3/2)")

    # Find approximate outer edge
    rhos = np.linspace(8000, 8732, 100)
    Dvals = [Delta0(r) for r in rhos]
    # Find where Delta0 -> 0
    for rho_test in [8000.0, 8500.0, 8700.0, 8725.0, 8730.0, 8731.8]:
        D0 = Delta0(rho_test)
        if D0 <= 0:
            continue
        dE = deltaE(rho_test)
        exact = -B + np.sqrt(D0 - 2*dE)
        approx = -B + np.sqrt(D0) - dE/np.sqrt(D0)
        resid = exact - approx
        scale = resid * (D0**1.5)
        print("%.1f\t\t%.3f\t\t%.3e\t%.3e\t%.2e" % (rho_test, D0, dE, resid, scale))

if __name__ == "__main__":
    table_convergence()
    table_edge()
