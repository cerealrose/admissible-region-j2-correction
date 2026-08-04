#!/usr/bin/env python3
"""
general_omega.py
General argument of perigee analysis.
Reproduces Sec. IV.E and Figure 1 (margin vs omega).
"""

import numpy as np
from numpy import sin, cos, pi, sqrt
import matplotlib.pyplot as plt

# Physical constants
MU = 398600.4418
RE = 6378.137
J2 = 1.08262668e-3

def delta_rp_omega(a, e, i_deg, omega_deg, u_deg):
    """
    General-omega perigee correction (Eq. 28).
    Returns delta_r_p(u; omega).
    """
    i = np.radians(i_deg)
    omega = np.radians(omega_deg)
    u = np.radians(u_deg)
    si2 = sin(i)**2

    # Coefficients from Eqs. 29
    c1 = (3*J2*RE**2/(8*a**2)) * (4 - 5*si2) * cos(omega)
    s1 = (3*J2*RE**2/(8*a**2)) * (4 - 7*si2) * sin(omega)
    c3 = (7*J2*RE**2/(8*a**2)) * si2 * cos(omega)
    s3 = (7*J2*RE**2/(8*a**2)) * si2 * sin(omega)

    A = 1.5 * J2 * (RE**2/a) * si2 * (e - 1)

    # delta_a(u) is omega-independent
    delta_a = A * cos(2*u)

    # delta_e(u; omega) in Fourier form
    delta_e = c1*cos(u) + s1*sin(u) + c3*cos(3*u) + s3*sin(3*u)

    # delta_r_p = delta_a*(1-e) - a*delta_e
    return delta_a*(1-e) - a*delta_e

def worst_case_margin(a, e, i_deg, omega_deg, n_u=10000):
    """Find |min_u delta_r_p(u; omega)| over u in [0, 2pi]."""
    u_vals = np.linspace(0, 2*pi, n_u)
    drp = [delta_rp_omega(a, e, i_deg, omega_deg, np.degrees(u)) for u in u_vals]
    return abs(min(drp))

def generate_figure():
    """Reproduce Figure 1: margin vs omega."""
    a, e, i = 7000, 0.01, 45.0
    omegas = np.linspace(0, 360, 361)
    margins = [worst_case_margin(a, e, i, o) for o in omegas]

    plt.figure(figsize=(8, 4))
    plt.plot(omegas, margins, 'k-', lw=1.2)
    plt.xlabel(r"Argument of perigee $\omega$ (deg)")
    plt.ylabel(r"Worst-case margin $|$min$_u \delta r_p(u; \omega)|$ (km)")
    plt.title(f"$a={a}$ km, $e={e}$, $i={i}^\circ$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/margin_vs_omega.png", dpi=300)
    print("Saved ../figures/margin_vs_omega.png")

    # Report key values
    print("Margin at omega=0:   %.2f km" % worst_case_margin(a, e, i, 0))
    print("Margin at omega=90:  %.2f km" % worst_case_margin(a, e, i, 90))
    print("Margin at omega=180: %.2f km" % worst_case_margin(a, e, i, 180))
    print("Margin at omega=270: %.2f km" % worst_case_margin(a, e, i, 270))

if __name__ == "__main__":
    generate_figure()
