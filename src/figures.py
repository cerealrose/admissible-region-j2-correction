#!/usr/bin/env python3
"""
figures.py
Generate all figures for the paper.
Requires outputs from perigee_boundary.py and general_omega.py.
"""

import numpy as np
from numpy import sin, cos, pi, sqrt
import matplotlib.pyplot as plt
from perigee_boundary import integrate_and_extract
from general_omega import delta_rp_omega, worst_case_margin

MU = 398600.4418
RE = 6378.137
J2 = 1.08262668e-3

def figure_secular():
    """Figure 2: Secular analysis over 20 orbital periods."""
    t, a_arr, e_arr, rp_arr, T0 = integrate_and_extract(7000, 0.01, np.radians(45), n_periods=20)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(t/T0, a_arr - 7000, 'k-', lw=0.8)
    axes[0].set_ylabel(r"$a(t) - a_0$ (km)")
    axes[0].set_title("Osculating semi-major axis")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t/T0, e_arr, 'k-', lw=0.8)
    axes[1].set_ylabel(r"$e(t)$")
    axes[1].set_title("Osculating eccentricity")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t/T0, rp_arr, 'k-', lw=0.8)
    axes[2].set_ylabel(r"$r_p(t)$ (km)")
    axes[2].set_xlabel("Orbital periods")
    axes[2].set_title("Perigee radius")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../figures/j2_secular_analysis.png", dpi=300)
    print("Saved j2_secular_analysis.png")

def figure_single_orbit():
    """Figure 3: Short-period oscillation within one orbit."""
    t, a_arr, e_arr, rp_arr, T0 = integrate_and_extract(7000, 0.01, np.radians(45), n_periods=20)

    # Extract one orbit in the middle
    mid = len(t)//2
    idx = slice(mid, mid + int(2*T0/(t[1]-t[0])))
    t_single = t[idx] - t[mid]
    rp_single = rp_arr[idx]

    plt.figure(figsize=(8, 4))
    plt.plot(t_single, rp_single, 'k-', lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(r"$r_p(t)$ (km)")
    plt.title("Short-period oscillation of $r_p(t)$ (one orbit)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/j2_rp_single_orbit.png", dpi=300)
    print("Saved j2_rp_single_orbit.png")

    pp = max(rp_single) - min(rp_single)
    natural = J2 * RE**2 / 7000
    print("Peak-to-peak: %.2f km" % pp)
    print("Natural scale J2*Re^2/a: %.2f km" % natural)
    print("Ratio: %.2f" % (pp/natural))

def figure_margin_sweep():
    """Figure 4: Three-panel margin sweep heatmaps."""
    # Panel 1: e vs i at a=7000
    e_range = np.linspace(0.01, 0.05, 60)
    i_range = np.linspace(0.1, 179.9, 90)
    E, I = np.meshgrid(e_range, i_range)
    M1 = np.zeros_like(E)
    for j in range(E.shape[1]):
        for k in range(E.shape[0]):
            M1[k,j] = worst_case_margin(7000, E[k,j], I[k,j], 0, n_u=2000)

    # Panel 2: a vs e at i=45
    a_range = np.linspace(6600, 8000, 60)
    e_range2 = np.linspace(0.01, 0.05, 60)
    A2, E2 = np.meshgrid(a_range, e_range2)
    M2 = np.zeros_like(A2)
    for j in range(A2.shape[1]):
        for k in range(A2.shape[0]):
            M2[k,j] = worst_case_margin(A2[k,j], E2[k,j], 45, 0, n_u=2000)

    # Panel 3: a vs i at e=0.02
    a_range3 = np.linspace(6600, 8000, 60)
    i_range3 = np.linspace(0.1, 179.9, 90)
    A3, I3 = np.meshgrid(a_range3, i_range3)
    M3 = np.zeros_like(A3)
    for j in range(A3.shape[1]):
        for k in range(A3.shape[0]):
            M3[k,j] = worst_case_margin(A3[k,j], 0.02, I3[k,j], 0, n_u=2000)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im1 = axes[0].pcolormesh(E*100, I, M1, shading='auto', cmap='viridis')
    axes[0].set_xlabel(r"$e$ (%)")
    axes[0].set_ylabel(r"$i$ (deg)")
    axes[0].set_title("Panel 1: $a=7000$ km")
    plt.colorbar(im1, ax=axes[0], label="Margin (km)")

    im2 = axes[1].pcolormesh(A2, E2*100, M2, shading='auto', cmap='viridis')
    axes[1].set_xlabel(r"$a$ (km)")
    axes[1].set_ylabel(r"$e$ (%)")
    axes[1].set_title("Panel 2: $i=45^\circ$")
    plt.colorbar(im2, ax=axes[1], label="Margin (km)")

    im3 = axes[2].pcolormesh(A3, I3, M3, shading='auto', cmap='viridis')
    axes[2].set_xlabel(r"$a$ (km)")
    axes[2].set_ylabel(r"$i$ (deg)")
    axes[2].set_title("Panel 3: $e=0.02$")
    plt.colorbar(im3, ax=axes[2], label="Margin (km)")

    plt.tight_layout()
    plt.savefig("../figures/margin_sweep_panels.png", dpi=300)
    print("Saved margin_sweep_panels.png")
    print("Margin ranges: Panel 1: %.2f-%.2f km" % (M1.min(), M1.max()))
    print("Margin ranges: Panel 2: %.2f-%.2f km" % (M2.min(), M2.max()))
    print("Margin ranges: Panel 3: %.2f-%.2f km" % (M3.min(), M3.max()))

if __name__ == "__main__":
    figure_secular()
    figure_single_orbit()
    figure_margin_sweep()
