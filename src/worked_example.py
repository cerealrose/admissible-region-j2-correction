"""
Sec. 6.1 worked example: synthetic LEO attributable, uncorrected vs.
J2-corrected admissible-region boundaries (energy constraint + perigee
constraint), reproducing Figure 4 (worked_example_boundaries.png).

Same synthetic attributable as Sec. 3.3 / Tables 1-2:
  station at q = (R_earth, 0, 0) km, qdot = (0, 0.4651, 0) km/s
  alpha=0.30, delta=0.50 rad, alphadot=8e-4, deltadot=4e-4 rad/s
  r_min = R_earth + 100 km

The right-panel zoom uses the perigee-constraint-active segment quoted in
the manuscript text, rho in [6088, 7385] km. If you re-run this against
your own saved worked-example intermediates and the shift/area numbers
differ from Sec. 6.1's quoted 0.0007-0.0047 km/s and 0.0029%, treat those
numbers (not this script) as the source of truth and let me know so I can
track down the discrepancy -- this is a from-scratch reconstruction of the
equations, not the original run.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
MU = 398600.4418          # km^3/s^2
R_EARTH = 6378.137        # km
J2 = 1.08262668e-3

# ---------------------------------------------------------- attributable
ALPHA, DELTA = 0.30, 0.50
ALPHADOT, DELTADOT = 8e-4, 4e-4
Q = np.array([R_EARTH, 0.0, 0.0])
QDOT = np.array([0.0, 0.4651, 0.0])
R_MIN = R_EARTH + 100.0

U_HAT = np.array([np.cos(DELTA) * np.cos(ALPHA),
                   np.cos(DELTA) * np.sin(ALPHA),
                   np.sin(DELTA)])
U_ALPHA = np.array([-np.cos(DELTA) * np.sin(ALPHA),
                     np.cos(DELTA) * np.cos(ALPHA),
                     0.0])
U_DELTA = np.array([-np.sin(DELTA) * np.cos(ALPHA),
                     -np.sin(DELTA) * np.sin(ALPHA),
                     np.cos(DELTA)])
W_VEC = ALPHADOT * U_ALPHA + DELTADOT * U_DELTA

B = np.dot(QDOT, U_HAT)
ETA2 = ALPHADOT ** 2 * np.cos(DELTA) ** 2 + DELTADOT ** 2
QW = np.dot(QDOT, W_VEC)
QDOT2 = np.dot(QDOT, QDOT)
QU = np.dot(Q, U_HAT)
Q2 = np.dot(Q, Q)


# ------------------------------------------------------ two-body AR (Sec. 2)
def r_of_rho(rho):
    return np.sqrt(Q2 + 2.0 * rho * QU + rho ** 2)


def C_of_rho(rho):
    return 0.5 * rho ** 2 * ETA2 + rho * QW + 0.5 * QDOT2 - MU / r_of_rho(rho)


def Delta0_of_rho(rho):
    return B ** 2 - 2.0 * C_of_rho(rho)


def rhodot0(rho, branch):
    d = np.maximum(Delta0_of_rho(rho), 0.0)
    return -B + branch * np.sqrt(d)


# ------------------------------------------------- J2 energy correction (Sec. 3)
def delta_E(rho):
    r = r_of_rho(rho)
    r_z = Q[2] + rho * np.sin(DELTA)
    return MU * J2 * R_EARTH ** 2 / (2.0 * r ** 3) * (3.0 * (r_z / r) ** 2 - 1.0)


def rhodot_exact(rho, branch):
    d = np.maximum(Delta0_of_rho(rho) - 2.0 * delta_E(rho), 0.0)
    return -B + branch * np.sqrt(d)


def rho_edge_search():
    rhos = np.linspace(1.0, 20000.0, 400000)
    d = Delta0_of_rho(rhos)
    pos = np.where(d >= 0)[0]
    return rhos[pos[0]], rhos[pos[-1]]


# --------------------------------------------- state vector -> osc. elements
def elements_from_rho_rhodot(rho, rhodot):
    r_vec = Q + rho * U_HAT
    v_vec = QDOT + rhodot * U_HAT + rho * W_VEC

    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    energy = 0.5 * v2 - MU / r
    a = -MU / (2.0 * energy)

    e_vec = np.cross(v_vec, h_vec) / MU - r_vec / r
    e = np.linalg.norm(e_vec)

    incl = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    if n > 1e-10 and e > 1e-10:
        cos_w = np.dot(n_vec, e_vec) / (n * e)
        omega = np.arccos(np.clip(cos_w, -1.0, 1.0))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    elif e > 1e-10:
        omega = np.arctan2(e_vec[1], e_vec[0])
        if h_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    r_p = a * (1.0 - e)
    return a, e, incl, omega, r_p


# ------------------------------------------ general-omega delta r_p (Sec. 4.5)
def min_delta_rp(a, e, incl, omega, n_u=3600):
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    A = 1.5 * J2 * (R_EARTH ** 2 / a) * np.sin(incl) ** 2 * (e - 1.0)
    c1 = (3.0 * J2 * R_EARTH ** 2 / (8.0 * a ** 2)) * (4.0 - 5.0 * np.sin(incl) ** 2) * np.cos(omega)
    s1 = (3.0 * J2 * R_EARTH ** 2 / (8.0 * a ** 2)) * (4.0 - 7.0 * np.sin(incl) ** 2) * np.sin(omega)
    c3 = (7.0 * J2 * R_EARTH ** 2 / (8.0 * a ** 2)) * np.sin(incl) ** 2 * np.cos(omega)
    s3 = (7.0 * J2 * R_EARTH ** 2 / (8.0 * a ** 2)) * np.sin(incl) ** 2 * np.sin(omega)
    Bc, Bs, Cc, Cs = -a * c1, -a * s1, -a * c3, -a * s3
    vals = (A * np.cos(2 * u) + Bc * np.cos(u) + Bs * np.sin(u)
            + Cc * np.cos(3 * u) + Cs * np.sin(3 * u))
    return vals.min()


def rp_corrected(rho, rhodot):
    a, e, incl, omega, r_p = elements_from_rho_rhodot(rho, rhodot)
    return r_p + min_delta_rp(a, e, incl, omega)


def rp_uncorrected(rho, rhodot):
    return elements_from_rho_rhodot(rho, rhodot)[4]


# --------------------------------------------------- perigee-boundary roots
def find_perigee_boundary(rho, corrected, rhodot_lo, rhodot_hi, n_scan=500):
    """All roots in rhodot of r_p(rho, rhodot) [+correction] - R_MIN = 0,
    scanning strictly inside (rhodot_lo, rhodot_hi) to avoid the near-
    parabolic edges where osculating a diverges."""
    f = (lambda rd: rp_corrected(rho, rd) - R_MIN) if corrected \
        else (lambda rd: rp_uncorrected(rho, rd) - R_MIN)
    margin = 1e-4 * (rhodot_hi - rhodot_lo)
    grid = np.linspace(rhodot_lo + margin, rhodot_hi - margin, n_scan)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.array([f(x) for x in grid])
    roots = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            roots.append(brentq(f, grid[i], grid[i + 1], xtol=1e-10))
    return roots


def main():
    rho_lo, rho_hi = rho_edge_search()
    print(f"Unperturbed energy-bounded rho-extent: [{rho_lo:.2f}, {rho_hi:.2f}] km")

    # ----------------------------------------------------------- left panel
    rho_full = np.linspace(rho_lo, rho_hi, 4000)
    plus0 = rhodot0(rho_full, +1)
    minus0 = rhodot0(rho_full, -1)
    plus_exact = rhodot_exact(rho_full, +1)
    minus_exact = rhodot_exact(rho_full, -1)

    # --------------------------------------------------- right panel (zoom)
    rho_seg_lo, rho_seg_hi = 6088.0, 7385.0  # quoted in Sec. 6.1
    rho_seg = np.linspace(rho_seg_lo, rho_seg_hi, 250)

    unc_upper, unc_lower = [], []
    cor_upper, cor_lower = [], []
    for rho in rho_seg:
        rd_lo, rd_hi = rhodot0(rho, -1), rhodot0(rho, +1)
        r_unc = find_perigee_boundary(rho, False, rd_lo, rd_hi)
        r_cor = find_perigee_boundary(rho, True, rd_lo, rd_hi)
        unc_upper.append(max(r_unc) if r_unc else np.nan)
        unc_lower.append(min(r_unc) if r_unc else np.nan)
        cor_upper.append(max(r_cor) if r_cor else np.nan)
        cor_lower.append(min(r_cor) if r_cor else np.nan)
    unc_upper, unc_lower = np.array(unc_upper), np.array(unc_lower)
    cor_upper, cor_lower = np.array(cor_upper), np.array(cor_lower)

    valid = ~np.isnan(unc_upper) & ~np.isnan(cor_upper) \
        & ~np.isnan(unc_lower) & ~np.isnan(cor_lower)
    if valid.sum() < 2:
        print("WARNING: too few valid perigee-boundary points in this "
              "segment -- widen rho_seg or check the attributable.")
    shift_upper = np.abs(cor_upper[valid] - unc_upper[valid])
    shift_lower = np.abs(cor_lower[valid] - unc_lower[valid])
    print(f"Perigee-boundary shift in rhodot: "
          f"{np.nanmin(np.r_[shift_upper, shift_lower]):.4f} - "
          f"{np.nanmax(np.r_[shift_upper, shift_lower]):.4f} km/s")

    strip_area = trapezoid(shift_upper + shift_lower, rho_seg[valid])
    ar_total_area = trapezoid(np.maximum(plus0 - minus0, 0.0), rho_full)
    print(f"Reclassified-area fraction: {100 * strip_area / ar_total_area:.4f} %")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(rho_full, plus0, color="black", lw=1.5, label="uncorrected")
    ax.plot(rho_full, minus0, color="black", lw=1.5)
    ax.plot(rho_full, plus_exact, color="tab:red", lw=1.0, ls="--", label=r"$J_2$-corrected")
    ax.plot(rho_full, minus_exact, color="tab:red", lw=1.0, ls="--")
    ax.fill_between(rho_full, minus0, plus0, color="grey", alpha=0.15)
    ax.axvspan(rho_seg_lo, rho_seg_hi, color="tab:blue", alpha=0.08)
    ax.set_xlabel(r"$\rho$ (km)")
    ax.set_ylabel(r"$\dot\rho$ (km/s)")
    ax.set_title("Energy-constraint boundary\n(full admissible region)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(rho_seg[valid], unc_upper[valid], color="black", lw=1.5, label="uncorrected")
    ax.plot(rho_seg[valid], unc_lower[valid], color="black", lw=1.5)
    ax.plot(rho_seg[valid], cor_upper[valid], color="tab:red", lw=1.2, ls="--", label=r"$J_2$-corrected")
    ax.plot(rho_seg[valid], cor_lower[valid], color="tab:red", lw=1.2, ls="--")
    ax.set_xlabel(r"$\rho$ (km)")
    ax.set_ylabel(r"$\dot\rho$ (km/s)")
    ax.set_title("Perigee-constraint boundary\n(zoom)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/worked_example_boundaries.png", dpi=200)
    print("Saved figures/worked_example_boundaries.png")


if __name__ == "__main__":
    main()