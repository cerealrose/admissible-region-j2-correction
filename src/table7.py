"""
Reproduces Table 7 — worst-case perigee-constraint safety margin |min_u δr_p(u)|
at ω = 0, via the exact cubic extremization of Sec. 4.4 (Eqs. 22-25).

This is the ω = 0 special case of the general-ω machinery in general_omega.py
(Sec. 4.5) — at ω = 0, π the degree-six polynomial (Eq. 30) collapses exactly
to this cubic, since Bs = Cs = 0.
"""
import numpy as np

J2 = 1.08262668e-3
Re = 6378.137  # km


def perigee_coeffs(a, e, i_deg):
    """A, B, C from Eqs. (23)-(25), omega = 0."""
    i = np.radians(i_deg)
    s2i = np.sin(i) ** 2
    A = 1.5 * J2 * Re**2 / a * s2i * (e - 1)
    B = (3 / 8) * J2 * Re**2 / a * (5 * s2i - 4)
    C = -(7 / 8) * J2 * Re**2 / a * s2i
    return A, B, C


def worst_case_margin(a, e, i_deg):
    """Exact cubic extremization (Sec. 4.4): interior stationary points of
    f(x) = 4Cx^3 + 2Ax^2 + (B-3C)x - A satisfy f'(x) = 12Cx^2+4Ax+(B-3C) = 0;
    candidates are those roots (if in [-1,1]) plus the endpoints x = ±1.
    Returns |min_u δr_p(u)|.
    """
    A, B, C = perigee_coeffs(a, e, i_deg)

    def drp_of_x(x):
        # delta_r_p(u) with x = cos(u): cos(2u)=2x^2-1, cos(3u)=4x^3-3x
        return A * (2 * x**2 - 1) + B * x + C * (4 * x**3 - 3 * x)

    candidates = [-1.0, 1.0]
    if abs(C) > 1e-300:
        roots = np.roots([12 * C, 4 * A, (B - 3 * C)])
        candidates += [r.real for r in roots if abs(r.imag) < 1e-9 and -1 <= r.real <= 1]
    else:
        # C == 0 (e.g. equatorial, sin^2 i = 0): f'(x) is linear
        if A != 0:
            x_star = -(B - 3 * C) / (4 * A)
            if -1 <= x_star <= 1:
                candidates.append(x_star)

    values = [drp_of_x(x) for x in candidates]
    return -min(values)  # magnitude of the most negative excursion


if __name__ == "__main__":
    rows = [
        (7000, 0.010, 45.0),
        (7000, 0.020, 45.0),
        (7000, 0.010, 63.4),
        (7200, 0.015, 98.0),
    ]
    print(f"{'a (km)':>7} {'e':>7} {'i (deg)':>8}  margin (km)")
    for a, e, i_deg in rows:
        print(f"{a:7.0f} {e:7.3f} {i_deg:8.1f}  {worst_case_margin(a, e, i_deg):10.3f}")