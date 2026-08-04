#!/usr/bin/env python3
"""
symbolic_verification.py
Symbolic verification of key identities in the paper using SymPy.
Reproduces the checks described in Sec. II, III, and IV.
"""

import sympy as sp

# ---------------------------------------------------------------------------
# Sec. II: Kinematic identity
# ---------------------------------------------------------------------------
def verify_kinematic_identity():
    """Verify Eq. (4): |dot{r}|^2 decomposition."""
    # Define symbols
    rho, rhodot, alpha, delta, alphadot, deltadot = sp.symbols(
        'rho rhodot alpha delta alphadot deltadot', real=True
    )
    qx, qy, qz, qdx, qdy, qdz = sp.symbols('qx qy qz qdx qdy qdz', real=True)

    # Unit vector u(alpha, delta)
    u = sp.Matrix([
        sp.cos(delta)*sp.cos(alpha),
        sp.cos(delta)*sp.sin(alpha),
        sp.sin(delta)
    ])

    # Partials (non-unit)
    u_alpha = sp.diff(u, alpha)
    u_delta = sp.diff(u, delta)

    # Observer state
    q = sp.Matrix([qx, qy, qz])
    qdot = sp.Matrix([qdx, qdy, qdz])

    # Object state
    r = q + rho*u
    w = alphadot*u_alpha + deltadot*u_delta
    rdot = qdot + rhodot*u + rho*w

    # Compute |rdot|^2 directly
    speed2_direct = sp.simplify(rdot.dot(rdot))

    # Compute via Eq. (4)
    B = sp.simplify(qdot.dot(u))
    eta2 = sp.simplify(alphadot**2 * sp.cos(delta)**2 + deltadot**2)
    speed2_formula = (
        rhodot**2 + 2*B*rhodot + rho**2*eta2 
        + 2*rho*(qdot.dot(w)) + qdot.dot(qdot)
    )

    diff = sp.simplify(speed2_direct - speed2_formula)
    assert diff == 0, f"Kinematic identity residual: {diff}"
    print("[OK] Sec. II kinematic identity: residual = 0")

# ---------------------------------------------------------------------------
# Sec. III: Energy boundary expansion
# ---------------------------------------------------------------------------
def verify_energy_expansion():
    """Verify Eq. (6): first-order Taylor coefficient of sqrt(Delta0 - 2 deltaE)."""
    Delta0, deltaE = sp.symbols('Delta0 deltaE', positive=True, real=True)
    eps = sp.symbols('eps', real=True)

    expr = sp.sqrt(Delta0 - 2*eps*deltaE)
    series = sp.series(expr, eps, 0, 2)
    coeff = series.coeff(eps, 1)

    expected = -deltaE / sp.sqrt(Delta0)
    assert sp.simplify(coeff - expected) == 0
    print("[OK] Sec. III energy expansion: first-order coeff = -deltaE/sqrt(Delta0)")

# ---------------------------------------------------------------------------
# Sec. IV.E: General-omega reduction to omega=0
# ---------------------------------------------------------------------------
def verify_general_omega_reduction():
    """Verify that delta_e(u; omega=0) reduces to Eq. (19)."""
    u, omega, i, J2, Re, a = sp.symbols('u omega i J2 Re a', real=True, positive=True)
    si = sp.sin(i)

    # General omega formula (Eq. 27)
    delta_e_general = (
        J2*Re**2/(2*a**2) * (
            3*sp.cos(omega - u)
            - 2*si**2*sp.cos(omega)*sp.cos(u)
            - 7*si**2*sp.sin(u)**2*sp.cos(omega - u)
        )
    )

    # omega = 0 limit
    delta_e_0 = sp.simplify(delta_e_general.subs(omega, 0))

    # Published Eq. (19)
    delta_e_published = (
        J2*Re**2/(2*a**2) * sp.cos(u) * (
            3 - 2*si**2 - 7*si**2*sp.sin(u)**2
        )
    )

    diff = sp.simplify(delta_e_0 - delta_e_published)
    assert diff == 0, f"General-omega reduction residual: {diff}"
    print("[OK] Sec. IV.E: general-omega -> omega=0 reduction exact")

if __name__ == "__main__":
    verify_kinematic_identity()
    verify_energy_expansion()
    verify_general_omega_reduction()
    print("\nAll symbolic checks passed.")
