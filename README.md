# Analytic First-Order J2 Corrections to the Two-Body Admissible-Region Boundary

This repository contains the Python source code and verification scripts accompanying the paper:

> **Analytic First-Order J2 Corrections to the Two-Body Admissible-Region Boundary for Too-Short-Arc Orbit Determination**  
> P. A. Nesvinga  
> *The Journal of the Astronautical Sciences* (under review)

The paper derives closed-form, first-order J2 perturbation corrections to both boundary curves (energy and perigee constraints) of the admissible-region method for too-short-arc initial orbit determination. All symbolic identities are verified via SymPy residual checks, and all numerical claims are reproduced by direct RK45 integration of the J2-perturbed equations of motion.

## Repository structure

```
.
├── src/
│   ├── symbolic_verification.py   # SymPy checks (Sec. II, III, IV)
│   ├── energy_boundary.py         # Energy-constraint numerics (Tables I-II)
│   ├── perigee_boundary.py        # Perigee-constraint numerics (Tables III-IV)
│   ├── general_omega.py           # General-omega derivation (Sec. IV.E, Fig. 1)
│   ├── zonal_validation.py        # J2-J6 validation (Table V)
│   └── figures.py                 # Generate all figures (Figs. 2-4)
├── figures/                       # Output directory for PNG figures
└── data/                          # Output directory for generated tables
```

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- SymPy >= 1.9
- Matplotlib >= 3.4

Install via:
```bash
pip install -r requirements.txt
```

## Reproducing the results

### 1. Symbolic verification (Sec. II, III, IV)
```bash
python src/symbolic_verification.py
```
Confirms:
- Kinematic identity (Eq. 4)
- Energy-boundary Taylor expansion (Eq. 6)
- General-omega reduction to omega=0 (Eq. 27 -> Eq. 19)

### 2. Energy constraint (Sec. III.C, Tables I-II)
```bash
python src/energy_boundary.py
```
Reproduces convergence-order and edge-breakdown tables.

### 3. Perigee constraint (Sec. V, Tables III-IV)
```bash
python src/perigee_boundary.py
```
Runs RK45 integration and compares analytic peak-to-peak amplitude against numerical measurement for LEO, MEO, and GEO cases.

### 4. General argument of perigee (Sec. IV.E, Fig. 1)
```bash
python src/general_omega.py
```
Generates `figures/margin_vs_omega.png` and reports worst-case margin at key omega values.

### 5. Zonal validation (Sec. V.E, Table V)
```bash
python src/zonal_validation.py
```
Compares J2-only vs J2-J6 zonal integration to confirm residual error is not dominated by neglected harmonics.

### 6. All figures (Figs. 2-4)
```bash
python src/figures.py
```
Generates:
- `figures/j2_secular_analysis.png` (20-period secular check)
- `figures/j2_rp_single_orbit.png` (short-period oscillation)
- `figures/margin_sweep_panels.png` (3-panel LEO parameter sweep)

## Key results

| Quantity | Value | Location |
|----------|-------|----------|
| Energy-constraint correction | First-order accurate, O(J2^2) residual | Sec. III, Eq. 6 |
| Perigee-constraint validity threshold | e ≳ 10 J2 (R⊕/a)^2 | Sec. V.F, Eq. 26 |
| LEO safety margin (worst-case) | ~10–12 km | Sec. VI, Table VI |
| Margin variation with ω | Factor ~1.7 (10.96 km → 6.37 km) | Sec. IV.E, Fig. 1 |
| Reclassified AR area (worked example) | 0.0029% | Sec. VI.A |

## Citation

If you use this code, please cite the paper:

```bibtex
@article{nesvinga2026j2,
  author  = {Nesvinga, P. A.},
  title   = {Analytic First-Order J2 Corrections to the Two-Body Admissible-Region Boundary for Too-Short-Arc Orbit Determination},
  journal = {The Journal of the Astronautical Sciences},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT License — see LICENSE file.
