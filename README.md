# Analytic First-Order J2 Corrections to the Two-Body Admissible-Region Boundary

This repository contains the Python source code and verification scripts accompanying the paper:

> **Perturbation-Aware Admissible Regions: First-Order J2 Corrections to Too-Short-Arc Boundary Construction**
> P. A. Nesvinga
> *The Journal of the Astronautical Sciences* (under review)

The paper derives closed-form, first-order J2 perturbation corrections to both boundary curves (energy and perigee constraints) of the admissible-region method for too-short-arc initial orbit determination. All symbolic identities are verified via SymPy residual checks, and all numerical claims are reproduced by direct RK45 integration of the J2-perturbed equations of motion.

## Repository structure

```
.
├── src/
│   ├── symbolic_verification.py   # SymPy checks (Sec. 2, 3, 4)
│   ├── energy_boundary.py         # Energy-constraint numerics (Tables 1-2)
│   ├── perigee_boundary.py        # Perigee-constraint numerics (Tables 3-4)
│   ├── general_omega.py           # General-omega derivation (Sec. 4.5, Fig. 1)
│   ├── zonal_validation.py        # J2-J6 validation (Table 6)
│   ├── worked_example.py          # Generate synthetic-attributable boundaries (Sec. 6.1, Fig. 4)
│   └── figures.py                 # Generate figures (Figs. 2,3,5)
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

### 1. Symbolic verification (Sec. 2, 3, 4)
```bash
python src/symbolic_verification.py
```
Confirms:
- Kinematic identity (Eq. 5)
- Energy-boundary Taylor expansion (Eq. 12)
- General-omega reduction to omega=0 (Eq. 26 -> Eq. 19)

### 2. Energy constraint (Sec. 3.3, Tables 1-2)
```bash
python src/energy_boundary.py
```
Reproduces convergence-order and edge-breakdown tables.

### 3. Perigee constraint (Sec. 5, Tables 3-4)
```bash
python src/perigee_boundary.py
```
Runs RK45 integration and compares analytic peak-to-peak amplitude against numerical measurement for LEO, MEO, and GEO cases.

### 4. General argument of perigee (Sec. 4.5, Fig. 1)
```bash
python src/general_omega.py
```
Generates `figures/margin_vs_omega.png` and reports worst-case margin at key omega values.

### 5. Zonal validation (Sec. 5.5, Table 6)
```bash
python src/zonal_validation.py
```
Compares J2-only vs J2-J6 zonal integration to confirm residual error is not dominated by neglected harmonics.

### 6. Figures 2, 3, 5
```bash
python src/figures.py
```
Generates:
- `figures/j2_secular_analysis.png` (Fig. 2 — 20-period secular check)
- `figures/j2_rp_single_orbit.png` (Fig. 3 — short-period oscillation)
- `figures/margin_sweep_panels.png` (Fig. 5 — 3-panel LEO parameter sweep)

### 7. Worked example (Sec. 6.1, Fig. 4)
```bash
python src/worked_example.py
```
Reproduces the synthetic-attributable boundaries: the full admissible
region under the energy constraint (left panel) and a zoom on the
perigee-constraint boundary segment (right panel), uncorrected vs.
J2-corrected.

## Key results

| Quantity | Value | Location |
|----------|-------|----------|
| Energy-constraint correction | First-order accurate, O(J2^2) residual | Sec. 3, Eq. 12 |
| Perigee-constraint validity threshold | e ≳ 10 J2 (R⊕/a)^2 | Sec. 5.6, Eq. 31 |
| LEO safety margin (worst-case) | ~10–12 km | Sec. 6, Table 7 |
| Margin variation with ω | Factor ~1.7 (10.96 km → 6.37 km) | Sec. 4.5, Fig. 1 |
| Reclassified AR area (worked example) | 0.0029% | Sec. 6.1 |

## Citation

If you use this code, please cite the paper:

```bibtex
@article{nesvinga2026j2,
  author  = {Nesvinga, P. A.},
  title   = {Perturbation-Aware Admissible Regions: First-Order J2 Corrections to Too-Short-Arc Boundary Construction},
  journal = {The Journal of the Astronautical Sciences},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT License — see LICENSE file.
