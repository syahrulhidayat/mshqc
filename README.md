# mshqc

**A Pure Thin-Core C++20 / Python Quantum Chemistry Engine for $\mathcal{O}(V^4)$-Memory-Free Orbital-Optimized Møller–Plesset Perturbation Theory**

[![PyPI version](https://img.shields.io/pypi/v/mshqc.svg)](https://pypi.org/project/mshqc/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

`mshqc` v2.0 is a lean, high-performance electronic structure library built on an **irreducible 8-file C++20 kernel core** (~5,500 LOC, zero HDF5 dependency) paired with clean, modular Python method orchestration:

- **SCF References:** `RHF`, `UHF`, `ROHF` (Density Fitting `"df"` and full-pivoted `"cholesky"`), featuring Pure-Python SAD guess + zero-GEMM diagonal Iter-0 exchange, diagonal-equilibrated Pulay DIIS, level-shifting, `.npz` orbital checkpoints, and zero-overhead telemetry callbacks.
- **Canonical / Semicanonical MP:** `RMP2`, `UMP2`, `ROMP2`, `RMP3`, `UMP3`, `ROMP3` (all-electron & strict frozen-core).
- **Orbital-Optimized Perturbation Theory (TR-SOSCF):**
  - `OMP2`, `SCS-OMP2`, `SOS-OMP2`, `OMP2.5`, `OMP3` (RHF & UHF, all-electron & frozen-core)
  - **`kappa-OOMP2`** ($\kappa$-regularized OMP2 with degeneracy-safe L'Hôpital orbital gradients & exact analytical nuclear gradients)
  - **`LT-SOS-OMP2`** ($\mathcal{O}(N^4)$ zero-$T_2$ Laplace-Transformed SOS-OMP2 powered by OpenMP C++20 quadrature kernels)
- **Properties, Gradients & Visualization:** Analytical DF nuclear gradients (`DF-RHF`, `DF-UHF`, `DF-ROHF`, `DF-OMP2`, `DF-SCS-OMP2`, `DF-SOS-OMP2`, `DF-kappa-OOMP2`), geometry optimization, $\mathcal{O}(N)$ harmonic vibrational frequencies + RRHO thermochemistry, 1D PES bond scans, Natural Orbitals/NOONs, Mulliken/Löwdin charges, and `.molden` export.

> **Thesis & v1.0 Monolithic Archive:** The original monolithic C++20 implementation with exact 4-center ERIs (`eri_method="exact"`), Abelian `PointGroup`/`PetiteList` symmetry, and HDF5 out-of-core streaming is permanently archived under git tag `v1.0.5-thesis-archive` (and branch `test`) for full reproducibility of thesis benchmarks.

---

## Installation

### Option 1: Precompiled Wheels from PyPI (Linux, macOS, Windows; Python 3.10+)

```bash
pip install mshqc
```

### Basis Set Data (`mshqc-basis`)

Gaussian basis sets (`.gbs`) live in [`syahrulhidayat/mshqc-basis`](https://github.com/syahrulhidayat/mshqc-basis). When building from source, clone `mshqc-basis` and point `MSHQC_BASIS_PATH` to its `data/basis` directory (or copy into `data/basis/`).

### Option 2: Build from Source (~25–40 s, Zero-HDF5)

```bash
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
conda env create -f environment.yml
conda activate mshqc_314
pip install --no-build-isolation --no-deps -e . -v
```

---

## Quickstart

### 1. One-Line Multi-Method Pipeline (`mshqc.run`)

```python
import mshqc

h2o = """
O 0.0000 0.0000 0.1173
H 0.0000 0.7572 -0.4692
H 0.0000 -0.7572 -0.4692
"""

results = mshqc.run(
    h2o,
    method=["RHF", "OMP2", "SCS-OMP2", "LT-SOS-OMP2", "kappa-OMP2", "OMP2.5", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",  # "df" or "cholesky"
    units="angstrom",
    threshold=1e-9,
)

for name, info in results.items():
    print(f"{name:<12}: {info['energy']:.10f} Ha ({info['time']:.3f} s)")
```

### 2. Geometry Optimization, Frequencies & Analytic `kappa-OOMP2` Gradients

```python
import mshqc

mol = mshqc.build_molecule(
    "O 0 0 0.125\nH 0 0.780 -0.490\nH 0 -0.780 -0.490",
    units="angstrom",
)
opt = mol.optimize_geometry(method="kappa-omp2", basis="cc-pVDZ", g_convergence="gau_tight")
vib = mol.frequencies(method="omp2", basis="cc-pVDZ")
print(f"R_OH = {opt.bond_lengths_angstrom['O1-H2']:.6f} A | Freqs = {vib.frequencies_cm1}")
```

### 3. Natural Orbitals, Charges, `.molden` Export & `.npz` Checkpoints

```python
import mshqc
from mshqc import kernels as K, methods as M

mol = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
basis = mshqc.BasisSet("cc-pVDZ", mol)
aux = mshqc.BasisSet("cc-pVDZ-RI", mol)
ints = mshqc.IntegralEngine(mol, basis)

H = ints.compute_kinetic() + ints.compute_nuclear()
S = ints.compute_overlap()
B_ao = K.compute_df_ao_factor(mol, basis, aux, 1e-10)
scf_res = M.run_df_rhf(H, S, mol.nuclear_repulsion_energy(), B_ao, 5, P_init=M.build_sad_guess(mol, basis))

P_ao = M.scf_density_ao(scf_res)
no_res = M.compute_natural_orbitals(P_ao, S)
pop = M.mulliken_and_lowdin_charges(mol, basis, S, P_ao)
M.save_orbital_checkpoint("h2o_rhf.npz", scf_res)
M.write_molden("h2o.molden", mol, basis, scf_res.C_alpha, scf_res.orbital_energies_alpha, no_res.occupations)
print("Lowdin charges:", pop.lowdin_charges)
```

### 4. Stateless Interoperability with PySCF (`run_from_pyscf`)

```python
from pyscf import gto, scf
import mshqc.methods as M

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="cc-pVTZ")
mf = scf.RHF(mol).density_fit(auxbasis="cc-pVTZ-RI").run()
res = M.run_from_pyscf(mf, method="kappa-omp2")
print(f"PySCF -> mshqc DF-kappa-OOMP2: {res.energy_total:.10f} Ha ({res.iterations} it)")
```

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for open-shell radicals (`UHF`/`ROHF`/`ROMP2`/`ROMP3`), custom `optimize_orbitals_*` hooks, and HPC auto-profiling (`recommend_hpc_settings`).

---

## Testing

```bash
pytest tests/test_thin_core_stage3.py tests/test_thin_core_stage4.py tests/test_thin_core_stage5.py tests/test_thin_core_stage6.py tests/test_thin_core_stage7.py tests/test_thin_core_stage8_parity.py tests/test_thin_core_stage9_v21.py -v -s
```

---

## Citing mshqc

```bibtex
@software{Hidayat2026mshqc,
  author  = {Hidayat, Muhamad Syahrul},
  title   = {mshqc: A Hybrid C++20/Python Quantum Chemistry Engine for O(V^4)-Memory-Free Orbital-Optimized M{\o}ller--Plesset Perturbation Theory},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/syahrulhidayat/mshqc},
  note    = {PyPI: https://pypi.org/project/mshqc/ | ORCID: 0009-0005-4112-1807}
}
```

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.
