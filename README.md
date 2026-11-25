# mshqc

**A Pure Thin-Core C++20 / Python Quantum Chemistry Engine for $\mathcal{O}(V^4)$-Memory-Free Orbital-Optimized Møller–Plesset Perturbation Theory**

[![PyPI version](https://img.shields.io/pypi/v/mshqc.svg)](https://pypi.org/project/mshqc/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release CI](https://github.com/syahrulhidayat/mshqc/actions/workflows/release_public.yml/badge.svg)](https://github.com/syahrulhidayat/mshqc/actions/workflows/release_public.yml)

---

## Overview

`mshqc` v2.0 is a lean, high-performance electronic structure library built on an **irreducible 9-file C++20 kernel core** (~5,500 LOC, zero HDF5 dependency) paired with clean, modular Python orchestration and pluggable compute backends:

- **Unified ERI Modes:** Density Fitting (`"df"`), full-pivoted Cholesky decomposition (`"cholesky"`), and exact 4-center ERIs (`"exact"`, powered by TBLIS 4D tensor contractions).
- **SCF References & Symmetry:** `RHF`, `UHF`, `ROHF` featuring Pure-Python SAD guess + zero-GEMM diagonal Iter-0 exchange, diagonal-equilibrated Pulay DIIS, level-shifting, Pure-Python Abelian point-group symmetry detection & MO irrep labeling (`C2v`, `D2h`, etc.), `.npz` orbital checkpoints, and zero-overhead telemetry callbacks.
- **Canonical / Semicanonical MP:** `RMP2`, `UMP2`, `ROMP2`, `RMP3`, `UMP3`, `ROMP3` (all-electron & strict frozen-core).
- **Orbital-Optimized Perturbation Theory (TR-SOSCF):**
  - `OMP2`, `SCS-OMP2`, `SOS-OMP2`, `OMP2.5`, `OMP3` (RHF & UHF, all-electron & frozen-core)
  - **`kappa-OOMP2`** ($\kappa$-regularized OMP2 with degeneracy-safe L'Hôpital orbital gradients & exact analytical nuclear gradients)
  - **`LT-SOS-OMP2`** ($\mathcal{O}(N^4)$ zero-$T_2$ Laplace-Transformed SOS-OMP2 powered by OpenMP C++20 quadrature kernels)
- **Dynamic Compute Backends (`mshqc.backends`):** Switch transparently between the default C++20/TBLIS backend, pure NumPy (`use_backend("numpy")`), or PyTorch FP64 GPU/CPU acceleration (`enable_torch_backend(device="auto")`, supporting `RHF`, `LT-SOS-OMP2`, and `OMP3`; includes a free Colab/Kaggle runner in `benchmarks/colab_kaggle_gpu_runner.py`).
- **Properties, Gradients & Visualization:** Analytical DF nuclear gradients (`DF-RHF`, `DF-UHF`, `DF-ROHF`, `DF-OMP2`, `DF-SCS-OMP2`, `DF-SOS-OMP2`, `DF-kappa-OOMP2`), geometry optimization, $\mathcal{O}(N)$ harmonic vibrational frequencies + RRHO thermochemistry (all-electron & frozen-core), 1D PES bond scans, Natural Orbitals/NOONs, Mulliken/Löwdin charges, `.molden` export, and one-line presets (`mshqc.molecule("H2O")`, `mshqc.atom("O")`).

> **Thesis & v1.0 Monolithic Archive:** The original monolithic C++20 implementation with legacy C++ `PetiteList` symmetry and HDF5 out-of-core streaming is permanently archived under git tag `v1.0.5-thesis-archive` (and branch `test`) for full reproducibility of thesis benchmarks.

---

## Installation

### Option 1: Precompiled Wheels from PyPI (Linux, macOS ARM64, Windows x64; Python 3.10+)

```bash
pip install mshqc
```

### Basis Set Data (`mshqc-basis`)

Gaussian basis sets (`.gbs`) are bundled inside official PyPI wheels and maintained in [`syahrulhidayat/mshqc-basis`](https://github.com/syahrulhidayat/mshqc-basis). When building from source, clone `mshqc-basis` and point `MSHQC_BASIS_PATH` to its `data/basis` directory (or copy into `data/basis/`).

### Option 2: Build from Source (~25–40 s, Zero-HDF5)

```bash
git clone [https://github.com/syahrulhidayat/mshqc.git](https://github.com/syahrulhidayat/mshqc.git)
cd mshqc
conda env create -f environment.yml
conda activate mshqc_314
pip install --no-build-isolation --no-deps -e . -v
```

---

## Quickstart

### 1. One-Line Multi-Method Pipeline (`mshqc.run` & Presets)

```python
import mshqc

mol = mshqc.molecule("H2O")  # or pass XYZ string / .xyz file / list of tuples
results = mshqc.run(
    mol,
    method=["RHF", "OMP2", "SCS-OMP2", "LT-SOS-OMP2", "kappa-OMP2", "OMP2.5", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",  # "df", "cholesky", or "exact"
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
vib = mol.frequencies(method="omp2", basis="cc-pVDZ", freeze_core=True)
print(f"R_OH = {opt.bond_lengths_angstrom['O1-H2']:.6f} A | Freqs = {vib.frequencies_cm1}")
```

### 3. Dynamic Backends (NumPy / PyTorch CUDA FP64) & Abelian Symmetry

```python
import mshqc
from mshqc import methods as M
from mshqc.backends import enable_torch_backend, use_backend

mol = M.align_principal_axes(mshqc.molecule("H2O"))
sym = M.detect_abelian_symmetry(mol)
calc = mshqc.MSHQCCalculator(mol, basis_name="cc-pVDZ", eri_method="df")

# Switch to PyTorch CUDA FP64 (with automatic fallback to C++20 kernels)
enable_torch_backend(device="auto", name="torch", fallback_to_cpp=True)
with use_backend("torch"):
    scf_res = calc.run_scf()
    omp3_res = calc.run_omp3(scf_res, grad_thresh=1e-6)

labels = M.label_mo_irreps(mol, calc.basis, scf_res.C_alpha, calc._get_s_ovlp(), sym)
print(f"Point group: {sym.point_group} | Occ irreps: {labels[:5]} | E(OMP3)={omp3_res.energy_total:.8f} Ha")
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

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for open-shell radicals (`UHF`/`ROHF`/`ROMP2`/`ROMP3`), `.molden` export, `.npz` checkpoints, and custom `optimize_orbitals_*` hooks.

---

## Testing & Benchmarks

```bash
# Run full v2.0 verification suite (Stages 3–12, symmetry, backends, exact/DF/Cholesky parity)
pytest tests/ -v

# Quick universal & Bab IV benchmark runners
python benchmarks/run_universal_v2.py --tier quick --basis cc-pVDZ --extended
python benchmarks/run_bab4_v2.py --quick --skip-ram
```

---

## Citing mshqc

Official releases are archived on Zenodo via automated GitHub Actions release workflows.

```bibtex
@software{Hidayat2026mshqc,
  author  = {Hidayat, Muhamad Syahrul},
  title   = {mshqc: A Pure Thin-Core C++20/Python Quantum Chemistry Engine for O(V^4)-Memory-Free Orbital-Optimized M{\o}ller--Plesset Perturbation Theory},
  year    = {2026},
  version = {2.0.0},
  url     = {[https://github.com/syahrulhidayat/mshqc](https://github.com/syahrulhidayat/mshqc)},
  note    = {PyPI: [https://pypi.org/project/mshqc/](https://pypi.org/project/mshqc/) | ORCID: 0009-0005-4112-1807}
}
```

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.
