# mshqc

**A hybrid C++20 / Python quantum chemistry engine for $\mathcal{O}(V^4)$-memory-free orbital-optimized Møller–Plesset perturbation theory**

[![PyPI version](https://img.shields.io/pypi/v/mshqc.svg)](https://pypi.org/project/mshqc/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

`mshqc` is a high-performance electronic structure library implementing Hartree–Fock (`RHF`, `UHF`, `ROHF`), canonical and semicanonical Møller–Plesset perturbation theory (`RMP2`, `UMP2`, `ROMP2`, `RMP3`, `UMP3`, `ROMP3`), and orbital-optimized perturbation theory (`OMP2`, `SCS-OMP2`, `SOS-OMP2`, `OMP2.5`, `OMP3`) with analytical nuclear gradients, geometry optimization, and harmonic vibrational frequencies.

All solvers share a unified integral engine supporting exact four-center ERIs (`"exact"`), direct full-pivoted Cholesky decomposition (`"cholesky"`), and packed lower-triangular Density Fitting (`"df"`), paired with [TBLIS](https://github.com/devinamatthews/tblis) zero-copy tensor contractions and a matrix-free Steihaug–Toint Trust-Region SOSCF (`TR-SOSCF`) orbital optimizer.

---

## Three-Tier Hybrid Architecture

`mshqc` separates high-performance C++20 kernels from readable Python method orchestration so users get one-line convenience while researchers can prototype new correlated methods without recompiling C++:

1. **Tier 1 — High-Level Pipeline API (`mshqc.run`, `MSHQCCalculator`, `MSHQCSession`):** Run complete closed-shell or open-shell workflows (`RHF`/`UHF`/`ROHF` -> `MP2`/`MP3` -> `OMP2`/`SCS-OMP2`/`SOS-OMP2`/`OMP2.5`/`OMP3`), geometry optimization, and frequency analysis in one call with automatic 3-index factor reuse (`SCFResult.L_mat`).
2. **Tier 2 — Modular Python Method Layer (`mshqc.methods`):** Clean, inspectable Python orchestration for RHF/UHF orbital optimization (`run_df_omp_rhf`, `run_df_omp_uhf`), 4D `NumPy`/`einsum` tensor converters (`t2_to_4d`, `b3c_to_3d`), custom method plugins (`optimize_orbitals_rhf`), and stateless interoperability (`run_from_arrays`, `run_from_pyscf`).
3. **Tier 3 — Coarse-Grained C++20 HPC Kernels (`mshqc.kernels`):** GIL-free OpenMP/BLAS/TBLIS primitives for packed 3-index DF/Cholesky contractions (`compute_df_ao_factor`, `build_jk_3c_*`, `transform_3c_mo`), adaptive $\mathcal{O}(V^4)$-memory-free particle-particle ladder contractions, chunked $Y^{\text{ooov}}$ Z-vectors, and MO-space DF Hessian-vector products.

---

## Installation

### Option 1: Precompiled Wheels from PyPI (Linux, macOS, Windows; Python 3.10+)

```bash
pip install mshqc
```

### Basis Set Data (`mshqc-basis`)

Gaussian basis sets (`.gbs`) are maintained in the companion repository [`syahrulhidayat/mshqc-basis`](https://github.com/syahrulhidayat/mshqc-basis) to keep the core repository lightweight. Precompiled PyPI wheels bundle the standard basis files automatically; when building from source, clone `mshqc-basis` and point `MSHQC_BASIS_PATH` to its `data/basis` directory (or copy into `data/basis/`).

### Option 2: Build from Source (Conda / Micromamba + CMake)

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
O 0.0000  0.0000  0.1173
H 0.0000  0.7572 -0.4692
H 0.0000 -0.7572 -0.4692
"""

results = mshqc.run(
    h2o,
    method=["RHF", "OMP2", "SCS-OMP2", "OMP2.5", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",       # "exact", "cholesky", or "df"
    units="angstrom",
    threshold=1e-9,
)

for name, info in results.items():
    print(f"{name:<9}: {info['energy']:.10f} Ha ({info['time']:.3f} s)")
```

### 2. Prototyping a Custom Orbital-Optimized Method in Python (`mshqc.methods`)

```python
import mshqc
from mshqc import kernels as K, methods as M

mol = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
basis = mshqc.BasisSet("cc-pVDZ", mol)
ints = mshqc.IntegralEngine(mol, basis)
scf_res = mshqc.RHF(mol, basis, ints, mshqc.SCFConfig(eri_method="df", threshold=1e-10)).compute()

no = scf_res.n_occ_alpha
nv = basis.n_basis_functions() - no
H_core = ints.compute_kinetic() + ints.compute_nuclear()

def custom_scs_omp2(B_oo, B_ov, B_vv, eps_o, eps_v):
    e_os, e_ss, T2 = K.eval_rmp2_df(B_ov, eps_o, eps_v)
    g_oo, g_vv = K.eval_omp2_1rdm_rhf(T2, no, nv, c_os=1.2, c_ss=1.0 / 3.0)
    Z_vo = K.eval_omp2_zvec_rhf_df(T2, B_oo, B_ov, B_vv, no, nv, c_os=1.2, c_ss=1.0 / 3.0)
    return 1.2 * e_os + (1.0 / 3.0) * e_ss, g_oo, g_vv, Z_vo

res = M.optimize_orbitals_rhf(
    H_core, mol.nuclear_repulsion_energy(), scf_res.L_mat, scf_res.C_alpha, no,
    corr_fn=custom_scs_omp2, method_name="custom-scs-omp2",
)
print(f"Converged in {res.iterations} iterations: {res.energy_total:.10f} Ha")
```

### 3. Stateless Interoperability with PySCF (`run_from_pyscf`)

```python
from pyscf import gto, scf
import mshqc.methods as M

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="cc-pVTZ")
mf = scf.RHF(mol).density_fit(auxbasis="cc-pVTZ-RI").run()
res = M.run_from_pyscf(mf, method="omp2.5")
print(f"PySCF -> mshqc DF-OMP2.5: {res.energy_total:.10f} Ha ({res.iterations} it)")
```

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for open-shell radicals (`UHF`/`ROHF`/`ROMP2`/`ROMP3`), 4D `np.einsum` tensor helpers, geometry optimization, and HPC auto-profiling (`recommend_hpc_settings`).

---

## Testing

```bash
pytest tests/test_mshqc.py tests/test_hybrid_phase1.py tests/test_hybrid_phase2.py tests/test_hybrid_phase3.py tests/test_hybrid_phase4.py -v
```

---

## Citing mshqc

```bibtex
@software{Hidayat2026mshqc,
  author  = {Hidayat, Muhamad Syahrul},
  title   = {mshqc: A Hybrid C++20/Python Quantum Chemistry Engine for O(V^4)-Memory-Free Orbital-Optimized M{\o}ller--Plesset Perturbation Theory},
  year    = {2026},
  version = {1.0.4.post22},
  url     = {https://github.com/syahrulhidayat/mshqc},
  note    = {PyPI: https://pypi.org/project/mshqc/ | ORCID: 0009-0005-4112-1807}
}
```

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.
