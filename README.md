# mshqc

**A C++20 quantum chemistry engine for orbital-optimized Møller–Plesset perturbation theory**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

`mshqc` is a high-performance quantum chemistry package that implements a focused set of wave-function-based electronic structure methods — Hartree–Fock (`RHF`, `UHF`, `ROHF`), canonical Møller–Plesset perturbation theory (`RMP2`, `UMP2`, `RMP3`, `UMP3`), and orbital-optimized Møller–Plesset perturbation theory (`OMP2`, `OMP3`) — in a compact C++20 core with a `nanobind` Python interface.

All implemented solvers share a unified integral engine supporting exact four-center Electron Repulsion Integrals (`"exact"`), pivot-based Cholesky decomposition (`"cholesky"`), and auxiliary-basis Density Fitting / Resolution of the Identity (`"df"`), coupled with [TBLIS](https://github.com/devinamatthews/tblis) tensor contractions and a matrix-free Steihaug–Toint trust-region Second-Order SCF (`TrustRegionSOSCF`) orbital optimizer.

---

## Statement of Need

Mature quantum chemistry ecosystems such as Psi4 and PySCF provide broad method coverage, but their size and deep abstraction layers create a steep barrier to entry for researchers and students prototyping new orbital-optimization algorithms, three-index tensor factorizations, or custom contraction schedules. Conversely, lightweight educational codes typically implement only canonical RHF and RMP2 in pure Python, lacking open-shell references, third-order correlation (`MP3`/`OMP3`), point-group symmetry reduction, or rank-reduced three-index integrals.

`mshqc` bridges this gap by exposing a compact, inspectable C++20 codebase under a clean two-tier Python API:
1. **High-Level Pipeline API (`mshqc.run`, `MSHQCSession`, `quick_calculation`):** Run complete multi-method workflows (`SCF -> MP2 -> OMP2 -> OMP3`) in a single call with automatic point-group alignment (`PointGroup`), `PetiteList` symmetry reduction, and matched `-RI` auxiliary basis resolution.
2. **Low-Level C++20 Kernel API (`Molecule`, `BasisSet`, `IntegralEngine`, `RHF`/`UHF`/`ROHF`, `RMP2`/`UMP2`/`OMP2`, `RMP3`/`UMP3`/`OMP3`):** Instantiate C++20 solvers directly to inspect or modify intermediate matrices, orbital energies, and Cholesky/DF vectors with zero-copy `Eigen` <-> `NumPy` interoperability.

---

## Features

### Core Infrastructure & Symmetry
- Molecular geometry initialization in Bohr or Angstrom — `Molecule`, `Atom`, `build_molecule` [1]
- Bundled library of 538 Gaussian basis sets (`.gbs`), including Dunning `cc-pVXZ`, `aug-cc-pVXZ`, Karlsruhe `def2`, and matched `-RI` / `-JKFIT` auxiliary basis sets — `BasisSet` [2, 7]
- Automated point-group detection, canonical axis alignment, and symmetry-unique shell-pair generation — `PointGroup`, `PetiteList`, `BasisSymmetrizer` [3]

### Unified Integral Engine
- Analytical one- and two-electron integrals via `libcint` — `IntegralEngine` [4]
- Unified `eri_method` selection across all SCF and correlated solvers:
  - `"exact"`: Four-center ERIs with Schwarz inequality prescreening — `Screening` [8]
  - `"cholesky"`: Pivot-based three-index Cholesky decomposition — `CholeskyERI` [5, 6]
  - `"df"`: Auxiliary-basis three-index Density Fitting (RI) with optional HDF5 out-of-core storage [7, 21]

### Self-Consistent Field (SCF)
- Restricted (`RHF`), Unrestricted (`UHF`), and Restricted Open-Shell (`ROHF`) Hartree–Fock [9, 10, 11]
- Superposition of Atomic Densities (`SAD`) initial guess and Pulay `DIIS` extrapolation [12, 13]

### Møller–Plesset Perturbation Theory & Orbital Optimization
- Canonical second- and third-order perturbation theory: `RMP2`, `UMP2`, `RMP3`, `UMP3` [14, 15, 17]
- Orbital-optimized second- and third-order perturbation theory: `OMP2`, `OMP3` driven by matrix-free Steihaug–Toint trust-region `SOSCF` (`TrustRegionSOSCF`) and Z-vector response densities [16, 18, 19]

---

## Prerequisites & Dependencies

| Category | Packages | Scope | Reference |
|---|---|---|---|
| Linear algebra & tensors | `eigen`, `tblis`, `mkl`, `mkl-include`, `liblapacke` | Build + runtime | [20, 22] |
| Integrals & I/O | `libcint`, `hdf5`, `jemalloc` | Build + runtime | [4, 21] |
| Parallelism | `llvm-openmp` / `libgomp` | Build + runtime | [23] |
| Python bindings & runtime | `nanobind`, `numpy`, `scipy` | Build + runtime | [24, 25] |
| Testing | `pytest` | Dev / test | — |

---

## Installation

A `conda-forge` environment is recommended to ensure ABI-compatible builds of MKL, HDF5, OpenMP, `libcint`, and `tblis`.

```bash
# 1. Create and activate the Conda environment
conda env create -f environment.yml
conda activate mshqc_314

# 2. Build and install mshqc in editable mode
pip install -v -e . --no-build-isolation
```

### Build Options

By default, `mshqc` compiles with `-O3 -march=native` on local machines. When building portable binaries for heterogeneous clusters, override the C++ flags via `CMAKE_ARGS`:

```bash
CMAKE_ARGS="-DCMAKE_CXX_FLAGS='-O3 -mavx2 -mfma -DNDEBUG'" pip install -v -e . --no-build-isolation
```

---

## Quickstart

### 1. High-Level One-Liner API (`mshqc.run`)

`mshqc.run()` automatically detects molecular point-group symmetry, builds the `PetiteList`, resolves the auxiliary basis (`cc-pVTZ-RI` for `eri_method="df"`), and executes the requested method hierarchy in compiled C++20:

```python
import mshqc

h2o_xyz = """
O  0.0000000000   0.0000000000   0.1173000000
H  0.0000000000   0.7572000000  -0.4692000000
H  0.0000000000  -0.7572000000  -0.4692000000
"""

results = mshqc.run(
    h2o_xyz,
    method=["RHF", "RMP2", "OMP2", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",
    units="angstrom",
    threshold=1e-9,
)

for method_name, info in results.items():
    print(f"{method_name:<5} : {info['energy']:.10f} Ha ({info['time']:.3f} s)")
```

### 2. Low-Level C++20 Kernel API

For method development and direct inspection of intermediate quantities, instantiate the C++20 classes directly:

```python
import mshqc

# 1. Initialize molecule (supports atomic symbols/numbers and "angstrom" or "bohr" units)
mol = mshqc.Molecule(charge=0, multiplicity=1)
mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

# 2. Load basis set and analytical integral engine
basis     = mshqc.BasisSet("cc-pVTZ", mol)
integrals = mshqc.IntegralEngine(mol, basis)

# 3. Run RHF with Cholesky-decomposed ERIs (threshold = 1e-9 Ha)
scf_conf = mshqc.SCFConfig(eri_method="cholesky", threshold=1e-9)
scf_res  = mshqc.RHF(mol, basis, integrals, scf_conf).compute()
print(f"RHF  Total Energy : {scf_res.energy_total:.10f} Ha")

# 4. Run RMP2, OMP2, and OMP3 on the converged SCF reference
mp2_conf = mshqc.MP2Config(eri_method="cholesky", threshold=1e-9, gradient_threshold=1e-5)

rmp2_res = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
omp2_res = mshqc.OMP2(mol, basis, integrals, scf_res, mp2_conf).compute()
omp3_res = mshqc.OMP3(mol, basis, integrals, scf_res, mp2_conf).compute()

print(f"RMP2 Total Energy : {rmp2_res.energy_total:.10f} Ha (Corr: {rmp2_res.energy_mp2_corr:.10f} Ha)")
print(f"OMP2 Total Energy : {omp2_res.energy_total:.10f} Ha")
print(f"OMP3 Total Energy : {omp3_res.energy_total:.10f} Ha")
```

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for open-shell (`UHF`/`ROHF`/`UMP2`/`UMP3`), symmetry-adapted, and session-managed examples, and [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for build and runtime diagnostics.

---

## Testing

Run the automated validation test suite via `pytest`:

```bash
pytest tests/ -v
```

---

## Contributing

Bug reports and pull requests are welcome via the [GitHub issue tracker](https://github.com/syahrulhidayat/mshqc/issues). Please include your OS, compiler version, and `conda list` output when reporting build or runtime issues.

---

## Citing mshqc

If you use `mshqc` in published work, please cite:

```bibtex
@software{mshqc2026,
  author  = {Hidayat, Muhamad Syahrul},
  title   = {mshqc: A C++20 Quantum Chemistry Engine for Orbital-Optimized M{\o}ller--Plesset Perturbation Theory},
  year    = {2026},
  url     = {https://github.com/syahrulhidayat/mshqc},
  version = {1.0.3}
}
```

---

## References

[1] Helgaker, T., Jørgensen, P., & Olsen, J. (2000). *Molecular Electronic-Structure Theory*. Wiley. https://doi.org/10.1002/9781119019572  
[2] Dunning, T. H., Jr. (1989). Gaussian basis sets for use in correlated molecular calculations. I. *J. Chem. Phys.*, 90(2), 1007–1023. https://doi.org/10.1063/1.456153  
[3] Atkins, P. W., Child, M. S., & Phillips, C. S. G. (1970). *Tables for Group Theory*. Oxford University Press.  
[4] Sun, Q. (2015). Libcint: An efficient general integral library for Gaussian basis functions. *J. Comput. Chem.*, 36(22), 1664–1671. https://doi.org/10.1002/jcc.23981  
[5] Beebe, N. H. F., & Linderberg, J. (1977). Simplifications in the generation and transformation of two-electron integrals in molecular calculations. *Int. J. Quantum Chem.*, 12(4), 683–705. https://doi.org/10.1002/qua.560120408  
[6] Koch, H., Sánchez de Merás, A., & Pedersen, T. B. (2003). Reduced scaling in electronic structure calculations using Cholesky decompositions. *J. Chem. Phys.*, 118(21), 9481–9484. https://doi.org/10.1063/1.1578621  
[7] Weigend, F. (2002). A fully direct RI-HF algorithm: Implementation, optimised auxiliary basis sets, demonstration of accuracy and efficiency. *Phys. Chem. Chem. Phys.*, 4(18), 4285–4291. https://doi.org/10.1039/B204199P  
[8] Häser, M., & Ahlrichs, R. (1989). Improvements on the direct SCF method. *J. Comput. Chem.*, 10(1), 104–111. https://doi.org/10.1002/jcc.540100111  
[9] Roothaan, C. C. J. (1951). New developments in molecular orbital theory. *Rev. Mod. Phys.*, 23(2), 69–89. https://doi.org/10.1103/RevModPhys.23.69  
[10] Pople, J. A., & Nesbet, R. K. (1954). Self-consistent orbitals for radicals. *J. Chem. Phys.*, 22(3), 571–572. https://doi.org/10.1063/1.1740120  
[11] Roothaan, C. C. J. (1960). Self-Consistent Field Theory for Open Shells of Electronic Systems. *Rev. Mod. Phys.*, 32(2), 179–185. https://doi.org/10.1103/RevModPhys.32.179  
[12] Pulay, P. (1980). Convergence acceleration of iterative sequences. The case of SCF iteration. *Chem. Phys. Lett.*, 73(2), 393–398. https://doi.org/10.1016/0009-2614(80)80396-4  
[13] Pulay, P. (1982). Improved SCF convergence acceleration. *J. Comput. Chem.*, 3(4), 556–560. https://doi.org/10.1002/jcc.540030413  
[14] Møller, C., & Plesset, M. S. (1934). Note on an Approximation Treatment for Many-Electron Systems. *Phys. Rev.*, 46(7), 618–622. https://doi.org/10.1103/PhysRev.46.618  
[15] Head-Gordon, M., Pople, J. A., & Frisch, M. J. (1988). MP2 energy evaluation by direct methods. *Chem. Phys. Lett.*, 153(6), 503–506. https://doi.org/10.1016/0009-2614(88)85250-3  
[16] Neese, F., Schwabe, T., Kossmann, S., Schirmer, B., & Grimme, S. (2009). Assessment of Orbital-Optimized, Spin-Component Scaled Second-Order Many-Body Perturbation Theory for Thermochemistry and Kinetics. *J. Chem. Theory Comput.*, 5(11), 3060–3073. https://doi.org/10.1021/ct9003299  
[17] Pople, J. A., Binkley, J. S., & Seeger, R. (1976). Theoretical models incorporating electron correlation. *Int. J. Quantum Chem.*, 10(S10), 1–19. https://doi.org/10.1002/qua.560100802  
[18] Bozkaya, U., Turney, J. M., Yamaguchi, Y., Schaefer, H. F., & Sherrill, C. D. (2011). Quadratically convergent algorithm for orbital optimization in the orbital-optimized coupled-cluster doubles method and in orbital-optimized second-order Møller-Plesset perturbation theory. *J. Chem. Phys.*, 135(10), 104103. https://doi.org/10.1063/1.3631129  
[19] Bozkaya, U. (2011). Orbital-optimized third-order Møller–Plesset perturbation theory and its spin-component and spin-opposite scaled variants: Application to symmetry-breaking problems. *J. Chem. Phys.*, 135(22), 224103. https://doi.org/10.1063/1.3665134  
[20] Guennebaud, G., Jacob, B., et al. (2010). *Eigen v3*. https://eigen.tuxfamily.org  
[21] The HDF Group. (1997–2026). *Hierarchical Data Format, version 5*. https://www.hdfgroup.org/HDF5/  
[22] Matthews, D. A. (2017). High-Performance Tensor Contraction without Transposition. *SIAM J. Sci. Comput.*, 39(6), C476–C509. https://doi.org/10.1137/16M108968X  
[23] Dagum, L., & Menon, R. (1998). OpenMP: an industry standard API for shared-memory programming. *IEEE Comput. Sci. Eng.*, 5(1), 46–55. https://doi.org/10.1109/99.660313  
[24] Jakob, W. (2022). *nanobind: Tiny and efficient C++/Python bindings*. https://github.com/wjakob/nanobind  
[25] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2  

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.
