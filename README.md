# mshqc

**A lightweight C++20 quantum chemistry engine for orbital-optimized correlation methods**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

`mshqc` is a research-oriented quantum chemistry package that implements a focused set of
electronic structure methods — Hartree-Fock, MP2/MP3, and their orbital-optimized (OMP2/OMP3)
variants — in a compact C++20 core with a `nanobind`-based Python interface. It is built as a
**method-development testbed** rather than a general-purpose replacement for large ecosystems
such as Psi4 or PySCF.

---

## Statement of Need

Mature quantum chemistry packages provide broad method coverage, but their size and accumulated
architectural layers make them a heavy starting point for developers who want to prototype a new
correlation method, a new tensor-contraction strategy, or a new SCF convergence scheme. Iterating
on low-level memory layout or contraction ordering inside those codebases typically means working
through several abstraction layers before reaching the numerical kernel.

`mshqc` targets this gap directly. It exposes a small, greenfield C++20 core built around
[TBLIS](https://github.com/devinamatthews/tblis) for tensor contractions and HDF5-backed
out-of-core density fitting, so a developer working on, say, an orbital-optimization algorithm can
reach the relevant kernel in one or two hops instead of navigating a large legacy codebase. The
current focus of the package is a from-scratch implementation of orbital-optimized MP2/MP3
(OMP2/OMP3), using trust-region SOSCF with an approximate Hessian and a macro/micro-iteration
L-BFGS scheme for orbital rotation, alongside standard RHF/UHF/ROHF references and DIIS
acceleration.

---

## Comparison with Existing Software

| | mshqc | Psi4 | PySCF |
|---|---|---|---|
| Primary audience | Method prototyping | General production use | General production use |
| Core language | C++20 | C++ | Python + C |
| Method coverage | HF, MP2/MP3, OMP2/OMP3 | Very broad | Very broad |
| Codebase size | Small, single-purpose | Large | Large |
| Tensor backend | TBLIS | Internal | NumPy / internal |

This table is meant to orient a prospective user, not to claim superiority — `mshqc` deliberately
trades method breadth for a smaller surface area that's easier to modify.

---

## Features

### Core Infrastructure
- Molecular geometry initialization and basis set parsing — `Molecule`, `BasisSet` [1, 2]
- Automated point group detection and symmetry adaptation — `PointGroup`, `PetiteList` [3]

### Integrals & Memory Management
- Analytical integral evaluation via `libcint` — `IntegralEngine` [4]
- Out-of-core resolution-of-identity (density fitting) backed by HDF5
- Cholesky decomposition of electron repulsion integrals [5, 6]
- Schwarz integral screening [7]

### Self-Consistent Field
- Restricted, Unrestricted, and Restricted Open-shell Hartree-Fock — `RHF`, `UHF`, `ROHF` [8, 9, 10]
- DIIS convergence acceleration with SAD initial guess [11, 12]

### Møller–Plesset Perturbation Theory
- Second order: `RMP2`, `UMP2` [13, 14], and orbital-optimized `OMP2` [15]
- Third order: `RMP3`, `UMP3` [16], and orbital-optimized `OMP3` [17]

### Gradients & Geometry Optimization
- Analytical energy gradients and geometry optimization — `GradientResult`, `OptResult` [18, 19]

---

## Prerequisites & Dependencies

Build system: CMake ≥ 3.18 (target-based), C++20 standard, LTO/IPO enabled by default. The build
applies `-march=native` unless overridden — see [Build Options](#build-options) below.

| Category | Packages | Scope | Reference |
|---|---|---|---|
| Linear algebra & tensors | `eigen`, `tblis`, `mkl`, `mkl-include` | Build + runtime | [20, 21] |
| Integrals & memory | `libcint`, `jemalloc`, `hdf5` | Build + runtime | [4, 22] |
| Parallelism | `llvm-openmp` | Build + runtime | [23] |
| Python bindings | `nanobind` | Build + runtime | [24] |
| Python runtime | `numpy`, `scipy` | Runtime | [25] |
| Testing | `pytest` | Dev only | — |

---

## Installation

A `conda-forge` environment is the supported route, since it pins compatible ABI-matched builds
of MKL, HDF5, and OpenMP.

**1. Create `environment.yml`**

```yaml
name: mshqc
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.8
  - pip
  - cmake>=3.18
  - ninja
  - pkg-config
  - c-compiler
  - cxx-compiler
  - llvm-openmp
  - mkl
  - mkl-include
  - eigen
  - libcint
  - tblis
  - hdf5
  - jemalloc
  - nanobind
  - numpy>=1.20
  - scipy
  - pytest
```

**2. Provision the environment**

```bash
conda env create -f environment.yml
conda activate mshqc
```

**3. Build and install**

```bash
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install -v -e .
```

### Build Options

To disable `-march=native` (useful for building portable binaries or on heterogeneous clusters):

```bash
CMAKE_ARGS="-DMSHQC_NATIVE_ARCH=OFF" pip install -v -e .
```

---

## Quickstart

```python
import mshqc
import numpy as np

# 1. Initialize molecular geometry (Angstrom)
mol = mshqc.Molecule()
mol.add_atom(8,  0.0000000000,  0.0000000000,  0.1174000000)   # Oxygen
mol.add_atom(1,  0.0000000000,  0.7570000000, -0.4696000000)   # Hydrogen
mol.add_atom(1,  0.0000000000, -0.7570000000, -0.4696000000)   # Hydrogen

# 2. Assign basis set and instantiate integral engine
basis     = mshqc.BasisSet("cc-pVDZ", mol)
integrals = mshqc.IntegralEngine(mol, basis)

# 3. Configure and execute Restricted Hartree-Fock (RHF)
scf_config = mshqc.SCFConfig()
scf_config.energy_threshold = 1e-8
rhf = mshqc.RHF(mol, basis, integrals, scf_config)
rhf.compute()
print(f"RHF Converged Energy: {rhf.energy():.10f} Eh")

# 4. Execute Second-Order Møller-Plesset Perturbation (RMP2)
mp2_config = mshqc.MP2Config()
rmp2 = mshqc.RMP2(mol, basis, integrals, rhf, mp2_config)
rmp2.compute()
print(f"RMP2 Correlation Energy: {rmp2.correlation_energy():.10f} Eh")
print(f"RMP2 Total Energy:       {rmp2.total_energy():.10f} Eh")
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Contributing

Bug reports and pull requests are welcome via the
[issue tracker](https://github.com/syahrulhidayat/mshqc/issues). Please include your OS,
compiler version, and `environment.yml` output (`conda list`) when reporting build or runtime
issues — this project's small dependency surface makes ABI mismatches the most common source of
problems.

---

## Citing mshqc

If you use `mshqc` in published work, please cite:

```bibtex
@software{mshqc,
  author  = {Hidayat, Syahrul},
  title   = {mshqc: A C++20 Quantum Chemistry Engine for Orbital-Optimized Correlation Methods},
  year    = {2026},
  url     = {https://github.com/syahrulhidayat/mshqc},
  version = {<fill in release version>}
}
```

*(Update the entry above once a JOSS paper or Zenodo DOI is issued for the release.)*

---

## References

**Molecular Structure & Basis Sets**
[1] Helgaker, T., Jørgensen, P., & Olsen, J. (2000). *Molecular Electronic-Structure Theory*. Wiley. https://doi.org/10.1002/9781119019572
[2] Dunning, T. H., Jr. (1989). Gaussian basis sets for use in correlated molecular calculations. I. *The Journal of Chemical Physics*, 90(2), 1007–1023. https://doi.org/10.1063/1.456153

**Point Group Symmetry**
[3] Atkins, P. W., Child, M. S., & Phillips, C. S. G. (1970). *Tables for Group Theory*. Oxford University Press.

**Integral Evaluation**
[4] Sun, Q. (2015). Libcint: An efficient general integral library for Gaussian basis functions. *Journal of Computational Chemistry*, 36(22), 1664–1671. https://doi.org/10.1002/jcc.23981

**Cholesky Decomposition of ERIs**
[5] Beebe, N. H. F., & Linderberg, J. (1977). Simplifications in the generation and transformation of two-electron integrals in molecular calculations. *International Journal of Quantum Chemistry*, 12(4), 683–705. https://doi.org/10.1002/qua.560120408
[6] Koch, H., Sánchez de Merás, A., & Pedersen, T. B. (2003). Reduced scaling in electronic structure calculations using Cholesky decompositions. *The Journal of Chemical Physics*, 118(21), 9481–9484. https://doi.org/10.1063/1.1578621

**Integral Screening (Schwarz)**
[7] Häser, M., & Ahlrichs, R. (1989). Improvements on the direct SCF method. *Journal of Computational Chemistry*, 10(1), 104–111. https://doi.org/10.1002/jcc.540100111

**Hartree-Fock Theory**
[8] Roothaan, C. C. J. (1951). New developments in molecular orbital theory. *Reviews of Modern Physics*, 23(2), 69–89. https://doi.org/10.1103/RevModPhys.23.69
[9] Pople, J. A., & Nesbet, R. K. (1954). Self-consistent orbitals for radicals. *The Journal of Chemical Physics*, 22(3), 571–572. https://doi.org/10.1063/1.1740120
[10] Roothaan, C. C. J. (1960). Self-Consistent Field Theory for Open Shells of Electronic Systems. *Reviews of Modern Physics*, 32(2), 179–185. https://doi.org/10.1103/RevModPhys.32.179

**DIIS Convergence**
[11] Pulay, P. (1980). Convergence acceleration of iterative sequences. The case of SCF iteration. *Chemical Physics Letters*, 73(2), 393–398. https://doi.org/10.1016/0009-2614(80)80396-4
[12] Pulay, P. (1982). Improved SCF convergence acceleration. *Journal of Computational Chemistry*, 3(4), 556–560. https://doi.org/10.1002/jcc.540030413

**MP2 & Orbital-Optimized MP2**
[13] Møller, C., & Plesset, M. S. (1934). Note on an Approximation Treatment for Many-Electron Systems. *Physical Review*, 46(7), 618–622. https://doi.org/10.1103/PhysRev.46.618
[14] Head-Gordon, M., Pople, J. A., & Frisch, M. J. (1988). MP2 energy evaluation by direct methods. *Chemical Physics Letters*, 153(6), 503–506. https://doi.org/10.1016/0009-2614(88)85250-3
[15] Neese, F., Schwabe, T., Kossmann, S., Schirmer, B., & Grimme, S. (2009). Assessment of Orbital-Optimized, Spin-Component Scaled Second-Order Many-Body Perturbation Theory for Thermochemistry and Kinetics. *Journal of Chemical Theory and Computation*, 5(11), 3060–3073. https://doi.org/10.1021/ct9003299

**MP3 & Orbital-Optimized MP3**
[16] Pople, J. A., Binkley, J. S., & Seeger, R. (1976). Theoretical models incorporating electron correlation. *International Journal of Quantum Chemistry*, 10(S10), 1–19. https://doi.org/10.1002/qua.560100802
[17] Bozkaya, U., Turney, J. M., Yamaguchi, Y., Schaefer, H. F., & Sherrill, C. D. (2011). Quadratically convergent algorithm for orbital optimization in the orbital-optimized coupled-cluster doubles method and in orbital-optimized second-order Møller-Plesset perturbation theory. *The Journal of Chemical Physics*, 135(10), 104103. https://doi.org/10.1063/1.3631129

**Analytical Gradients**
[18] Pulay, P. (1969). Ab initio calculation of force constants and equilibrium geometries in polyatomic molecules. *Molecular Physics*, 17(2), 197–204. https://doi.org/10.1080/00268976900100941
[19] Pople, J. A., Krishnan, R., Schlegel, H. B., & Binkley, J. S. (1979). Derivative studies in Hartree-Fock and Møller-Plesset theories. *International Journal of Quantum Chemistry*, 16(S13), 225–241. https://doi.org/10.1002/qua.560160825

**Linear Algebra, Python Bindings, and I/O**
[20] Guennebaud, G., Jacob, B., et al. (2010). *Eigen v3*. http://eigen.tuxfamily.org
[21] Matthews, D. A. (2017). High-Performance Tensor Contraction without Transposition. *SIAM Journal on Scientific Computing*, 39(6), C476–C509. https://doi.org/10.1137/16M108968X
[22] The HDF Group. (1997–2024). *Hierarchical Data Format, version 5*. https://www.hdfgroup.org/HDF5/
[23] Dagum, L., & Menon, R. (1998). OpenMP: an industry standard API for shared-memory programming. *IEEE Computational Science and Engineering*, 5(1), 46–55. https://doi.org/10.1109/99.660313
[24] Jakob, W. (2022). *nanobind: Tiny and efficient C++/Python bindings*. https://github.com/wjakob/nanobind
[25] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.