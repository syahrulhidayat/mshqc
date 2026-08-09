# mshqc: Embedded MLIR Compiler Infrastructure for High-Performance Quantum Chemistry

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Statement of Need & Architectural Novelty

`mshqc` is a high-performance quantum chemistry library implementing modern electronic structure
methods. Unlike traditional packages that rely on the eager execution of monolithic tensor kernels—
often incurring severe memory-bandwidth bottlenecks due to intermediate read/write operations—`mshqc`
introduces an embedded C++20 compiler pipeline powered by MLIR (`MshqcDialect`).

This architecture performs domain-specific **cross-kernel loop fusion** and **cache-aware tiling**
on a high-level Intermediate Representation (IR) prior to Just-In-Time (JIT) code generation via the
LLVM ORCJIT infrastructure. This approach systematically minimizes intermediate DRAM traffic
by enforcing temporal and spatial data locality within L1/L2 cache boundaries during highly-contracted
operations. Interoperability with Python is enforced via zero-overhead `nanobind` interfaces.


---

## 🌟 Capabilities & Exposed Features

The current implementation provides a hermetic Python API exposing the following
hardware-optimized routines:

### Core Infrastructure
- Molecular geometry initialization and basis set parsing (`Molecule`, `BasisSet`) [5, 6].
- Automated point group detection and symmetry operations (`PointGroup`, `PetiteList`) [7].

### Integral Engines
- Analytical integral evaluation (`IntegralEngine`) [8].
- Cholesky decomposition for Electron Repulsion Integrals (`CholeskyERI`) [9, 10].
- Integral screening (`Screening`) [11].

### Self-Consistent Field (SCF)
- Restricted Hartree-Fock (`RHF`) [12].
- Unrestricted Hartree-Fock (`UHF`) [13].
- Restricted Open-shell Hartree-Fock (`ROHF`) [14].
- Direct Inversion in the Iterative Subspace convergence acceleration (`DIIS`) [15, 16].

### Møller-Plesset Perturbation Theory
- Second-Order: `RMP2`, `UMP2` [17, 18], and Orbital-Optimized `OMP2` [19].
- Third-Order: `RMP3`, `UMP3` [20], and Orbital-Optimized `OMP3` [21].

### Analytical Gradients & Geometry Optimization
- Energy gradient evaluation and geometry optimization logic (`GradientResult`, `OptResult`) [22, 23].

---

## 📋 Prerequisites & Dependency Graph

The build system relies on **Target-Based CMake** (minimum version 3.18) and enforces a strict
C++20 standard.

### Essential HPC Libraries

| Category | Packages | Reference |
|---|---|---|
| Compiler Backend | `llvmdev`, `mlir` | [1, 2] |
| Linear Algebra & Tensors | `eigen`, `tblis`, `mkl`, `mkl-include` | [24, 25] |
| Integrals & Memory | `libcint`, `jemalloc`, `hdf5` | [8, 26] |
| Parallelism | `llvm-openmp` | [27] |
| Python Bindings | `nanobind`, `numpy`, `scipy`, `pytest` | [4, 28] |

---

## 🛠️ Reproducible Installation (Conda/CMake)

To ensure a hermetic build environment and bypass system-level ABI conflicts, the use of
`conda-forge` is mandatory.

**1. Define the Environment Configuration**

Create an `environment.yml` file in the root directory:

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
  - llvmdev
  - mlir
  - jemalloc
  - nanobind
  - numpy>=1.20
  - scipy
  - pytest
```

**2. Provision the HPC Environment**

```bash
conda env create -f environment.yml
conda activate mshqc
```

**3. Compile and Install**

The CMake configuration automatically standardizes symbol visibility for LTO/IPO and applies
hardware-specific native optimizations (`-march=native`).

```bash
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc
pip install -v -e .
```

---

## 🚀 Quickstart

Initialize a molecule and run an SCF-to-MP2 calculation:

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

## 📚 References

> Full bibliography with DOI links. Paste this block verbatim — every `[N]` marker in the
> text above maps to the corresponding entry here.

**Compiler Infrastructure & JIT**

[1] Lattner, C., Amini, M., Bondhugula, U., et al. (2021). MLIR: Scaling Compiler Infrastructure
for Domain Specific Computation. *2021 IEEE/ACM International Symposium on Code Generation and
Optimization (CGO)*, 2–14. https://doi.org/10.1109/CGO51591.2021.9370308

[2] Lattner, C., & Adve, V. (2004). LLVM: A Compilation Framework for Lifelong Program Analysis
& Transformation. *Proceedings of the International Symposium on Code Generation and
Optimization (CGO)*, 75–86. https://doi.org/10.1109/CGO.2004.1281665

**Cache-Blocking & Loop Tiling**

[3] Bondhugula, U., Hartono, A., Ramanujam, J., & Sadayappan, P. (2008). A Practical Automatic
Polyhedral Parallelizer and Locality Optimizer. *Proceedings of the 29th ACM SIGPLAN Conference
on Programming Language Design and Implementation (PLDI)*, 101–113.
https://doi.org/10.1145/1375581.1375595

**Python Bindings**

[4] Jakob, W. (2022). *nanobind: Tiny and efficient C++/Python bindings*.
https://github.com/wjakob/nanobind

**Molecular Structure & Basis Sets**

[5] Helgaker, T., Jørgensen, P., & Olsen, J. (2000). *Molecular Electronic-Structure Theory*.
Wiley. https://doi.org/10.1002/9781119019572

[6] Dunning, T. H., Jr. (1989). Gaussian basis sets for use in correlated molecular calculations.
I. The atoms boron through neon and hydrogen. *The Journal of Chemical Physics*, 90(2), 1007–1023.
https://doi.org/10.1063/1.456153

**Point Group Symmetry**

[7] Atkins, P. W., Child, M. S., & Phillips, C. S. G. (1970). *Tables for Group Theory*. Oxford
University Press.

**Integral Evaluation — libcint**

[8] Sun, Q. (2015). Libcint: An efficient general integral library for Gaussian basis functions.
*Journal of Computational Chemistry*, 36(22), 1664–1671. https://doi.org/10.1002/jcc.23981

**Cholesky Decomposition of ERIs**

[9] Beebe, N. H. F., & Linderberg, J. (1977). Simplifications in the generation and transformation
of two-electron integrals in molecular calculations. *International Journal of Quantum Chemistry*,
12(4), 683–705. https://doi.org/10.1002/qua.560120408

[10] Koch, H., Sánchez de Merás, A., & Pedersen, T. B. (2003). Reduced scaling in electronic
structure calculations using Cholesky decompositions. *The Journal of Chemical Physics*, 118(21),
9481–9484. https://doi.org/10.1063/1.1578621

**Integral Screening (Schwarz)**

[11] Häser, M., & Ahlrichs, R. (1989). Improvements on the direct SCF method. *Journal of
Computational Chemistry*, 10(1), 104–111. https://doi.org/10.1002/jcc.540100111

**Restricted Hartree-Fock**

[12] Roothaan, C. C. J. (1951). New developments in molecular orbital theory. *Reviews of Modern
Physics*, 23(2), 69–89. https://doi.org/10.1103/RevModPhys.23.69

**Unrestricted Hartree-Fock**

[13] Pople, J. A., & Nesbet, R. K. (1954). Self-consistent orbitals for radicals. *The Journal of
Chemical Physics*, 22(3), 571–572. https://doi.org/10.1063/1.1740120

**Restricted Open-shell Hartree-Fock**

[14] Roothaan, C. C. J. (1960). Self-Consistent Field Theory for Open Shells of Electronic
Systems. *Reviews of Modern Physics*, 32(2), 179–185.
https://doi.org/10.1103/RevModPhys.32.179

**DIIS Convergence**

[15] Pulay, P. (1980). Convergence acceleration of iterative sequences. The case of SCF
iteration. *Chemical Physics Letters*, 73(2), 393–398.
https://doi.org/10.1016/0009-2614(80)80396-4

[16] Pulay, P. (1982). Improved SCF convergence acceleration. *Journal of Computational
Chemistry*, 3(4), 556–560. https://doi.org/10.1002/jcc.540030413

**MP2 — Restricted & Unrestricted**

[17] Møller, C., & Plesset, M. S. (1934). Note on an Approximation Treatment for Many-Electron
Systems. *Physical Review*, 46(7), 618–622. https://doi.org/10.1103/PhysRev.46.618

[18] Head-Gordon, M., Pople, J. A., & Frisch, M. J. (1988). MP2 energy evaluation by direct
methods. *Chemical Physics Letters*, 153(6), 503–506.
https://doi.org/10.1016/0009-2614(88)85250-3

**Orbital-Optimized MP2**

[19] Neese, F., Schwabe, T., Kossmann, S., Schirmer, B., & Grimme, S. (2009). Assessment of
Orbital-Optimized, Spin-Component Scaled Second-Order Many-Body Perturbation Theory for
Thermochemistry and Kinetics. *Journal of Chemical Theory and Computation*, 5(11), 3060–3073.
https://doi.org/10.1021/ct9003299

**MP3**

[20] Pople, J. A., Binkley, J. S., & Seeger, R. (1976). Theoretical models incorporating
electron correlation. *International Journal of Quantum Chemistry*, 10(S10), 1–19.
https://doi.org/10.1002/qua.560100802

**Orbital-Optimized MP3**

[21] Bozkaya, U., Turney, J. M., Yamaguchi, Y., Schaefer, H. F., & Sherrill, C. D. (2011).
Quadratically convergent algorithm for orbital optimization in the orbital-optimized
coupled-cluster doubles method and in orbital-optimized second-order Møller-Plesset perturbation
theory. *The Journal of Chemical Physics*, 135(10), 104103.
https://doi.org/10.1063/1.3631129

**Analytical Gradients**

[22] Pulay, P. (1969). Ab initio calculation of force constants and equilibrium geometries in
polyatomic molecules. *Molecular Physics*, 17(2), 197–204.
https://doi.org/10.1080/00268976900100941

[23] Pople, J. A., Krishnan, R., Schlegel, H. B., & Binkley, J. S. (1979). Derivative studies in
Hartree-Fock and Møller-Plesset theories. *International Journal of Quantum Chemistry*, 16(S13),
225–241. https://doi.org/10.1002/qua.560160825

**Linear Algebra — Eigen & TBLIS**

[24] Guennebaud, G., Jacob, B., et al. (2010). *Eigen v3*. http://eigen.tuxfamily.org

[25] Matthews, D. A. (2017). High-Performance Tensor Contraction without Transposition. *SIAM
Journal on Scientific Computing*, 39(6), C476–C509. https://doi.org/10.1137/16M108968X

**HDF5 for I/O**

[26] The HDF Group. (1997–2024). *Hierarchical Data Format, version 5*.
https://www.hdfgroup.org/HDF5/

**OpenMP Parallelism**

[27] Dagum, L., & Menon, R. (1998). OpenMP: an industry standard API for shared-memory
programming. *IEEE Computational Science and Engineering*, 5(1), 46–55.
https://doi.org/10.1109/99.660313

**NumPy / SciPy**

[28] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with
NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.