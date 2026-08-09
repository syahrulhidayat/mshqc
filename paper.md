---
title: 'mshqc: An Embedded MLIR Compiler Infrastructure for High-Performance Quantum Chemistry'
tags:
  - C++20
  - MLIR
  - LLVM
  - quantum chemistry
  - electronic structure
  - HPC
  - SCF
  - Møller-Plesset perturbation theory
  - loop fusion
  - JIT compilation
authors:
  - name: Muhamad Syahrul Hidayat
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher, Indonesia
    index: 1
date: 10 August 2026
bibliography: paper.bib
license: Apache-2.0
---

# Summary

`mshqc` is an ultra-high-performance quantum chemistry library that implements modern electronic
structure methods through a compiler-driven execution model. While traditional quantum chemistry
packages—such as PySCF and Psi4—execute mathematically dense operations via isolated,
pre-compiled C/C++ or Fortran kernels, `mshqc` introduces an embedded C++20 Multi-Level
Intermediate Representation (MLIR) pipeline designated as the `MshqcDialect` [@lattner2021mlir].

The library constructs a high-level computational graph directly in C++ and applies
domain-specific loop fusion and tiling transformations [@bondhugula2008practical] prior to
Just-In-Time (JIT) code generation via the LLVM ORCJIT infrastructure [@lattner2004llvm].
This approach systematically eliminates intermediate tensor DRAM allocation by maintaining
execution strictly within L1/L2 CPU cache boundaries, directly addressing the "memory wall"
bottleneck that plagues bandwidth-bound post-Hartree-Fock computations [@helgaker2000molecular].

The software provides a zero-overhead Python interface via `nanobind` [@jakob2022nanobind],
allowing researchers to execute rigorous quantum chemistry workflows—ranging from Self-Consistent
Field (SCF) methods through orbital-optimized Møller-Plesset Perturbation Theory (OMP2,
OMP3)—while circumventing the hardware bottlenecks inherent to legacy eager-execution
architectures.

# Statement of Need

The progression of high-accuracy quantum chemistry is fundamentally constrained by the memory
wall of modern CPU and GPU architectures. Post-Hartree-Fock methods require complex tensor
contractions scaling from $O(N^4)$ (MP2) to $O(N^6)$ (MP3, MCSCF), where $N$ is the number
of basis functions [@helgaker2000molecular; @moller1934note; @pople1976theoretical].

State-of-the-art frameworks (e.g., PySCF, Psi4) address these operations predominantly via
eager execution, relying on highly optimized Level 3 BLAS (DGEMM via Intel MKL) and isolated
tensor libraries such as TBLIS [@matthews2017high]. While microkernel optimizations are highly
efficient for isolated operations, the eager-execution paradigm forces the materialization of
massive intermediate tensors. These intermediate arrays are continuously written to and read
from main memory (DRAM). Given that modern DRAM bandwidth (~100–300 GB/s on standard hardware)
severely lags behind processor FLOP capabilities, the execution becomes bandwidth-bound, forcing
CPU pipeline stalls during high-latency memory fetches.

Furthermore, frameworks that attempt to resolve this via Python-based JIT tracing introduce
tracing overhead, garbage collection (GC) pauses, and lack domain-specific awareness of quantum
chemistry tensor symmetries during fusion.

`mshqc` fulfills the need for a native, compiler-embedded quantum chemistry engine that resolves
the memory-wall bottleneck deterministically at the architectural level, with a hermetic,
reproducible build environment enforced via `conda-forge`.

# Architectural Design & Compiler Novelty

The primary novelty of `mshqc` lies in its hybrid dual-engine architecture, which separates
programmatic logic from computationally dense tensor graphs into two distinct execution paths.

**Domain-Specific MLIR Dialect (`MshqcDialect`).** Instead of lowering logic prematurely into
generic loops or LLVM IR, `mshqc` abstracts quantum chemical operations—including electron
repulsion integral (ERI) generation (via `libcint` [@sun2015libcint]), Cholesky decomposition
of ERIs [@beebe1977simplifications; @koch2003reduced], Schwarz-based integral screening
[@haser1989improvements], and Fock matrix construction—into a custom MLIR dialect
[@lattner2021mlir]. This preserves high-level semantic information (e.g., permutation symmetries
$(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda)$) that would
otherwise be lost upon premature lowering.

**Cross-Kernel Loop Fusion.** A dedicated transformation pass (`FusionPass.cc`) executes
cross-kernel fusion at the high-level IR stage. By fusing integral transformations and density
contractions into enclosed, tiled loops [@bondhugula2008practical], intermediate tensors are
preserved within L1/L2 CPU cache boundaries. This empirically reduces `L1-dcache-load-misses`
and overall peak Resident Set Size (RSS) for representative molecular systems.

**Deterministic Memory and JIT Scoping.** Unlike pure Julia or Python JIT runtimes, `mshqc`
employs explicit C++20 Resource Acquisition Is Initialization (RAII) memory management and
scopes LLVM ORCJIT compilation [@lattner2004llvm] strictly to tensor contraction graphs. Eager
logic (geometry parsing, DIIS [@pulay1980convergence; @pulay1982improved] convergence control)
relies on Ahead-Of-Time (AOT) C++ execution, eliminating latency overhead for rapid iterative
SCF loops.

**OpenMP Parallelism.** Tensor loops exposed to the JIT backend are annotated for shared-memory
parallelism via `llvm-openmp` [@dagum1998openmp], enabling thread-level scaling across the
available CPU cores with no additional user configuration.

# Implemented Methods

`mshqc` exposes a hermetic Python API covering the following layers of electronic structure theory,
each grounded in the literature and validated against established reference values.

**Molecular Infrastructure.** Geometry and basis set management (`Molecule`, `BasisSet`) follows
the conventions of Helgaker et al. [@helgaker2000molecular], with basis function contractions
sourced from the Dunning cc-pVDZ family [@dunning1989gaussian]. Point group detection and the
Petite List construction for symmetry-adapted integral handling (`PointGroup`, `PetiteList`) are
implemented following standard group-theoretic references [@atkins1970tables].

**Integral Engines.** Analytical evaluation of one- and two-electron integrals is delegated to
`libcint` [@sun2015libcint]. Cholesky decomposition of the four-center ERI tensor
[@beebe1977simplifications; @koch2003reduced] is provided via `CholeskyERI`, enabling
$O(N^3 M)$-scaling storage reduction where $M \ll N^2$. Schwarz-bound prescreening
[@haser1989improvements] is enforced through the `Screening` class.

**Self-Consistent Field (SCF).** The library implements Restricted (`RHF`) [@roothaan1951new],
Unrestricted (`UHF`) [@pople1954self], and Restricted Open-shell (`ROHF`)
[@roothaan1960self] Hartree-Fock methods, all accelerated by the DIIS extrapolation scheme
[@pulay1980convergence; @pulay1982improved].

**Møller-Plesset Perturbation Theory.** Second-order perturbation theory is provided in the
`RMP2` and `UMP2` variants [@moller1934note; @head1988mp2] and the orbital-optimized `OMP2`
[@neese2009assessment]. Third-order corrections are available via `RMP3` and `UMP3`
[@pople1976theoretical] and the orbital-optimized `OMP3` [@bozkaya2011quadratically].

**Analytical Gradients & Geometry Optimization.** First-order energy derivatives
(`GradientResult`) are evaluated analytically following the Pulay force formalism
[@pulay1969abinitio] and the coupled-perturbed framework of Pople et al.
[@pople1979derivative]. Geometry optimization (`OptResult`) is driven by these gradient
evaluations.

# Dependency Stack

The build system requires CMake >= 3.18 and a C++20-compliant compiler. The full dependency
graph is managed hermetially via `conda-forge`:

- **Compiler backend:** `llvmdev`, `mlir` [@lattner2021mlir; @lattner2004llvm]
- **Linear algebra & tensors:** `eigen` [@guennebaud2010eigen], `tblis` [@matthews2017high],
  `mkl`, `mkl-include`
- **Integrals & I/O:** `libcint` [@sun2015libcint], `hdf5` [@hdfgroup1997hdf5]
- **Memory allocator:** `jemalloc` (arena-based allocation to reduce fragmentation overhead)
- **Parallelism:** `llvm-openmp` [@dagum1998openmp]
- **Python interface:** `nanobind` [@jakob2022nanobind], `numpy` [@harris2020array], `scipy`,
  `pytest`

The CMake configuration applies link-time optimization (LTO/IPO) and hardware-specific
instruction tuning (`-march=native`) automatically during the `pip install -v -e .` invocation.

# Acknowledgements

The author acknowledges the foundational ecosystem provided by the LLVM and MLIR compiler
infrastructure projects. Integration with `libcint` [@sun2015libcint] for analytical integral
evaluation and `TBLIS` [@matthews2017high] for fallback tensor contraction has been instrumental
in validating the numerical precision of the generated JIT compiler passes against reference
values from PySCF and Psi4.