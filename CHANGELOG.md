# Changelog

All notable changes to `mshqc` are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-25

### Breaking Changes
- **Pure Thin-Core C++20 Architecture (Zero-HDF5):** Refactored the C++20 core down to 9 irreducible kernel files (`src/core/`, `src/ints/`, `src/integrals/`, `src/kernels/`) and 2 `nanobind` binding files (`bindings.cc` = 161 LOC, 44 kernel symbols fully stubbed in `_mshqc.pyi`), removing HDF5 build/runtime dependencies (`MSHQC_NO_HDF5=1`).
- **Legacy v1.0 Archive:** The original monolithic C++20 engine (`BaseSCF`, C++ `PetiteList`, HDF5 out-of-core streaming) is permanently preserved under tag `v1.0.5-thesis-archive` (and branch `test`) for full reproducibility of thesis benchmarks.
- **Modular Python Orchestration API:** High-level workflows now route through `mshqc.run`, `MSHQCCalculator`, `MSHQCSession`, `mshqc.methods`, `mshqc.backends`, and `mshqc.kernels` instead of legacy `SCFConfig`/`MP2Config` wrappers.

### Added
- **Extended Orbital-Optimized Methods (RHF & UHF, All-Electron & Frozen-Core):**
  - `SCS-OMP2`, `SOS-OMP2`, `OMP2.5`, and $\kappa$-regularized **`kappa-OOMP2`** with degeneracy-safe L'Hôpital orbital gradients.
  - $\mathcal{O}(N^4)$ zero-$T_2$ Laplace-Transformed **`LT-SOS-OMP2`** backed by OpenMP C++20 quadrature kernels and adaptive NNLS quadrature caching.
- **Analytical DF Nuclear Gradients, Geometry Optimization & Spectroscopy:**
  - Analytical DF gradients (all-electron & frozen-core) for `DF-RHF`, `DF-UHF`, `DF-ROHF`, `DF-OMP2`, `DF-SCS-OMP2`, `DF-SOS-OMP2`, and `DF-kappa-OOMP2` (verified against central finite differences to $\le 6.5 \times 10^{-8}\text{ Ha/Bohr}$).
  - Matrix-free Trust-Region SOSCF geometry optimizer in Eckart internal vibrational coordinates, harmonic vibrational frequencies + RRHO thermochemistry, and 1D PES scans.
  - Excited-state `DF-CIS` and `DF-TDHF` prototypes.
- **Dynamic Compute Backends (`mshqc.backends`):** Transparent context-managed switching (`use_backend`) between C++20/TBLIS, Pure-NumPy (parity $|\Delta E| = 2.84 \times 10^{-14}\text{ Ha}$), and optional PyTorch CUDA/CPU FP64 (`enable_torch_backend`) with automatic C++ fallback.
- **Pure-Python Abelian Symmetry, Properties & Interoperability:**
  - Principal-axis alignment, Abelian point-group detection ($C_{2v}$, $D_{2h}$, etc.), AO symmetrization, and MO irrep labeling.
  - Natural Orbitals / NOONs ($N_U$), Mulliken & Löwdin population analysis, `.molden` export, `.npz` orbital checkpoints, built-in molecule/atom presets (`mshqc.molecule`, `mshqc.atom`), and stateless PySCF interoperability (`run_from_pyscf`).
- **Automated 3-OS Release & Zenodo Archiving:** Cross-platform `cibuildwheel` workflow for Linux (`x86_64`), macOS (`arm64`), and Windows (`x64`) with automated PyPI upload, GitHub Releases, `SHA256SUMS.txt`, and Zenodo DOI publication.

## [1.0.5] - 2026-11-15
- Final monolithic C++20 release archived as `v1.0.5-thesis-archive` with 507-test Psi4 cross-validation benchmark and precompiled cross-platform PyPI wheels.
