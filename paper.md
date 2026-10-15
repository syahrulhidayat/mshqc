---
title: 'mshqc: A C++20 Quantum Chemistry Engine for Orbital-Optimized Møller–Plesset Perturbation Theory'
author: 'Muhamad Syahrul Hidayat (Department of Physics, Universitas Gadjah Mada)'
tags:
  - C++20
  - Python
  - quantum chemistry
  - electronic structure
  - Møller-Plesset perturbation theory
  - orbital optimization
  - density fitting
  - Cholesky decomposition
  - tensor contraction
authors:
  - name: Muhamad Syahrul Hidayat
    orcid: 0009-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada, Yogyakarta, Indonesia
    index: 1
date: 15 October 2026
bibliography: paper.bib
license: Apache-2.0
---

# Summary

`mshqc` is a high-performance quantum chemistry library that implements a focused set of wave-function-based electronic structure methods in a compact C++20 core with a low-overhead Python interface built on `nanobind` [@jakob2022nanobind]. The library provides restricted, unrestricted, and restricted open-shell Hartree–Fock references (`RHF`, `UHF`, `ROHF`) [@roothaan1951new; @pople1954self; @roothaan1960self] accelerated by Pulay's Direct Inversion in the Iterative Subspace (`DIIS`) [@pulay1980convergence; @pulay1982improved] and Superposition of Atomic Densities (`SAD`) initial guesses, alongside second- and third-order Møller–Plesset perturbation theory in both canonical (`RMP2`, `UMP2`, `RMP3`, `UMP3`) [@moller1934note; @pople1976theoretical; @head1988mp2] and orbital-optimized forms (`OMP2`, `OMP3`) [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3].

To eliminate steep memory and computational bottlenecks in four-center electron repulsion integrals (ERIs) evaluated via `libcint` [@sun2015libcint], `mshqc` unifies three integral representations across all implemented SCF and correlated solvers: exact four-center tensors (`exact`) with Schwarz prescreening [@haser1989improvements], pivot-based Cholesky decomposition (`cholesky`) [@beebe1977simplifications; @koch2003reduced], and auxiliary-basis Density Fitting / Resolution of the Identity (`df`) [@feyereisen1993use; @weigend2002fully]. High-dimensional tensor contractions in the correlated kernels are dispatched to `TBLIS` [@matthews2017high], while orbital rotations in `OMP2` and `OMP3` are driven by a matrix-free Steihaug–Toint trust-region Second-Order SCF (`SOSCF`) optimizer.

# Statement of need

Orbital-optimized Møller–Plesset perturbation methods (`OMP2` and `OMP3`) relax the reference orbitals in the presence of dynamical electron correlation, restoring Hellmann–Feynman satisfaction for one-electron properties, eliminating artificial spin contamination in open-shell radicals, and mitigating first-order orbital relaxation errors [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3]. However, because each macro-iteration of an orbital-optimized calculation requires re-transforming two-electron integrals and re-evaluating $\mathcal{O}(N^5)$ (`OMP2`) or $\mathcal{O}(N^6)$ (`OMP3`) amplitude and response contractions, naive implementations quickly become memory- and compute-bound.

Researchers and students who wish to prototype new orbital-optimization algorithms, three-index tensor factorizations, or custom contraction schedules face a steep barrier to entry in large production suites. General-purpose ecosystems are designed for broad method coverage, which necessarily introduces deep abstraction layers between the high-level Python driver and the low-level numerical kernels. `mshqc` addresses this need by providing a compact, inspectable C++20 codebase where the SCF engines, three-index `cholesky`/`df` integral transformers, `TBLIS` tensor contractions, and trust-region `SOSCF` orbital optimizers are unified under a clean, two-tier API. Users can execute complete multi-method pipelines in a single Python call via `mshqc.run()`, or instantiate individual C++ classes directly to inspect and modify intermediate tensors.

# State of the field

Mature open-source quantum chemistry packages such as `Psi4` [@smith2020psi4] and `PySCF` [@sun2020recent] offer comprehensive electronic structure capabilities, including canonical and orbital-optimized perturbation theory. In `Psi4`, orbital-optimized methods are implemented within the compiled `fnocc` and `occ` modules, whereas `PySCF` combines C-level integral evaluation with Python/NumPy tensor operations. While both ecosystems excel as general-purpose production tools, modifying their internal integral transformation pipelines or orbital Hessian preconditioners requires navigating hundreds of thousands of lines of code.

Conversely, educational or lightweight quantum chemistry projects typically implement only canonical `RHF` and `RMP2` in pure Python, lacking open-shell references (`UHF`/`ROHF`), third-order correlation (`MP3`/`OMP3`), point-group symmetry reduction, or rank-reduced three-index integrals (`cholesky` and `df`). `mshqc` bridges the gap between minimalist teaching codes and monolithic production suites: it maintains a small, single-purpose C++20 codebase while delivering execution speeds that match or exceed production packages on `OMP2` and `OMP3` workloads.

# Software design

`mshqc` is organized into modular C++20 layers exposed to Python through `nanobind` [@jakob2022nanobind] with zero-copy `Eigen` [@guennebaud2010eigen] matrix interoperability and automatic Global Interpreter Lock (`GIL`) release during compute-intensive calls:

1. **Molecular Symmetry & Basis Infrastructure (`Molecule`, `BasisSet`, `PointGroup`, `PetiteList`):** Parses Gaussian basis set (`.gbs`) files bundled directly within the Python package—including the Dunning correlation-consistent families (`cc-pVXZ`) [@dunning1989gaussian] and matched `RI`/`JKFIT` auxiliary basis sets [@weigend2002fully]—with automatic contraction normalization [@helgaker2000molecular]. Automated point-group detection aligns molecular coordinates to canonical symmetry axes and constructs the `PetiteList` of symmetry-unique shell pairs [@atkins1970tables].
2. **Unified Integral Engine (`IntegralEngine`, `Screening`, `CholeskyERI`, `DensityFittingERI`):** Evaluates analytical one- and two-electron integrals in spherical harmonic or Cartesian representations via `libcint` [@sun2015libcint]. All solvers accept a unified `eri_method` configuration (`"exact"`, `"cholesky"`, or `"df"`), reducing two-electron RAM storage from $\mathcal{O}(N^4)$ four-center tensors to $\mathcal{O}(N_{\text{aux}} N^2)$ three-center metric-decomposed matrices $B_{pq}^Q$ (with optional `HDF5`-backed out-of-core disk storage [@hdfgroup1997hdf5]).
3. **Self-Consistent Field (`RHF`, `UHF`, `ROHF`):** Shares a common base infrastructure (`BaseSCF`) supporting SAD initial guesses, Schwarz screening, symmetry-reduced Fock construction, and `DIIS` extrapolation. For open-shell systems under Density Fitting, exchange contractions exploit occupied-orbital rank projection to avoid $\mathcal{O}(N_{\text{aux}} N^3)$ bottlenecks.
4. **Correlated & Orbital-Optimized Solvers (`RMP2`, `UMP2`, `OMP2`, `RMP3`, `UMP3`, `OMP3`):** Dispatches four-index tensor contractions to `TBLIS` [@matthews2017high] and three-index `df`/`cholesky` contractions to cache-blocked `OpenMP` + `BLAS` (`DGEMM`/`DSYRK`) kernels [@dagum1998openmp]. In `OMP2` and `OMP3`, non-redundant orbital rotation parameters $\kappa_{ai}$ are optimized using a matrix-free Steihaug–Toint truncated conjugate gradient (`TrustRegionSOSCF`) solver coupled with unrelaxed and Z-vector response densities.

# Validation and performance

All methods in `mshqc` (`RHF`/`UHF`, `RMP2`/`UMP2`, `OMP2`, and `OMP3`) have been benchmarked against `Psi4` [@smith2020psi4] across closed-shell (`Ne`, $\text{H}_2\text{O}$) and open-shell (`Li`, `C`, `O`) systems using the `cc-pVTZ` basis set (`cc-pVTZ-RI` auxiliary basis for `df`, all electrons correlated, decomposition and convergence thresholds set to $10^{-9}\text{ Ha}$). Across all 60 benchmark calculations (`5 systems × 3 ERI modes × 4 methods`), `mshqc` reproduces `Psi4` reference energies to within $|\Delta E| \le 1.27\ \mu\text{Ha}$ (with 54 of 60 runs matching to $0.00\ \mu\text{Ha}$ at $10^{-8}\text{ Ha}$ precision, and all 20 `df` runs matching `Psi4` to $|\Delta E| \le 0.01\ \mu\text{Ha}$).

As shown in Figure 1 and Table 1, `mshqc` outperforms `Psi4` in all 60 test cases, achieving wall-clock speedups of $2.4\times$ to $18.6\times$ on `OMP2` and $2.8\times$ to $11.6\times$ on `OMP3`. On $\text{H}_2\text{O}$ ($N = 58$, $N_{\text{aux}} = 141$), three-center Density Fitting reduces the raw ERI memory footprint from $90.5\text{ MB}$ to $3.8\text{ MB}$ ($23.9\times$ RAM reduction) while reducing the `OMP3` execution time from $0.797\text{ s}$ (`exact`) to $0.358\text{ s}$ (`df`).

![Benchmark evaluation of `mshqc` vs `Psi4` in the `cc-pVTZ` (`cc-pVTZ-RI`) basis: (A) `OMP3` wall-clock speedup ($t_{\text{Psi4}} / t_{\text{mshqc}}$) across `exact`, `cholesky`, and `df` integral modes; (B) Trust-Region SOSCF orbital gradient norm $\|\mathbf{g}\|_2$ convergence per macro-iteration; (C) ERI tensor RAM footprint scaling comparing four-center $\mathcal{O}(N^4)$ storage against three-center Cholesky and Density Fitting representations.](benchmark_figure.pdf){ width=100% }

Table 1: Representative benchmark comparison between `mshqc` and `Psi4` (`cc-pVTZ` / `cc-pVTZ-RI`, threshold $10^{-9}\text{ Ha}$), reporting converged total energy, absolute deviation $|\Delta E| = |E_{\text{mshqc}} - E_{\text{Psi4}}|$ in $\mu\text{Ha}$, wall-clock times in seconds, and speedup.

|System|ERI|Method|Energy (Ha)|$|\Delta E|$|`mshqc` (s)|`Psi4` (s)|Speedup|
|:--|:--|:--|--:|--:|--:|--:|--:|
|Ne ($^1S$)|exact|RHF|-128.53186164|0.00|0.200|0.749|3.75x|
|Ne ($^1S$)|exact|OMP2|-128.81022310|0.00|0.093|0.226|2.43x|
|Ne ($^1S$)|exact|OMP3|-128.81027068|0.00|0.117|0.353|3.02x|
|Ne ($^1S$)|cholesky|OMP3|-128.81027068|0.00|0.125|0.545|4.36x|
|Ne ($^1S$)|df|OMP3|-128.81042564|0.00|0.116|0.326|2.81x|
|H$_2$O ($^1A_1$)|exact|RHF|-76.05712742|0.00|0.062|0.365|5.89x|
|H$_2$O ($^1A_1$)|exact|OMP2|-76.33406054|0.00|0.496|1.569|3.16x|
|H$_2$O ($^1A_1$)|exact|OMP3|-76.33711056|0.00|0.797|4.074|5.11x|
|H$_2$O ($^1A_1$)|cholesky|OMP2|-76.33406044|0.10|0.330|3.146|9.53x|
|H$_2$O ($^1A_1$)|cholesky|OMP3|-76.33711183|1.27|0.705|7.382|10.47x|
|H$_2$O ($^1A_1$)|df|RMP2|-76.33220605|0.00|0.005|0.066|13.20x|
|H$_2$O ($^1A_1$)|df|OMP2|-76.33402331|0.00|0.144|0.698|4.85x|
|H$_2$O ($^1A_1$)|df|OMP3|-76.33725847|0.00|0.358|1.638|4.58x|
|Li ($^2S$, UHF)|exact|OMP2|-7.44391221|0.00|0.033|0.572|17.33x|
|Li ($^2S$, UHF)|cholesky|OMP3|-7.44569499|0.00|0.149|1.410|9.46x|
|Li ($^2S$, UHF)|df|OMP3|-7.44569708|0.00|0.104|1.202|11.56x|
|C ($^3P$, UHF)|exact|OMP3|-37.78269153|0.00|0.299|0.913|3.05x|
|C ($^3P$, UHF)|cholesky|OMP3|-37.78269153|0.00|0.291|1.546|5.31x|
|C ($^3P$, UHF)|df|OMP2|-37.76763043|0.00|0.055|0.471|8.56x|
|C ($^3P$, UHF)|df|OMP3|-37.78281240|0.00|0.192|1.068|5.56x|
|O ($^3P$, UHF)|exact|OMP3|-74.98104071|0.00|0.297|0.972|3.27x|
|O ($^3P$, UHF)|cholesky|OMP2|-74.96657683|0.00|0.092|0.818|8.89x|
|O ($^3P$, UHF)|cholesky|OMP3|-74.98104071|0.00|0.360|1.741|4.84x|
|O ($^3P$, UHF)|df|OMP2|-74.96656224|0.00|0.044|0.533|12.11x|
|O ($^3P$, UHF)|df|OMP3|-74.98111725|0.00|0.312|1.066|3.42x|

# Availability and documentation

`mshqc` is released under the Apache License 2.0 and hosted on GitHub (<https://github.com/syahrulhidayat/mshqc>), with precompiled Linux wheels built via automated GitHub Actions workflows. Documentation, quickstart tutorials, and troubleshooting guides are included in the `docs/` directory of the repository.

# AI usage disclosure

During the preparation of this work, the author used generative AI tools (Gemini) to assist with code refactoring, performance profiling analysis, and manuscript editing. The author reviewed, validated, and tested all code and text against analytical quantum chemistry references and assumes full responsibility for the published work.

# Acknowledgements

The author acknowledges the developers of `libcint` [@sun2015libcint] for analytical Gaussian integral evaluation and `TBLIS` [@matthews2017high] for transpose-free tensor contractions, as well as the `PySCF` [@sun2020recent] and `Psi4` [@smith2020psi4] projects for providing reference values during numerical validation.

# References