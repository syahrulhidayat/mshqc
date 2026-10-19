---
title: 'mshqc: A C++20 Quantum Chemistry Engine for Orbital-Optimized Møller–Plesset Perturbation Theory'
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
date: 19 October 2026
bibliography: paper.bib
---

# Summary

`mshqc` is a high-performance quantum chemistry library implementing wave-function-based electronic structure methods in a compact C++20 core with a Python interface built on `nanobind` [@jakob2022nanobind]. It provides restricted, unrestricted, and restricted open-shell Hartree–Fock references (`RHF`, `UHF`, `ROHF`) [@roothaan1951new; @pople1954self; @roothaan1960self] with Superposition of Atomic Densities (`SAD`) initial guesses and Pulay's `DIIS` extrapolation [@pulay1980convergence; @pulay1982improved], alongside canonical (`RMP2`, `UMP2`, `RMP3`, `UMP3`) [@moller1934note; @pople1976theoretical; @head1988mp2] and orbital-optimized (`OMP2`, `OMP3`) [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3] Møller–Plesset perturbation theory.

To overcome four-center electron repulsion integral (ERI) bottlenecks evaluated via `libcint` [@sun2015libcint], `mshqc` unifies three integral modes across all solvers: exact four-center tensors (`exact`) with Schwarz screening [@haser1989improvements], pivoted Cholesky decomposition (`cholesky`) [@beebe1977simplifications; @koch2003reduced], and auxiliary-basis Density Fitting (`df`) [@feyereisen1993use; @weigend2002fully]. High-dimensional contractions are dispatched to `TBLIS` [@matthews2017high], while orbital rotations in `OMP2` and `OMP3` are driven by a matrix-free Steihaug–Toint trust-region Second-Order SCF (`TrustRegionSOSCF`) optimizer.

# Statement of need

Orbital-optimized Møller–Plesset methods (`OMP2`, `OMP3`) relax reference orbitals in the presence of electron correlation, restoring Hellmann–Feynman satisfaction for one-electron properties and eliminating artificial spin contamination in open-shell radicals [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3]. However, each macro-iteration requires re-transforming two-electron integrals and evaluating $\mathcal{O}(N^5)$ (`OMP2`) or $\mathcal{O}(N^6)$ (`OMP3`) contractions, making naive implementations memory- and compute-bound.

Researchers and students prototyping orbital-optimization algorithms or three-index tensor factorizations face a steep barrier to entry in large production suites due to deep abstraction layers. `mshqc` provides an inspectable C++20 codebase where SCF solvers, `cholesky`/`df` integral transformers, `TBLIS` contractions, and trust-region `SOSCF` optimizers are unified under a two-tier API: users can run multi-method pipelines in one line via `mshqc.run()` or instantiate C++ classes directly in Python to inspect intermediate tensors.

# State of the field

Production packages such as `Psi4` [@smith2020psi4] (`fnocc`/`dfocc`) and `PySCF` [@sun2020recent] provide comprehensive canonical and orbital-optimized perturbation theory, but modifying their internal integral transformation pipelines or Hessian preconditioners requires navigating hundreds of thousands of lines of code. Conversely, lightweight educational codes typically implement only canonical `RHF` and `RMP2` in pure Python, lacking open-shell references (`UHF`/`ROHF`), third-order correlation (`MP3`/`OMP3`), point-group symmetry, or three-index integrals. `mshqc` bridges this gap with a compact C++20 architecture that matches or exceeds production speeds on `OMP2` and `OMP3` workloads.

# Software design

`mshqc` is structured into four C++20 layers exposed to Python via `nanobind` [@jakob2022nanobind] with zero-copy `Eigen` [@guennebaud2010eigen] interoperability and automatic `GIL` release:

1. **Symmetry & Basis Infrastructure (`Molecule`, `BasisSet`, `PointGroup`, `PetiteList`):** Parses 538 bundled Gaussian basis sets (`.gbs`), including Dunning `cc-pVXZ` [@dunning1989gaussian] and `-RI`/`-JKFIT` auxiliary bases [@weigend2002fully], with contraction normalization [@helgaker2000molecular], canonical axis alignment, and `PetiteList` symmetry reduction [@atkins1970tables].
2. **Unified Integral Engine (`IntegralEngine`, `Screening`, `CholeskyERI`, `DensityFittingERI`):** Evaluates integrals via `libcint` [@sun2015libcint] under a unified `eri_method` (`"exact"`, `"cholesky"`, `"df"`), reducing storage from $\mathcal{O}(N^4)$ to $\mathcal{O}(N_{	ext{aux}} N^2)$ three-center tensors $B_{pq}^Q$ with optional `HDF5` out-of-core storage [@hdfgroup1997hdf5].
3. **Self-Consistent Field (`RHF`, `UHF`, `ROHF`):** Shares a common `BaseSCF` architecture with `SAD` guesses, Schwarz screening, `DIIS` extrapolation, and rank-projected `df` exchange contractions.
4. **Correlated & Orbital-Optimized Solvers (`RMP2`, `UMP2`, `OMP2`, `RMP3`, `UMP3`, `OMP3`):** Dispatches four-index contractions to `TBLIS` [@matthews2017high] and three-index contractions to cache-blocked `OpenMP` + `BLAS` kernels [@dagum1998openmp], coupled with `TrustRegionSOSCF` orbital optimization.

# Validation and performance

All solvers in `mshqc` were validated against `Psi4` [@smith2020psi4] across closed-shell (`Ne`, $	ext{H}_2	ext{O}$) and open-shell (`Li`, `C`, `O`) systems in the `cc-pVTZ` (`cc-pVTZ-RI`) basis at a $10^{-9}	ext{ Ha}$ threshold. Across all 60 benchmark runs (`5 systems × 3 ERI modes × 4 methods`), `mshqc` reproduces `Psi4` energies within $|\Delta E| \le 1.27\ \mu	ext{Ha}$ (with 54 of 60 runs matching to $0.00\ \mu	ext{Ha}$ at $10^{-8}	ext{ Ha}$ precision, and all 20 `df` runs within $|\Delta E| \le 0.01\ \mu	ext{Ha}$).

As shown in Figure 1 and Table 1, `mshqc` achieves wall-clock speedups of $2.4	imes$ to $18.6	imes$ on `OMP2` and $2.8	imes$ to $11.6	imes$ on `OMP3` over `Psi4`, converging the orbital gradient norm $\|\mathbf{g}\|_2 < 10^{-5}	ext{ Ha}$ in 3 to 4 macro-iterations. On $	ext{H}_2	ext{O}$ ($N = 58$, $N_{	ext{aux}} = 141$), switching from `exact` to `cholesky` and `df` integrals reduces `OMP3` wall-clock time from $0.797	ext{ s}$ to $0.705	ext{ s}$ and $0.358	ext{ s}$.

![Benchmark evaluation of `mshqc` vs `Psi4` in the `cc-pVTZ` (`cc-pVTZ-RI`) basis: (A) `OMP3` wall-clock speedup ($t_{	ext{Psi4}} / t_{	ext{mshqc}}$) across `exact`, `cholesky`, and `df` integral modes; (B) Trust-Region SOSCF orbital gradient norm $\|\mathbf{g}\|_2$ convergence per macro-iteration.](benchmark_figure.pdf){ width=100% }

Table 1: Representative benchmark comparison between `mshqc` and `Psi4` (`cc-pVTZ` / `cc-pVTZ-RI`, threshold $10^{-9}	ext{ Ha}$), reporting total energy, absolute deviation $|\Delta E|$ ($\mu	ext{Ha}$), wall-clock times (s), and speedup.

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

`mshqc` is released under the Apache License 2.0 at <https://github.com/syahrulhidayat/mshqc>. Precompiled `manylinux_2_28` (`x86_64`) wheels and source distributions are available on PyPI (<https://pypi.org/project/mshqc/>) via `pip install mshqc`. Automated `pytest` validation suites, quickstart tutorials, and troubleshooting guides are provided in `tests/` and `docs/`.

# AI usage disclosure

Generative AI tools (Gemini) were used to assist with code refactoring, profiling analysis, and manuscript editing. All code and text were validated against analytical references by the author, who assumes full responsibility for this work.

# Acknowledgements

The author acknowledges the developers of `libcint` [@sun2015libcint], `TBLIS` [@matthews2017high], `PySCF` [@sun2020recent], and `Psi4` [@smith2020psi4] for foundational libraries and validation references.

# References
