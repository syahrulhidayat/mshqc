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
    orcid: 0009-0005-4112-1807
    affiliation: 1
affiliations:
  - name: Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada, Yogyakarta, Indonesia
    index: 1
date: 20 October 2026
bibliography: paper.bib
---

# Summary

`mshqc` is a high-performance quantum chemistry library implementing wave-function-based electronic structure methods in a compact C++20 core with a Python interface built on `nanobind` [@jakob2022nanobind]. It provides restricted, unrestricted, and restricted open-shell Hartree–Fock references (`RHF`, `UHF`, `ROHF`) [@roothaan1951new; @pople1954self; @roothaan1960self] with Superposition of Atomic Densities (`SAD`) initial guesses and Pulay's `DIIS` extrapolation [@pulay1980convergence; @pulay1982improved], alongside canonical (`RMP2`, `UMP2`, `RMP3`, `UMP3`) [@moller1934note; @pople1976theoretical; @head1988mp2] and orbital-optimized (`OMP2`, `OMP3`) [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3] Møller–Plesset perturbation theory.

To overcome four-center electron repulsion integral (ERI) bottlenecks evaluated via `libcint` [@sun2015libcint], `mshqc` unifies three integral modes across all solvers: exact four-center tensors (`exact`) with Schwarz screening [@haser1989improvements], full-pivoted Cholesky decomposition (`cholesky`) [@beebe1977simplifications; @koch2003reduced], and auxiliary-basis Density Fitting (`df`) [@feyereisen1993use; @weigend2002fully]. High-dimensional contractions are dispatched to `TBLIS` [@matthews2017high] and `BLAS` Level-3 kernels, while orbital rotations in `OMP2` and `OMP3` are driven by a matrix-free Steihaug–Toint trust-region Second-Order SCF (`TrustRegionSOSCF`) optimizer.

# Statement of need

Orbital-optimized Møller–Plesset methods (`OMP2`, `OMP3`) relax reference orbitals in the presence of electron correlation, restoring Hellmann–Feynman satisfaction for one-electron properties and eliminating artificial spin contamination in open-shell radicals [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3]. However, each macro-iteration requires re-transforming two-electron integrals and evaluating $\mathcal{O}(N^5)$ (`OMP2`) or $\mathcal{O}(N^6)$ (`OMP3`) contractions, making naive implementations memory- and compute-bound.

Researchers and students prototyping orbital-optimization algorithms or three-index tensor factorizations face a steep barrier to entry in large production suites due to deep abstraction layers. `mshqc` provides an inspectable C++20 codebase where SCF solvers, `cholesky`/`df` integral transformers, `TBLIS` contractions, and trust-region `SOSCF` optimizers are unified under a two-tier API: users can run multi-method pipelines in one line via `mshqc.run()` or instantiate C++20 classes directly in Python to inspect intermediate tensors.

# State of the field

Production packages such as `Psi4` [@smith2020psi4] (`fnocc`/`dfocc`) and `PySCF` [@sun2020recent] provide comprehensive canonical and orbital-optimized perturbation theory, but modifying their internal integral transformation pipelines or Hessian preconditioners requires navigating hundreds of thousands of lines of code. Conversely, lightweight educational codes typically implement only canonical `RHF` and `RMP2` in pure Python, lacking open-shell references (`UHF`/`ROHF`), third-order correlation (`MP3`/`OMP3`), point-group symmetry, or three-index integrals. `mshqc` bridges this gap with a compact C++20 architecture that matches reference energies to machine precision while achieving multi-fold speedups on `OMP2` and `OMP3` workloads.

# Software design

`mshqc` is structured into four C++20 layers exposed to Python via `nanobind` [@jakob2022nanobind] with zero-copy `Eigen` [@guennebaud2010eigen] interoperability and automatic `GIL` release:

1. **Symmetry & Basis Infrastructure (`Molecule`, `BasisSet`, `PointGroup`, `PetiteList`):** Parses 538 bundled Gaussian basis sets (`.gbs`), including Dunning `cc-pVXZ` [@dunning1989gaussian] and `-RI`/`-JKFIT` auxiliary bases [@weigend2002fully], with contraction normalization [@helgaker2000molecular], canonical axis alignment, and `PetiteList` symmetry reduction [@atkins1970tables].
2. **Unified Integral Engine (`IntegralEngine`, `Screening`, `CholeskyERI`, `DensityFittingERI`):** Evaluates integrals via `libcint` [@sun2015libcint] under a unified `eri_method` (`"exact"`, `"cholesky"`, `"df"`), reducing storage from $\mathcal{O}(N^4)$ to $\mathcal{O}(N_{\text{aux}} N^2)$ three-center tensors $B_{pq}^Q$ with optional `HDF5` out-of-core storage [@hdfgroup1997hdf5]. `CholeskyERI` employs unrestricted global diagonal pivoting across one- and two-center shell pairs with an unamplified Schwarz screening cutoff ($\tau_{\text{screen}} = \min(10^{-12}, 10^{-4}\tau_{\text{CD}})$) to prevent $1/\sqrt{D_{\max}}$ error amplification.
3. **Self-Consistent Field (`RHF`, `UHF`, `ROHF`):** Shares a common `BaseSCF` architecture with `SAD` guesses, lock-free `incore` quartet caching, `DIIS` extrapolation, spin-resolved Frobenius density convergence ($\Delta P = \sqrt{\|\Delta \mathbf{P}^\alpha\|_F^2 + \|\Delta \mathbf{P}^\beta\|_F^2}$), and post-convergence canonical Fock synchronization without `DIIS` mixing.
4. **Correlated & Orbital-Optimized Solvers (`RMP2`, `UMP2`, `OMP2`, `RMP3`, `UMP3`, `OMP3`):** Dispatches four-index contractions to `TBLIS` [@matthews2017high] and three-index `Z`-vector response contractions to `OpenMP` + `BLAS` Level-3 kernels [@dagum1998openmp], coupled with `TrustRegionSOSCF` orbital optimization.

# Validation and performance

All solvers in `mshqc` were validated against `Psi4` [@smith2020psi4] across closed-shell (`Ne`, $\text{H}_2\text{O}$) and open-shell (`Li`, `C`, `O`) systems in the `cc-pVTZ` (`cc-pVTZ-RI`) basis at a $10^{-9}\text{ Ha}$ convergence and Cholesky threshold. Across all 75 benchmark runs (`5 systems × 3 ERI modes × 5 methods`: `RHF`/`UHF`, `RMP2`/`UMP2`, `RMP3`/`UMP3`, `OMP2`, `OMP3`), `mshqc` reproduces `Psi4` reference energies with zero deviation ($|\Delta E| = 0.00\ \mu\text{Ha}$ across all 75/75 runs at $10^{-8}\text{ Ha}$ precision).

As shown in Figure 1 and Table 1, `mshqc` achieves wall-clock speedups of $3.27\times$ to $97.25\times$ on `OMP2` and $5.06\times$ to $81.11\times$ on `OMP3` over `Psi4` (and up to $20.32\times$ on `MP3`), converging the orbital gradient norm $\|\mathbf{g}\|_2 < 10^{-5}\text{ Ha}$ in 3 to 4 macro-iterations. On $\text{H}_2\text{O}$ ($N = 58$, $N_{\text{aux}} = 141$), three-center factorizations reduce ERI memory by $23.9\times$ ($90.5\text{ MB}$ to $3.8\text{ MB}$) and lower `OMP3` wall-clock time from $0.628\text{ s}$ (`exact`) to $0.250\text{ s}$ (`df`).

![Benchmark evaluation of `mshqc` vs `Psi4` in the `cc-pVTZ` (`cc-pVTZ-RI`) basis: (A) `OMP3` wall-clock speedup ($t_{\text{Psi4}} / t_{\text{mshqc}}$) across `exact`, `cholesky`, and `df` modes; (B) Trust-Region SOSCF orbital gradient norm $\|\mathbf{g}\|_2$ convergence per macro-iteration; (C) ERI tensor RAM scaling comparing four-center `exact` ($\mathcal{O}(N^4)$) against three-center `cholesky` ($\mathcal{O}(M N^2)$) and `df` ($\mathcal{O}(N_{\text{aux}} N^2)$).](benchmark_figure.pdf){ width=100% }

Table 1: Representative benchmark comparison between `mshqc` and `Psi4` (`cc-pVTZ` / `cc-pVTZ-RI`, threshold $10^{-9}\text{ Ha}$), reporting total energy, absolute deviation $|\Delta E|$ ($\mu\text{Ha}$), wall-clock times (s), and speedup.

|System|ERI|Method|Energy (Ha)|$|\Delta E|$|`mshqc` (s)|`Psi4` (s)|Speedup|
|:--|:--|:--|--:|--:|--:|--:|--:|
|Ne ($^1S$)|exact|RHF|-128.53186164|0.00|0.009|0.523|58.11x|
|Ne ($^1S$)|exact|RMP3|-128.80972987|0.00|0.019|0.386|20.32x|
|Ne ($^1S$)|exact|OMP2|-128.81022310|0.00|0.019|0.379|19.95x|
|Ne ($^1S$)|exact|OMP3|-128.81027068|0.00|0.038|0.736|19.37x|
|Ne ($^1S$)|cholesky|OMP3|-128.81027068|0.00|0.074|0.730|9.86x|
|Ne ($^1S$)|df|OMP2|-128.81022440|0.00|0.006|0.302|50.33x|
|Ne ($^1S$)|df|OMP3|-128.81042564|0.00|0.023|0.516|22.43x|
|H$_2$O ($^1A_1$)|exact|RHF|-76.05712742|0.00|0.101|0.574|5.68x|
|H$_2$O ($^1A_1$)|exact|OMP2|-76.33406054|0.00|0.354|1.158|3.27x|
|H$_2$O ($^1A_1$)|exact|OMP3|-76.33711056|0.00|0.628|3.180|5.06x|
|H$_2$O ($^1A_1$)|cholesky|OMP2|-76.33406054|0.00|0.256|1.588|6.20x|
|H$_2$O ($^1A_1$)|cholesky|OMP3|-76.33711056|0.00|0.831|4.537|5.46x|
|H$_2$O ($^1A_1$)|df|RMP2|-76.33220605|0.00|0.002|0.052|26.00x|
|H$_2$O ($^1A_1$)|df|OMP2|-76.33402331|0.00|0.057|0.470|8.25x|
|H$_2$O ($^1A_1$)|df|OMP3|-76.33725847|0.00|0.250|1.315|5.26x|
|Li ($^2S$, UHF)|exact|OMP2|-7.44391221|0.00|0.014|0.459|32.79x|
|Li ($^2S$, UHF)|cholesky|OMP3|-7.44569499|0.00|0.037|1.696|45.84x|
|Li ($^2S$, UHF)|df|OMP2|-7.44391392|0.00|0.008|0.778|97.25x|
|Li ($^2S$, UHF)|df|OMP3|-7.44569708|0.00|0.019|1.541|81.11x|
|C ($^3P$, UHF)|exact|OMP3|-37.78269153|0.00|0.194|1.076|5.55x|
|C ($^3P$, UHF)|cholesky|OMP3|-37.78269153|0.00|0.093|2.091|22.48x|
|C ($^3P$, UHF)|df|OMP2|-37.76763043|0.00|0.013|0.713|54.85x|
|C ($^3P$, UHF)|df|OMP3|-37.78281240|0.00|0.045|1.539|34.20x|
|O ($^3P$, UHF)|exact|OMP3|-74.98104071|0.00|0.191|1.181|6.18x|
|O ($^3P$, UHF)|cholesky|OMP2|-74.96657683|0.00|0.036|0.914|25.39x|
|O ($^3P$, UHF)|cholesky|OMP3|-74.98104071|0.00|0.125|3.119|24.95x|
|O ($^3P$, UHF)|df|OMP2|-74.96656224|0.00|0.015|0.924|61.60x|
|O ($^3P$, UHF)|df|OMP3|-74.98111725|0.00|0.059|2.275|38.56x|

# Availability and documentation

`mshqc` is released under the Apache License 2.0 at <https://github.com/syahrulhidayat/mshqc>. Precompiled `manylinux_2_28` (`x86_64`) wheels and source distributions are available on PyPI (<https://pypi.org/project/mshqc/>) via `pip install mshqc`. Automated `pytest` validation suites, quickstart tutorials, and troubleshooting guides are provided in `tests/` and `docs/`.

# AI usage disclosure

Generative AI tools (Gemini) were used to assist with code refactoring, profiling analysis, and manuscript editing. All code and text were validated against analytical references by the author, who assumes full responsibility for this work.

# Acknowledgements

The author acknowledges the developers of `libcint` [@sun2015libcint], `TBLIS` [@matthews2017high], `PySCF` [@sun2020recent], and `Psi4` [@smith2020psi4] for foundational libraries and validation references.

# References
