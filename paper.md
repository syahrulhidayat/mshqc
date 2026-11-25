---
title: 'mshqc: A Pure Thin-Core C++20/Python Quantum Chemistry Engine for O(V^4)-Memory-Free Orbital-Optimized Møller–Plesset Perturbation Theory'
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
date: 25 November 2026
bibliography: paper.bib
---

# Summary

`mshqc` (v2.0) is a lean, high-performance quantum chemistry library implementing wave-function-based electronic structure methods on an irreducible **9-file C++20 kernel core** (~5,500 LOC, zero-HDF5) paired with modular Python method orchestration via `nanobind` [@jakob2022nanobind]. It provides restricted, unrestricted, and restricted open-shell Hartree–Fock references (`RHF`, `UHF`, `ROHF`) [@roothaan1951new; @pople1954self; @roothaan1960self] with Superposition of Atomic Densities (`SAD`) guesses, diagonal-equilibrated Pulay `DIIS` extrapolation [@pulay1980convergence; @pulay1982improved], and pure-Python Abelian point-group symmetry labeling [@atkins1970tables]. On top of these references, `mshqc` implements canonical and semicanonical (`RMP2`, `UMP2`, `ROMP2`, `RMP3`, `UMP3`, `ROMP3`) [@moller1934note; @pople1976theoretical; @head1988mp2] and orbital-optimized perturbation theory (`OMP2`, `SCS-OMP2`, `SOS-OMP2`, `OMP2.5`, `OMP3`, $\kappa$-regularized `kappa-OOMP2`, and $\mathcal{O}(N^4)$ zero-$T_2$ Laplace-transformed `LT-SOS-OMP2`) [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3; @grimme2003improved; @jung2004scaled; @almlof1991elimination; @lee2018regularized] for both all-electron and frozen-core manifolds.

To remove four-center electron repulsion integral (ERI) and $\mathcal{O}(V^4)$ particle-particle ladder storage bottlenecks evaluated via `libcint` [@sun2015libcint], `mshqc` unifies three integral modes across all solvers: exact four-center tensors (`exact`) with Schwarz screening [@haser1989improvements], full-pivoted Cholesky decomposition (`cholesky`) [@beebe1977simplifications; @koch2003reduced], and auxiliary-basis Density Fitting (`df`) [@feyereisen1993use; @weigend2002fully]. High-dimensional contractions are dispatched to `TBLIS` [@matthews2017high] and `BLAS` Level-3 kernels, while orbital rotations and molecular geometry optimizations are driven by a matrix-free Steihaug–Toint trust-region Second-Order SCF (`TR-SOSCF`) solver with exact analytical DF nuclear gradients.

# Statement of need

Orbital-optimized Møller–Plesset methods (`OMP2`, `kappa-OOMP2`, `OMP3`) relax reference orbitals in the presence of electron correlation, restoring Hellmann–Feynman satisfaction for one-electron properties, regularizing small-gap divergences, and suppressing artificial spin contamination in open-shell radicals [@neese2009assessment; @bozkaya2011quadratically; @bozkaya2011omp3; @lee2018regularized]. However, each macro-iteration requires re-transforming two-electron integrals and evaluating $\mathcal{O}(N^5)$ (`OMP2`) or $\mathcal{O}(N^6)$ (`OMP3`) contractions, making naive implementations memory- and compute-bound.

Researchers and students prototyping orbital-optimization variants, regularizers, or three-index tensor factorizations face a steep barrier in large production suites due to deep abstraction layers. `mshqc` v2.0 separates compute-critical C++20 kernels from readable Python orchestration across three tiers: users can run multi-method pipelines in one line via `mshqc.run()`, compose or customize orbital optimizers and dynamic compute backends (`C++20`, `NumPy`, or optional `PyTorch` FP64 GPU) in `mshqc.methods`/`mshqc.backends`, or invoke coarse-grained C++20 kernels directly via `mshqc.kernels`.

# State of the field

Production suites such as `Psi4` [@smith2020psi4] (`fnocc`/`dfocc`) and `PySCF` [@sun2020recent] offer comprehensive electronic structure capabilities, yet modifying internal integral transformation pipelines, orbital-gradient terms, or Hessian preconditioners requires navigating hundreds of thousands of lines of code. Conversely, lightweight educational codes typically implement only canonical `RHF` and `RMP2` in pure Python without open-shell references (`UHF`/`ROHF`), third-order orbital optimization (`OMP3`), $\kappa$-regularization, Laplace-transformed $\mathcal{O}(N^4)$ scaling, analytical gradients, or three-index factorizations. `mshqc` bridges this gap with a compact 9-file C++20 core and clean Python layer that matches reference energies to machine precision while maintaining a low memory footprint and cross-platform wheel distribution.

# Software design

`mshqc` v2.0 adopts a zero-HDF5 **Pure Thin-Core** architecture exposed to Python via `nanobind` [@jakob2022nanobind] with zero-copy `Eigen` [@guennebaud2010eigen] interoperability and automatic `GIL` release (with the original monolithic v1.0 engine permanently archived under git tag `v1.0.5-thesis-archive`):

1. **Irreducible C++20 Kernel Core (9 `.cc` files, 44 bound kernel primitives):** Encapsulates Gaussian basis parsing (`Molecule`, `BasisSet`, 538 bundled `.gbs` bases including Dunning `cc-pVXZ` [@dunning1989gaussian] and `-RI`/`-JKFIT` auxiliary sets [@weigend2002fully] with contraction normalization [@helgaker2000molecular]), `libcint` [@sun2015libcint] AO integrals, unified `exact`/`cholesky`/`df` integral transformers, `OpenMP` [@dagum1998openmp] + `BLAS` Level-3 `J`/`K` builders, `TBLIS` [@matthews2017high] tensor contractions, $\mathcal{O}(V^4)$-free `OMP3` particle-particle ladder kernels, and `OpenMP` C++20 quadrature kernels for `LT-SOS-OMP2`. `CholeskyERI` uses unrestricted global diagonal pivoting with an unamplified Schwarz screening cutoff ($\tau_{\text{screen}} = \min(10^{-12}, 10^{-4}\tau_{\text{CD}})$) to avoid $1/\sqrt{D_{\max}}$ error amplification.
2. **Modular Python Orchestration (`mshqc.methods`):** Implements `DF`/`Cholesky`/`Exact` `RHF`, `UHF`, and `ROHF` (with Pure-Python SAD guess, zero-GEMM diagonal Iter-0 exchange, and Pulay DIIS), unified `TR-SOSCF` orbital optimization (`OMP2`, `SCS-OMP2`, `SOS-OMP2`, `LT-SOS-OMP2`, degeneracy-safe L'Hôpital `kappa-OOMP2`, `OMP2.5`, `OMP3`), analytical DF nuclear gradients (all-electron & frozen-core), Eckart-projected geometry optimization, harmonic frequencies + RRHO thermochemistry, `DF-CIS`/`TDHF` excited states, pure-Python Abelian symmetry labeling, Natural Orbitals/NOONs, Mulliken/Löwdin charges, `.molden` export, `.npz` checkpoints, and stateless `PySCF` interoperability (`run_from_pyscf`).
3. **Pluggable Compute Backends (`mshqc.backends`):** Exposes a context-managed backend registry (`use_backend`) supporting the default C++20/TBLIS engine, a reference Pure-NumPy backend for rapid prototyping and numerical auditing, and an optional `PyTorch` CUDA/CPU FP64 backend with automatic C++20 fallback.

# Validation and performance

`mshqc` v2.0 is continuously verified via automated GitHub Actions CI (`34 passed, 2 skipped` on CPU runners for optional `PySCF`/`PyTorch` tests), a 26-system universal validation suite across all three ERI modes (`cc-pVDZ`: **78/78 PASS** in $42.29\text{ s}$ on 4 threads), and a 234-configuration `cc-pVTZ`/`cc-pVTZ-RI` solver suite (`7 systems × 3 ERI modes × 9–12 methods`, threshold $10^{-9}\text{ Ha}$, total solver wall time $76.83\text{ s}$). Across all closed- and open-shell benchmark systems (`Ne`, $\text{H}_2\text{O}$, `Li`, `C`, `O`), `mshqc` v2.0 reproduces `Psi4` [@smith2020psi4] reference energies to $0.00\ \mu\text{Ha}$ at $10^{-8}\text{ Ha}$ precision, while `exact` and `cholesky` ($\tau_{\text{CD}} = 10^{-9}$) agree to $\le 1.3 \times 10^{-9}\text{ Ha}$.

Table 1 reports representative `mshqc` v2.0 energies, wall-clock times (4 threads), and $\langle\hat{S}^2\rangle$ values in the `cc-pVTZ` (`cc-pVTZ-RI`) basis. Analytic orbital gradients for `OMP2` and `OMP3` agree with central finite differences to $2.27 \times 10^{-11}\text{–}1.49 \times 10^{-9}\text{ Ha/rad}$ ($\text{H}_2\text{O}$ and `C`), converging `TR-SOSCF` in 3–6 macro-iterations to $g_{\text{rms}} < 10^{-6}\text{ Ha}$. Analytical DF nuclear gradients (all-electron and frozen-core `OMP2` and `kappa-OOMP2` on $\text{H}_2\text{O}$ and `OH`) match central finite differences to $3.48 \times 10^{-8}\text{–}6.46 \times 10^{-8}\text{ Ha/Bohr}$ with translational invariance $\le 7.11 \times 10^{-15}$. The $\mathcal{O}(N^4)$ `LT-SOS-OMP2` engine matches its $\mathcal{O}(N^5)$ `SOS-OMP2` reference to $|\Delta E| \le 2.27 \times 10^{-13}\text{ Ha}$ ($\text{C}_6\text{H}_6$, 114 basis functions), and the Pure-NumPy backend matches C++20 kernels to $|\Delta E| = 2.84 \times 10^{-14}\text{ Ha}$.

![Benchmark evaluation of `mshqc` vs `Psi4` in the `cc-pVTZ` (`cc-pVTZ-RI`) basis: (A) `OMP3` wall-clock speedup ($t_{\text{Psi4}} / t_{\text{mshqc}}$) across `exact`, `cholesky`, and `df` modes; (B) Trust-Region SOSCF orbital gradient norm $\|\mathbf{g}\|_2$ convergence per macro-iteration; (C) ERI tensor RAM scaling comparing four-center `exact` ($\mathcal{O}(N^4)$) against three-center `cholesky` ($\mathcal{O}(M N^2)$) and `df` ($\mathcal{O}(N_{\text{aux}} N^2)$).](docs/assets/benchmark_figure.pdf){ width=100% }

Table 1: Representative `mshqc` v2.0.0 solver benchmarks (`cc-pVTZ` / `cc-pVTZ-RI`, $\tau = 10^{-9}\text{ Ha}$, 4 threads), reporting total energy, wall-clock time (s), macro-iterations, and spin expectation value $\langle\hat{S}^2\rangle$.

|System|ERI|Method|Energy (Ha)|Time (s)|Iter|$\langle\hat{S}^2\rangle$|
|:--|:--|:--|--:|--:|--:|--:|
|Ne ($^1S$)|exact|RHF|-128.53186164|0.023|10|0.000000|
|Ne ($^1S$)|exact|OMP3|-128.81027068|0.050|3|0.000000|
|Ne ($^1S$)|df|OMP2|-128.81022440|0.008|3|0.000000|
|Ne ($^1S$)|df|OMP3|-128.81042564|0.030|3|0.000000|
|H$_2$O ($^1A_1$)|exact|OMP2|-76.33406054|0.224|4|0.000000|
|H$_2$O ($^1A_1$)|exact|OMP3|-76.33711056|0.530|4|0.000000|
|H$_2$O ($^1A_1$)|cholesky|OMP3|-76.33711056|1.085|4|0.000000|
|H$_2$O ($^1A_1$)|df|OMP2|-76.33402331|0.040|4|0.000000|
|H$_2$O ($^1A_1$)|df|kappa-OOMP2|-76.32973119|0.057|4|0.000000|
|H$_2$O ($^1A_1$)|df|LT-SOS-OMP2|-76.32966314|0.120|4|0.000000|
|H$_2$O ($^1A_1$)|df|OMP3|-76.33725847|0.274|4|0.000000|
|C ($^3P$, UHF)|df|UHF|-37.69163610|0.019|11|2.009038|
|C ($^3P$, UHF)|df|OMP3|-37.78281240|0.088|4|2.005662|
|O ($^3P$, UHF)|df|UHF|-74.81174765|0.020|12|2.007199|
|O ($^3P$, UHF)|df|OMP3|-74.98111725|0.112|4|2.004788|
|OH ($^2\Pi$, UHF)|df|UHF|-75.41925555|0.033|16|0.756048|
|OH ($^2\Pi$, UHF)|df|OMP2|-75.63258148|0.057|6|0.753706|
|OH ($^2\Pi$, UHF)|df|OMP3|-75.64435319|0.389|5|0.753670|
|O$_2$ ($^3\Sigma_g^-$, UHF)|df|UHF|-149.67517176|0.115|14|2.042555|
|O$_2$ ($^3\Sigma_g^-$, UHF)|df|OMP2|-150.14712919|0.147|5|2.006996|
|O$_2$ ($^3\Sigma_g^-$, UHF)|df|OMP3|-150.13330241|2.131|4|2.016366|

# Research impact statement

`mshqc` serves as the primary computational engine for electronic-structure research at the Department of Physics, Universitas Gadjah Mada, investigating $\mathcal{O}(V^4)$-memory-free orbital-optimized Møller–Plesset perturbation theory (`OMP2`, `kappa-OOMP2`, `LT-SOS-OMP2`, and `OMP3`) via rank-reduced tensor factorizations and spin-symmetry restoration in open-shell radicals [@hidayat2026thesis]. By refactoring the particle-particle ladder two-particle reduced density matrix (2-RDM) through an $\mathcal{O}(O^3V)$ intermediate ($Y_{ijmf}^{ooov}$) and symmetry-packed occupied-pair slicing, `mshqc` compresses AO ERI storage on $\text{H}_2\text{O}$/`cc-pVQZ` ($N = 115$, $N_{\text{aux}} = 242$) by $108.36\times$ ($1334.39\text{ MB}$ to $12.32\text{ MB}$), reduces peak MO tensor memory in `DF-OMP3` by $72.32\times$ ($2234.04\text{ MB}$ to $30.89\text{ MB}$), and keeps total process Peak RSS (`VmHWM`) at $284.87\text{ MB}$ ($206.73\text{ MB}$ net working RAM). On $\text{H}_2\text{O}$/`cc-pVTZ`, `DF-OMP3` lowers net working RAM by $3.38\times$ relative to `exact` ($216.02\text{ MB}$ to $63.92\text{ MB}$).

Across open-shell doublet and triplet radicals (`Li`, `C`, `O`, `OH`, and biradical $\text{O}_2$) benchmarked against `UHF` and `ROHF`/`ROMP2`/`ROMP3` limits, orbital relaxation in `mshqc` eliminates $33.49\%$ to $83.57\%$ of artificial spin contamination $\Delta\langle\hat{S}^2\rangle$. Furthermore, cross-auditing `mshqc` against `Psi4` across 150 `cc-pVTZ` solver configurations and a 507-point `cc-pVDZ` validation suite uncovered an upstream wavefunction orbital-update bug in `Psi4`'s `dfocc` module (`cd_omp2_manager`), where converged Cholesky-`OMP2` orbitals were not propagated back to the wavefunction object prior to evaluating $\langle\hat{S}^2\rangle$. Precompiled cross-platform wheels on PyPI (`pip install mshqc`) and bundled basis sets enable immediate adoption in quantum chemistry research and teaching workflows.

# Availability and documentation

`mshqc` is released under the Apache License 2.0 at <https://github.com/syahrulhidayat/mshqc>. Precompiled wheels for Linux (`manylinux_2_28` `x86_64`), macOS (`arm64`), and Windows (`x64`) along with source distributions are published on PyPI (<https://pypi.org/project/mshqc/>) and archived with permanent DOIs on Zenodo via automated GitHub Actions release pipelines. Automated `pytest` validation suites, benchmark runners (`benchmarks/`), quickstart tutorials, and troubleshooting guides are provided in `tests/` and `docs/`.

# AI usage disclosure

Generative AI tools (Gemini) were used to assist with code refactoring, profiling analysis, and manuscript editing. All code, equations, and benchmark numbers were independently executed and validated against analytical and numerical references by the author, who assumes full responsibility for this work.

# Acknowledgements

The author acknowledges the developers of `libcint` [@sun2015libcint], `TBLIS` [@matthews2017high], `PySCF` [@sun2020recent], and `Psi4` [@smith2020psi4] for foundational libraries and validation references.

# References
