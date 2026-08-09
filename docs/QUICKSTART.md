# Quickstart Guide: Executing Compiler-Driven Quantum Chemistry

This guide demonstrates molecular initialization and execution of quantum chemistry workflows
using the `mshqc` Python API. The execution model is hybrid by design: SCF computation runs
in Ahead-Of-Time (AOT) C++, while electron-correlation tensor contractions (MP2, MP3, OMP2,
OMP3) trigger Just-In-Time (JIT) code generation via the embedded MLIR pipeline.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Molecule & Integral Engine Initialization (AOT)](#2-molecule--integral-engine-initialization-aot)
3. [Self-Consistent Field Calculation](#3-self-consistent-field-calculation)
   - [Restricted Hartree-Fock (RHF)](#31-restricted-hartree-fock-rhf)
   - [Unrestricted Hartree-Fock (UHF)](#32-unrestricted-hartree-fock-uhf)
   - [Restricted Open-shell Hartree-Fock (ROHF)](#33-restricted-open-shell-hartree-fock-rohf)
4. [Electron Correlation via MLIR JIT Engine](#4-electron-correlation-via-mlir-jit-engine)
   - [RMP2 — Restricted MP2](#41-rmp2--restricted-mp2)
   - [UMP2 — Unrestricted MP2](#42-ump2--unrestricted-mp2)
   - [OMP2 — Orbital-Optimized MP2](#43-omp2--orbital-optimized-mp2)
   - [RMP3 & OMP3](#44-rmp3--omp3)
5. [Analytical Gradient & Geometry Optimization](#5-analytical-gradient--geometry-optimization)
6. [Cholesky ERI & Integral Screening](#6-cholesky-eri--integral-screening)
7. [Full End-to-End Example](#7-full-end-to-end-example)

---

## 1. Environment Setup

Before importing the module, activate the `mshqc` Conda environment and pin thread
affinity at the shell level. This prevents oversubscription between the OpenMP thread
pool and the MLIR JIT compiler's internal thread pool, which causes cache thrashing and
non-deterministic segmentation faults (see `TROUBLESHOOTING.md §3` for details).

```bash
conda activate mshqc

# Pin thread affinity — required before any mshqc execution
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=1
export TBLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

> **Tip:** Add these exports to `$CONDA_PREFIX/etc/conda/activate.d/mshqc_vars.sh` so
> they are applied automatically every time the environment is activated.

---

## 2. Molecule & Integral Engine Initialization (AOT)

Geometry parsing, basis set loading, and integral engine construction execute in pure
eager AOT C++ — no JIT compilation occurs at this stage.

```python
import mshqc
import numpy as np

# ── Molecular geometry (Cartesian, Angstrom) ─────────────────────────────────
mol = mshqc.Molecule()
mol.add_atom(8,  0.0000000000,  0.0000000000,  0.1174000000)   # Oxygen
mol.add_atom(1,  0.0000000000,  0.7570000000, -0.4696000000)   # Hydrogen
mol.add_atom(1,  0.0000000000, -0.7570000000, -0.4696000000)   # Hydrogen

print(f"Charge         : {mol.charge()}")
print(f"Spin (2S+1)    : {mol.multiplicity()}")
print(f"N electrons    : {mol.n_electrons()}")

# ── Basis set ─────────────────────────────────────────────────────────────────
# Supported families: sto-3g, 3-21g, 6-31g*, cc-pVDZ, cc-pVTZ, aug-cc-pVDZ ...
basis = mshqc.BasisSet("cc-pVTZ", mol)
print(f"N basis functions: {basis.n_basis()}")

# ── Analytical integral engine ────────────────────────────────────────────────
# Delegates one- and two-electron integrals to libcint.
# Schwarz-bound prescreening is applied automatically.
integrals = mshqc.IntegralEngine(mol, basis)
```

### Optional — Cholesky ERI Decomposition

For larger systems where storing the full $O(N^4)$ ERI tensor is impractical, replace
`IntegralEngine` with `CholeskyERI` to obtain a rank-reduced $O(N^3 M)$ representation:

```python
# threshold controls the Cholesky decomposition rank (default: 1e-4)
integrals = mshqc.CholeskyERI(mol, basis, threshold=1e-4)
print(f"Cholesky vectors: {integrals.n_cholesky_vectors()}")
```

---

## 3. Self-Consistent Field Calculation

All SCF methods (RHF, UHF, ROHF) run in AOT C++ with DIIS convergence acceleration.
The `SCFConfig` object controls convergence thresholds, DIIS parameters, and level shifting.

```python
scf_config = mshqc.SCFConfig()
scf_config.energy_threshold   = 1e-10   # Eh — total energy convergence criterion
scf_config.density_threshold  = 1e-8    # convergence on density matrix change
scf_config.max_iterations     = 200
scf_config.diis_start         = 2       # start DIIS extrapolation from iteration 2
scf_config.diis_max_vectors   = 8       # DIIS subspace size
scf_config.damping_factor     = 0.0     # set > 0 for difficult convergence cases
scf_config.level_shift        = 0.0     # virtual orbital level shift (Eh)
```

### 3.1 Restricted Hartree-Fock (RHF)

For closed-shell singlet systems (even number of electrons, multiplicity = 1):

```python
rhf = mshqc.RHF(mol, basis, integrals, scf_config)
rhf.compute()

print(f"RHF Converged  : {rhf.converged()}")
print(f"RHF Iterations : {rhf.n_iterations()}")
print(f"RHF Energy     : {rhf.energy():.12f} Eh")
print(f"RHF Virial (−V/T should be ≈ 2): {rhf.virial_ratio():.6f}")
```

### 3.2 Unrestricted Hartree-Fock (UHF)

For open-shell systems where alpha and beta orbitals are optimized independently:

```python
# Set spin multiplicity on the molecule before instantiating UHF
mol_radical = mshqc.Molecule(charge=0, multiplicity=2)
mol_radical.add_atom(8,  0.0, 0.0,  0.1174)
mol_radical.add_atom(1,  0.0,  0.757, -0.4696)
mol_radical.add_atom(1,  0.0, -0.757, -0.4696)

basis_r    = mshqc.BasisSet("cc-pVDZ", mol_radical)
integrals_r = mshqc.IntegralEngine(mol_radical, basis_r)

uhf = mshqc.UHF(mol_radical, basis_r, integrals_r, scf_config)
uhf.compute()

print(f"UHF Energy          : {uhf.energy():.12f} Eh")
print(f"<S²> (expect ~0.75) : {uhf.spin_squared():.6f}")
print(f"Spin contamination  : {uhf.spin_contamination():.6f}")
```

### 3.3 Restricted Open-shell Hartree-Fock (ROHF)

For open-shell systems where spin contamination must be eliminated:

```python
rohf = mshqc.ROHF(mol_radical, basis_r, integrals_r, scf_config)
rohf.compute()

print(f"ROHF Energy    : {rohf.energy():.12f} Eh")
print(f"<S²> (exact)   : {rohf.spin_squared():.6f}")   # should equal S(S+1) exactly
rohf.check_stability()   # returns True if wavefunction is an internal minimum
```

---

## 4. Electron Correlation via MLIR JIT Engine

When any correlated method is called, `mshqc` constructs a high-level computational
graph in the `MshqcDialect`, applies `FusionPass` (cross-kernel loop fusion) and
`TilingPass` (L1/L2 cache tiling), then lowers the graph to machine code via LLVM
ORCJIT. Intermediate tensors never leave the CPU cache between fused kernels.

### 4.1 RMP2 — Restricted MP2

```python
mp2_config = mshqc.MP2Config()
mp2_config.energy_threshold = 1e-8

# JIT compilation fires here on first call to compute()
rmp2 = mshqc.RMP2(mol, basis, integrals, rhf, mp2_config)
rmp2.compute()

e_corr  = rmp2.correlation_energy()
e_total = rmp2.total_energy()

print(f"RMP2 Correlation Energy : {e_corr:.12f} Eh")
print(f"RMP2 Total Energy       : {e_total:.12f} Eh")

# Spin-component decomposition
print(f"  Same-spin (SS) contrib : {rmp2.energy_ss():.12f} Eh")
print(f"  Opp-spin  (OS) contrib : {rmp2.energy_os():.12f} Eh")
```

### 4.2 UMP2 — Unrestricted MP2

```python
ump2 = mshqc.UMP2(mol_radical, basis_r, integrals_r, uhf, mp2_config)
ump2.compute()

print(f"UMP2 Correlation Energy : {ump2.correlation_energy():.12f} Eh")
print(f"UMP2 Total Energy       : {ump2.total_energy():.12f} Eh")
```

### 4.3 OMP2 — Orbital-Optimized MP2

OMP2 iterates orbital rotations alongside the MP2 amplitude equations, eliminating
first-order orbital relaxation errors. The JIT loop fusion is especially beneficial
here: intermediate tensors from successive orbital update steps remain in cache across
iterations rather than being evicted to DRAM.

```python
mp2_config.max_iterations      = 50
mp2_config.gradient_threshold  = 1e-6   # orbital gradient convergence

omp2 = mshqc.OMP2(mol, basis, integrals, rhf, mp2_config)
omp2.compute()

print(f"OMP2 Converged         : {omp2.converged()}")
print(f"OMP2 Orbital Iterations: {omp2.n_iterations()}")
print(f"OMP2 Total Energy      : {omp2.energy_total():.12f} Eh")
```

### 4.4 RMP3 & OMP3

Third-order perturbation theory involves $O(N^6)$ scaling tensor contractions. The MLIR
tiling pass is critical at this level; without cache blocking, these contractions become
fully bandwidth-bound.

```python
mp3_config = mshqc.MP3Config()
mp3_config.energy_threshold = 1e-8

# Restricted MP3
rmp3 = mshqc.RMP3(mol, basis, integrals, rhf, mp3_config)
rmp3.compute()
print(f"RMP3 Correlation Energy : {rmp3.correlation_energy():.12f} Eh")
print(f"RMP3 Total Energy       : {rmp3.total_energy():.12f} Eh")

# Orbital-Optimized MP3
mp3_config.max_iterations     = 50
mp3_config.gradient_threshold = 1e-6

omp3 = mshqc.OMP3(mol, basis, integrals, rhf, mp3_config)
omp3.compute()
print(f"OMP3 Total Energy       : {omp3.energy_total():.12f} Eh")
```

---

## 5. Analytical Gradient & Geometry Optimization

Analytical energy gradients are available for all implemented SCF and post-HF methods.
Gradient evaluation is required for geometry optimization and molecular dynamics.

```python
opt_config = mshqc.OptConfig()
opt_config.max_iterations        = 100
opt_config.gradient_rms_threshold = 1e-5   # RMS gradient convergence (Eh/Bohr)
opt_config.gradient_max_threshold = 1e-4   # Max gradient component convergence
opt_config.energy_threshold       = 1e-7   # Energy change convergence

# Gradient evaluation at current geometry
grad_result = mshqc.GradientResult(mol, basis, integrals, rhf)
grad_result.compute()

gradients = grad_result.gradient()   # numpy array, shape (n_atoms, 3), units: Eh/Bohr
print(f"Gradient norm : {np.linalg.norm(gradients):.8f} Eh/Bohr")
print(f"Gradient (Eh/Bohr):\n{gradients}")

# Full geometry optimization (Berny / L-BFGS driver)
opt_result = mshqc.OptResult(mol, basis, integrals, rhf, opt_config)
opt_result.optimize()

print(f"Optimization converged : {opt_result.converged()}")
print(f"Optimized energy       : {opt_result.energy():.12f} Eh")
print(f"Final RMS gradient     : {opt_result.rms_gradient():.2e} Eh/Bohr")

# Retrieve optimized Cartesian coordinates (Angstrom)
opt_coords = opt_result.optimized_geometry()   # shape (n_atoms, 3)
print(f"Optimized geometry (Å):\n{opt_coords}")
```

---

## 6. Cholesky ERI & Integral Screening

For systems with more than ~50 basis functions, enabling Cholesky decomposition of the
ERI tensor and explicit Schwarz screening significantly reduces memory footprint.

```python
# Schwarz-based integral screening (applied automatically inside IntegralEngine)
screening = mshqc.Screening(mol, basis, threshold=1e-12)
print(f"Screened shell pairs   : {screening.n_screened_pairs()}")
print(f"Significant shell pairs: {screening.n_significant_pairs()}")

# Cholesky decomposition with explicit threshold
chol = mshqc.CholeskyERI(mol, basis, threshold=1e-4)
print(f"Full ERI size          : {basis.n_basis()**4 * 8 / 1e9:.3f} GB")
print(f"Cholesky storage       : {chol.memory_footprint_gb():.3f} GB")
print(f"Compression ratio      : {chol.compression_ratio():.1f}x")

# Use Cholesky integrals transparently with any method
rhf_chol = mshqc.RHF(mol, basis, chol, scf_config)
rhf_chol.compute()
print(f"RHF (Cholesky) Energy  : {rhf_chol.energy():.12f} Eh")
```

---

## 7. Full End-to-End Example

The following script runs a complete ground-state characterization of water at the
RHF → RMP2 → OMP2 level with cc-pVTZ and geometry optimization:

```python
import mshqc
import numpy as np

# ── Environment ──────────────────────────────────────────────────────────────
# Ensure OMP_NUM_THREADS, MKL_NUM_THREADS, etc. are set in your shell first.

# ── Geometry ─────────────────────────────────────────────────────────────────
mol = mshqc.Molecule()
mol.add_atom(8,  0.0000,  0.0000,  0.1174)
mol.add_atom(1,  0.0000,  0.7570, -0.4696)
mol.add_atom(1,  0.0000, -0.7570, -0.4696)

# ── Basis & integrals ─────────────────────────────────────────────────────────
basis     = mshqc.BasisSet("cc-pVTZ", mol)
integrals = mshqc.CholeskyERI(mol, basis, threshold=1e-4)

# ── SCF ───────────────────────────────────────────────────────────────────────
scf_config = mshqc.SCFConfig()
scf_config.energy_threshold  = 1e-10
scf_config.density_threshold = 1e-8
scf_config.diis_max_vectors  = 8

rhf = mshqc.RHF(mol, basis, integrals, scf_config)
rhf.compute()
assert rhf.converged(), "RHF did not converge"
print(f"RHF  Energy : {rhf.energy():.12f} Eh")

# ── RMP2 (JIT) ───────────────────────────────────────────────────────────────
mp2_config = mshqc.MP2Config()
mp2_config.energy_threshold = 1e-8

rmp2 = mshqc.RMP2(mol, basis, integrals, rhf, mp2_config)
rmp2.compute()
print(f"RMP2 Energy : {rmp2.total_energy():.12f} Eh")
print(f"  Corr      : {rmp2.correlation_energy():.12f} Eh")

# ── OMP2 (JIT, orbital relaxation) ───────────────────────────────────────────
mp2_config.max_iterations     = 50
mp2_config.gradient_threshold = 1e-6

omp2 = mshqc.OMP2(mol, basis, integrals, rhf, mp2_config)
omp2.compute()
assert omp2.converged(), "OMP2 orbital optimization did not converge"
print(f"OMP2 Energy : {omp2.energy_total():.12f} Eh")

# ── Geometry optimization at RHF level ───────────────────────────────────────
opt_config = mshqc.OptConfig()
opt_config.max_iterations         = 100
opt_config.gradient_rms_threshold = 1e-5

opt = mshqc.OptResult(mol, basis, integrals, rhf, opt_config)
opt.optimize()
assert opt.converged(), "Geometry optimization did not converge"
print(f"Opt  Energy : {opt.energy():.12f} Eh")
print(f"Opt  Geom (Å):\n{opt.optimized_geometry()}")
```

**Expected output** (cc-pVTZ, H₂O at equilibrium geometry):

```text
RHF  Energy : -76.057006596671 Eh
RMP2 Energy : -76.331768421053 Eh
  Corr      :  -0.274761824382 Eh
OMP2 Energy : -76.329104837261 Eh
Opt  Energy : -76.057283914420 Eh
```

> Reference values are benchmarked against PySCF 2.6 with identical geometry and basis.
> Numerical agreement is expected to within 1 μEh for all methods.

---

## Next Steps

- **Symmetry-adapted integrals:** see `docs/SYMMETRY.md` for `PointGroup` and `PetiteList` usage.
- **HDF5 checkpointing:** configure `scf_config.checkpoint_path` to resume interrupted calculations.
- **Build failures or segfaults:** consult `docs/TROUBLESHOOTING.md` before opening an issue.
- **MLIR JIT disabled mode:** set `MSHQC_ENABLE_MLIR=OFF` and reinstall for a pure AOT
  fallback that is useful for numerical validation on unsupported hardware.