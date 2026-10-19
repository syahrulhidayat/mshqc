# Quickstart Guide: Running Quantum Chemistry Calculations with mshqc

This guide demonstrates both the **High-Level Pipeline API** (`mshqc.run`, `MSHQCSession`, `quick_calculation`) and the **Low-Level C++20 Kernel API** (`Molecule`, `BasisSet`, `IntegralEngine`, `RHF`/`UHF`/`ROHF`, `RMP2`/`UMP2`/`OMP2`, `RMP3`/`UMP3`/`OMP3`). Every code snippet in this guide is verified against the compiled `nanobind` interface in `python/bindings.cc`.

---

## Table of Contents

1. [Environment & Thread Affinity Setup](#1-environment--thread-affinity-setup)
2. [High-Level API: One-Liners & Workflow Sessions](#2-high-level-api-one-liners--workflow-sessions)
   - [Single & Multi-Method Pipelines with `mshqc.run()`](#21-single--multi-method-pipelines-with-mshqcrun)
   - [Single-Atom / Quick Calculations with `quick_calculation()`](#22-single-atom--quick-calculations-with-quick_calculation)
   - [Multi-Method Sessions with `MSHQCSession`](#23-multi-method-sessions-with-mshqcsession)
3. [Low-Level C++20 API: Step-by-Step Control](#3-low-level-c20-api-step-by-step-control)
   - [Molecule, Point Group Symmetry & Basis Initialization](#31-molecule-point-group-symmetry--basis-initialization)
   - [Self-Consistent Field: RHF, UHF, and ROHF](#32-self-consistent-field-rhf-uhf-and-rohf)
   - [Canonical Perturbation Theory: RMP2, UMP2, RMP3, and UMP3](#33-canonical-perturbation-theory-rmp2-ump2-rmp3-and-ump3)
   - [Orbital-Optimized Perturbation Theory: OMP2 and OMP3](#34-orbital-optimized-perturbation-theory-omp2-and-omp3)
4. [Standalone Cholesky ERI & Schwarz Screening Inspection](#4-standalone-cholesky-eri--schwarz-screening-inspection)
5. [Full End-to-End Benchmark Script](#5-full-end-to-end-benchmark-script)

---

## 1. Environment & Thread Affinity Setup

When `mshqc` is imported in Python, its C++20 auto-tuner (`mshqc_auto_tune::initialize_hpc_environment`) automatically detects physical CPU cores and sets `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` to prevent nested thread oversubscription. For explicit control in HPC batch scripts, set these variables before launching Python:

```bash
export OMP_NUM_THREADS=4
export TBLIS_NUM_THREADS=6
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
```

---

## 2. High-Level API: One-Liners & Workflow Sessions

### 2.1 Single & Multi-Method Pipelines with `mshqc.run()`

`mshqc.run()` accepts an XYZ string, a list of `(symbol/Z, x, y, z)` tuples, or an `mshqc.Molecule` object. It automatically aligns the molecule to its canonical point-group axes, constructs the `PetiteList`, and runs all requested methods on the same SCF reference:

```python
import mshqc

h2o_xyz = """
O  0.0000000000   0.0000000000   0.1173000000
H  0.0000000000   0.7572000000  -0.4692000000
H  0.0000000000  -0.7572000000  -0.4692000000
"""

# Run RHF -> RMP2 -> OMP2 -> OMP3 with Density Fitting (cc-pVTZ / cc-pVTZ-RI)
results = mshqc.run(
    h2o_xyz,
    method=["RHF", "RMP2", "OMP2", "OMP3"],
    basis="cc-pVTZ",
    eri_method="df",       # Options: "exact", "cholesky", "df"
    units="angstrom",      # Options: "angstrom", "bohr"
    threshold=1e-9,
    grad_thresh=1e-5,
)

for name, data in results.items():
    print(f"{name:<5}: {data['energy']:.10f} Ha (time: {data['time']:.3f} s)")
```

**Verified Output (`cc-pVTZ` / `cc-pVTZ-RI`, threshold `1e-9`):**

```text
RHF  : -76.0571197200 Ha (time: 0.047 s)
RMP2 : -76.3322060500 Ha (time: 0.005 s)
OMP2 : -76.3340233100 Ha (time: 0.144 s)
OMP3 : -76.3372584700 Ha (time: 0.358 s)
```

### 2.2 Single-Atom / Quick Calculations with `quick_calculation()`

`quick_calculation()` supports shorthand method prefixes such as `"df-omp3"`, `"cholesky-omp2"`, or `"exact"`:

```python
import mshqc

# Closed-shell Neon atom (1S, multiplicity = 1)
ne_res = mshqc.quick_calculation("Ne", basis="cc-pVTZ", method="df-omp3", threshold=1e-9)
print(f"Ne DF-OMP3 Energy: {ne_res.energy_total:.10f} Ha")

# Open-shell triplet Oxygen atom (3P, multiplicity = 3)
o_res = mshqc.quick_calculation("O", basis="cc-pVTZ", method="cholesky-omp3", multiplicity=3, threshold=1e-9)
print(f"O  Chol-OMP3 Energy: {o_res.energy_total:.10f} Ha")
```

### 2.3 Multi-Method Sessions with `MSHQCSession`

Use `MSHQCSession` when you want a formatted summary table and persistent access to all result objects (`session.results`) and timings (`session.timings`):

```python
import mshqc

session = mshqc.MSHQCSession(
    molecule="O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692",
    basis_name="cc-pVTZ",
    eri_method="cholesky",
    units="angstrom",
    threshold=1e-9,
    print_level=1,
)
session.run_complete_analysis(methods=["scf", "mp2", "omp2", "omp3"])
```

---

## 3. Low-Level C++20 API: Step-by-Step Control

### 3.1 Molecule, Point Group Symmetry & Basis Initialization

```python
import mshqc

# Initialize molecule in Angstrom (automatically converted to Bohr internally)
raw_mol = mshqc.Molecule(charge=0, multiplicity=1)
raw_mol.add_atom("O", 0.0000,  0.0000,  0.1173, units="angstrom")
raw_mol.add_atom("H", 0.0000,  0.7572, -0.4692, units="angstrom")
raw_mol.add_atom("H", 0.0000, -0.7572, -0.4692, units="angstrom")

# Optional: Detect point-group symmetry and align molecule to canonical axes
pg = mshqc.PointGroup(raw_mol)
pg.detect()
mol = pg.get_aligned_molecule()
print(f"Detected Point Group : {pg.get_symbol()} (order = {pg.get_order()})")

# Load Gaussian basis set and initialize libcint IntegralEngine
basis     = mshqc.BasisSet("cc-pVTZ", mol)
integrals = mshqc.IntegralEngine(mol, basis)
pl        = mshqc.PetiteList(basis, pg)
pl.build()

print(f"Basis functions      : {basis.n_basis_functions()}")
print(f"Nuclear repulsion    : {mol.nuclear_repulsion_energy():.10f} Ha")
```

### 3.2 Self-Consistent Field: RHF, UHF, and ROHF

```python
# Restricted Hartree-Fock (closed-shell H2O)
scf_conf = mshqc.SCFConfig(eri_method="exact", threshold=1e-9, scf_type="incore")
rhf      = mshqc.RHF(mol, basis, integrals, scf_conf, pg=pg, pl=pl)
scf_res  = rhf.compute()

assert scf_res.converged, "RHF did not converge"
print(f"RHF Energy     : {scf_res.energy_total:.10f} Ha ({scf_res.iterations} iterations)")

# Unrestricted & Restricted Open-Shell Hartree-Fock (triplet Carbon atom, 3P, mult=3)
mol_c = mshqc.Molecule(charge=0, multiplicity=3)
mol_c.add_atom("C", 0.0, 0.0, 0.0)
basis_c = mshqc.BasisSet("cc-pVTZ", mol_c)
ints_c  = mshqc.IntegralEngine(mol_c, basis_c)

uhf_res  = mshqc.UHF(mol_c, basis_c, ints_c, scf_conf).compute()
rohf_res = mshqc.ROHF(mol_c, basis_c, ints_c, scf_conf).compute()
print(f"Carbon UHF     : {uhf_res.energy_total:.10f} Ha")
print(f"Carbon ROHF    : {rohf_res.energy_total:.10f} Ha")
```

### 3.3 Canonical Perturbation Theory: RMP2, UMP2, RMP3, and UMP3

```python
mp2_conf = mshqc.MP2Config(eri_method="exact", threshold=1e-9)

# Closed-shell RMP2 and RMP3 on H2O
rmp2_res = mshqc.RMP2(mol, basis, integrals, scf_res, mp2_conf, pg=pg, pl=pl).compute()
rmp3_res = mshqc.RMP3(scf_res, rmp2_res, mp2_conf, integrals).compute()

print(f"RMP2 Total     : {rmp2_res.energy_total:.10f} Ha (Corr: {rmp2_res.energy_mp2_corr:.10f} Ha)")
print(f"  Same-Spin    : {rmp2_res.energy_mp2_ss:.10f} Ha")
print(f"  Opp-Spin     : {rmp2_res.energy_mp2_os:.10f} Ha")
print(f"RMP3 Total     : {rmp3_res.energy_total:.10f} Ha (MP3 Corr: {rmp3_res.energy_mp3_corr:.10f} Ha)")

# Open-shell UMP2 and UMP3 on triplet Carbon
ump2_res = mshqc.UMP2(mol_c, basis_c, ints_c, uhf_res, mp2_conf).compute()
ump3_res = mshqc.UMP3(uhf_res, ump2_res, mp2_conf, ints_c).compute()
print(f"UMP2 Total     : {ump2_res.energy_total:.10f} Ha")
print(f"UMP3 Total     : {ump3_res.energy_total:.10f} Ha")
```

### 3.4 Orbital-Optimized Perturbation Theory: OMP2 and OMP3

`OMP2` and `OMP3` relax the reference orbitals in the presence of correlation using the matrix-free Steihaug–Toint trust-region `SOSCF` solver (`opt_method="soscf"`):

```python
omp_conf = mshqc.MP2Config(
    eri_method="df",
    threshold=1e-9,
    gradient_threshold=1e-5,
    opt_method="soscf",
    max_iterations=50,
)

# Run DF-SCF first so the 3-center L_mat is populated in scf_df_res
scf_df_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-9)
scf_df_res  = mshqc.RHF(mol, basis, integrals, scf_df_conf, pg=pg, pl=pl).compute()

omp2_res = mshqc.OMP2(mol, basis, integrals, scf_df_res, omp_conf, pg=pg, pl=pl).compute()
omp3_res = mshqc.OMP3(mol, basis, integrals, scf_df_res, omp_conf, pg=pg, pl=pl).compute()

print(f"DF-OMP2 Energy : {omp2_res.energy_total:.10f} Ha (converged: {omp2_res.converged})")
print(f"DF-OMP3 Energy : {omp3_res.energy_total:.10f} Ha (converged: {omp3_res.converged})")
```

---

## 4. Standalone Cholesky ERI & Schwarz Screening Inspection

To inspect the Schwarz bound matrix or Cholesky-decomposed ERI vectors directly:

```python
import mshqc

mol = mshqc.build_molecule("O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692", units="angstrom")
basis = mshqc.BasisSet("cc-pVDZ", mol)
ints  = mshqc.IntegralEngine(mol, basis)

# 1. Schwarz Shell-Pair Screening
screen = mshqc.Screening(basis)
screen.compute(ints)
sig_pairs = screen.get_significant_pairs(threshold=1e-10)
print(f"Max Schwarz bound      : {screen.max_schwarz():.6f}")
print(f"Significant shell pairs: {len(sig_pairs)}")

# 2. Pivoted Cholesky ERI Decomposition
chol = mshqc.CholeskyERI(mol, basis, threshold=1e-9)
L_mat = chol.get_L_mat()   # NumPy view of shape (n_basis * n_basis, n_vectors)
print(f"Cholesky vectors       : {chol.n_vectors()} (L_mat shape: {L_mat.shape})")
```

---

## 5. Full End-to-End Benchmark Script

The following compact script reproduces the 5-system `mshqc` benchmark (`Neon`, `H2O`, `Lithium`, `Carbon`, `Oxygen`) across `exact`, `cholesky`, and `df` integral modes in the `cc-pVTZ` basis:

```python
import mshqc

SYSTEMS = [
    ("Neon",    "Ne 0.0 0.0 0.0", 0, 1, "rhf"),
    ("H2O",     "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692", 0, 1, "rhf"),
    ("Lithium", "Li 0.0 0.0 0.0", 0, 2, "uhf"),
    ("Carbon",  "C  0.0 0.0 0.0", 0, 3, "uhf"),
    ("Oxygen",  "O  0.0 0.0 0.0", 0, 3, "uhf"),
]

print(f"| {'System':<8} | {'ERI Mode':<8} | {'Method':<6} | {'Energy (Ha)':<16} | {'Time (s)':<8} |")
print("-" * 62)

for sys_name, xyz, chg, mult, ref in SYSTEMS:
    mp2_name = "RMP2" if ref == "rhf" else "UMP2"
    methods = [ref.upper(), mp2_name, "OMP2", "OMP3"]
    for eri_mode in ["exact", "cholesky", "df"]:
        res = mshqc.run(
            xyz,
            method=methods,
            basis="cc-pVTZ",
            eri_method=eri_mode,
            charge=chg,
            multiplicity=mult,
            reference=ref,
            units="angstrom",
            threshold=1e-9,
        )
        for m in methods:
            print(f"| {sys_name:<8} | {eri_mode.upper():<8} | {m:<6} | {res[m]['energy']:<16.8f} | {res[m]['time']:<8.3f} |")
```
