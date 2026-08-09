# Contributing to mshqc

Thank you for your interest in contributing to `mshqc`. This repository adopts strict
high-performance computing (HPC) software engineering standards. Every contribution is
evaluated against concrete hardware efficiency metrics — SIMD utilization, cache locality,
memory bandwidth reduction, and numerical precision — not against high-level programming
conventions alone.

This document is the authoritative reference for all contributors. Read it in full before
opening a Pull Request.

---

## Table of Contents

1. [Architectural Overview for Contributors](#1-architectural-overview-for-contributors)
2. [Development Environment Setup](#2-development-environment-setup)
3. [MLIR Dialect Modifications & TableGen (`MshqcOps.td`)](#3-mlir-dialect-modifications--tablegen-mshqcopstd)
4. [Compiler Pass Pipeline Modifications (`FusionPass`, `TilingPass`)](#4-compiler-pass-pipeline-modifications-fusionpass-tilingpass)
5. [Numerical Ground Truth & Precision Validation](#5-numerical-ground-truth--precision-validation)
6. [Bug Reports & JIT Anomaly Reporting](#6-bug-reports--jit-anomaly-reporting)
7. [Pull Request Requirements](#7-pull-request-requirements)
8. [Code Style & C++20 Conventions](#8-code-style--c20-conventions)
9. [Test Suite Execution Protocol](#9-test-suite-execution-protocol)
10. [Commit Message Convention](#10-commit-message-convention)

---

## 1. Architectural Overview for Contributors

`mshqc` operates as a **dual-engine framework**. Understanding the boundary between the two
execution paths is a prerequisite for any contribution:

| Engine | Execution Model | Scope | Validation Role |
|---|---|---|---|
| **Eager (AOT)** | Pre-compiled C++20 via TBLIS/Eigen/MKL | Geometry parsing, SCF iterations, DIIS, gradient logic | **Numerical ground truth** |
| **JIT (MLIR)** | LLVM ORCJIT via `MshqcDialect` | Electron-correlation tensor contractions (MP2, MP3, OMP2, OMP3) | Performance-critical path |

The eager engine is not a degraded fallback — it is the **absolute numerical reference**.
Any modification to the JIT path must produce energies that agree with the eager path to
within the precision tolerance defined in [Section 5](#5-numerical-ground-truth--precision-validation).

The novelty that `mshqc` contributes to the field lies entirely in the MLIR
cross-kernel loop fusion and cache-tiling infrastructure (`FusionPass.cc`,
`TilingPass.cc`). Contributions that touch these components are subject to the most
rigorous review criteria in this document.

---

## 2. Development Environment Setup

All development must occur inside the hermetic `conda-forge` environment. Do not use
system-level compilers or system-installed HPC libraries — ABI mismatches between
`apt`/`dnf` packages and `conda-forge` packages will produce silent numerical errors
or runtime crashes.

```bash
# Clone the repository
git clone https://github.com/syahrulhidayat/mshqc.git
cd mshqc

# Provision the hermetic build environment
conda env create -f environment.yml
conda activate mshqc

# Set thread affinity — required before any build or test invocation
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=1
export TBLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Build in editable mode with full verbosity
pip install -v -e .
```

To build with the MLIR JIT backend disabled (eager-only, for numerical validation):

```bash
MSHQC_ENABLE_MLIR=OFF pip install -v -e .
```

To re-enable:

```bash
MSHQC_ENABLE_MLIR=ON pip install -v -e .
```

---

## 3. MLIR Dialect Modifications & TableGen (`MshqcOps.td`)

Every new high-level tensor operation must be defined at the domain-specific IR
abstraction layer through the TableGen interface before it touches any downstream
lowering or Python binding.

### 3.1 Zero-Copy Abstraction Requirement

New operation definitions **must not** trigger implicit tensor copy instructions inside
the IR. Use `memref` descriptors exclusively to facilitate in-place data updates.
Introducing a hidden allocation (e.g., materializing a buffer to hold an intermediate
result that should be fused away) is a blocking defect and will cause the PR to be
rejected without further review.

Correct pattern — in-place update via `memref`:

```mlir
// Acceptable: result written directly into a caller-provided memref
mshqc.fock_build %density : memref<?x?xf64>, %fock : memref<?x?xf64>
```

Incorrect pattern — implicit allocation:

```mlir
// Rejected: operation materializes an unnamed intermediate buffer
%tmp = mshqc.fock_build %density : memref<?x?xf64> -> tensor<?x?xf64>
```

### 3.2 Independent Graph Verification via `mshqc-opt`

Before integrating a new dialect operation into the Python `nanobind` bindings, the full
graph lowering pipeline must be validated in isolation using the standalone compiler
utility:

```bash
# Dump the MLIR IR after dialect registration
mshqc-opt --mshqc-pipeline your_op.mlir

# Validate lowering to LLVM dialect without triggering JIT execution
mshqc-opt --convert-mshqc-to-llvm --verify-diagnostics your_op.mlir
```

The `--verify-diagnostics` flag enforces that every `expected-error` and
`expected-warning` annotation in the test IR is matched exactly. A clean run with no
unmatched diagnostics is required before the operation can be merged.

### 3.3 Permutation Symmetry Encoding

New ERI-level operations must encode the permutation symmetry
$(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda) = (\lambda\sigma|\mu\nu)$
directly in the TableGen trait list. Operations that ignore these symmetries will
quadruplicate memory traffic and defeat the purpose of the fusion infrastructure.

---

## 4. Compiler Pass Pipeline Modifications (`FusionPass`, `TilingPass`)

The loop transformation passes in `src/compiler/Passes/` are the most performance-critical
components in the repository. Modifications to these files are held to the highest
evidence standard.

### 4.1 Mandatory Hardware Profiling Evidence

**Every PR that modifies a loop tiling parameter, a fusion boundary, or a pass ordering
must include quantitative hardware counter evidence.** Theoretical or algorithmic
arguments alone are not sufficient for merge approval.

Required evidence format — attach the output of `perf stat` comparing the candidate
branch against `main` on an identical molecular system:

```bash
# Baseline measurement (main branch)
git checkout main && pip install -q -e .
perf stat -e L1-dcache-load-misses,LLC-load-misses,cache-misses,instructions \
  python benchmark/h2o_mp2_ccpvtz.py 2> baseline_perf.txt

# Candidate measurement (your branch)
git checkout your-branch && pip install -q -e .
perf stat -e L1-dcache-load-misses,LLC-load-misses,cache-misses,instructions \
  python benchmark/h2o_mp2_ccpvtz.py 2> candidate_perf.txt

diff baseline_perf.txt candidate_perf.txt
```

A PR is eligible for merge only if `L1-dcache-load-misses` and `LLC-load-misses` show
a statistically meaningful reduction (≥ 5% on the benchmark system) with no regression
in any other counter.

### 4.2 Tile Size Calibration

Tile sizes in `TilingPass.cc` must be calibrated to the L1 data cache size of the
target architecture, not hardcoded to arbitrary powers of two. Use `lscpu` to query
the hardware topology before proposing new tile parameters:

```bash
lscpu | grep -E "L1d|L2|L3|Socket|Core"
getconf LEVEL1_DCACHE_SIZE   # in bytes
```

Document the target CPU model, L1d size, and the derivation of tile dimensions in
the PR description. A tile configuration that overflows L1d is a correctness and
performance defect regardless of benchmark throughput numbers.

### 4.3 FusionPass Boundary Invariants

The `FusionPass` must preserve the following invariants across any modification:

- **No intermediate tensor escapes the fused loop nest.** A tensor that is written in
  one kernel and read in the next must remain in a register or L1 cache line — it must
  never be written to a named `memref` allocation between fused kernels.
- **Numerical associativity is preserved.** Floating-point reassociation introduced by
  vectorization must not shift energy values by more than the tolerance in Section 5.
- **Pass ordering is stable.** `FusionPass` must run before `TilingPass`. Reversing
  this order collapses the cache-locality guarantee.

---

## 5. Numerical Ground Truth & Precision Validation

JIT-generated machine code is susceptible to precision drift from aggressive instruction
scheduling, floating-point reassociation during vectorization, and FMA contraction.
The eager C++ engine (TBLIS microkernel path) is the absolute numerical reference for
all energy quantities.

### 5.1 Precision Tolerance

| Quantity | Tolerance |
|---|---|
| SCF total energy (per iteration) | ≤ $10^{-8}$ $E_h$ |
| MP2 / MP3 correlation energy | ≤ $10^{-8}$ $E_h$ |
| OMP2 / OMP3 orbital gradient norm | ≤ $10^{-6}$ $E_h$/Bohr |
| Analytical gradient component | ≤ $10^{-6}$ $E_h$/Bohr |

Any JIT-generated result that deviates from the eager reference beyond these thresholds
is a blocking defect. The PR will not be merged until the root cause (typically
a precision-unsafe reassociation in a vectorized loop) is identified and resolved.

### 5.2 Validation Protocol

Run the precision cross-check suite explicitly before opening a PR:

```bash
# Step 1 — generate eager (ground truth) reference values
MSHQC_ENABLE_MLIR=OFF pip install -q -e .
OMP_NUM_THREADS=1 pytest test/precision/ -v --tb=short 2>&1 | tee eager_ref.txt

# Step 2 — run JIT path and compare against reference
MSHQC_ENABLE_MLIR=ON pip install -q -e .
OMP_NUM_THREADS=1 pytest test/precision/ -v --tb=short 2>&1 | tee jit_result.txt

# Step 3 — diff the energy tables
python tools/compare_energies.py eager_ref.txt jit_result.txt --tolerance 1e-8
```

Attach the output of `compare_energies.py` to your PR description. A non-zero exit
code from this script is a hard block on merging.

### 5.3 Regression Benchmark Suite

All methods must pass the reference energy regression suite against established
quantum chemistry packages (PySCF 2.6, Psi4 1.9) before a PR is eligible for review:

```bash
OMP_NUM_THREADS=1 pytest test/regression/ -v --benchmark-reference pyscf
```

Regression failures against published reference values are treated as correctness bugs,
not performance issues.

---

## 6. Bug Reports & JIT Anomaly Reporting

### 6.1 Segmentation Fault in LLVM ORCJIT

Stack traces that reference `mlir::ExecutionEngine`, `llvm::orc::RTDyldObjectLinkingLayer`,
or `mlir::linalg::LinalgOp` indicate a failure inside the JIT lowering pipeline. When
filing such an issue, you **must** include all of the following:

1. **Abstract IR before lowering** — dump via:
   ```bash
   MSHQC_DUMP_IR=1 python your_script.py > ir_dump.mlir 2>&1
   ```
2. **Thread affinity parameters** at the time of the crash:
   ```bash
   echo "OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS TBLIS=$TBLIS_NUM_THREADS"
   ```
3. **C-level stack trace** from `gdb`:
   ```bash
   gdb -batch -ex run -ex bt python -- your_script.py 2>&1 | tail -60
   ```
4. **CPU topology** (`lscpu` output) and `conda list` from the active environment.
5. **Molecular system specification** — geometry (XYZ block), basis set name, and method.

Issues that omit any of the above will be closed with a request to provide the missing
information before reopening.

### 6.2 Numerical Anomaly (NaN / Inf / Energy Drift)

For non-crashing numerical issues, additionally provide:

- The iteration number and energy value at which divergence first appears.
- Whether the same system converges correctly under `MSHQC_ENABLE_MLIR=OFF`.
- The `SCFConfig` and `MP2Config` / `MP3Config` parameters used.

---

## 7. Pull Request Requirements

A PR is eligible for review only when **all** of the following are satisfied:

- [ ] The hermetic Conda environment (`environment.yml`) was used exclusively — no
      system-level compiler or library was involved.
- [ ] `OMP_NUM_THREADS=1 pytest test/ -v` passes with zero failures on both the eager
      and JIT build variants.
- [ ] For MLIR/pass modifications: `perf stat` hardware counter evidence is attached
      (see Section 4.1).
- [ ] For any new method or kernel: `compare_energies.py` output is attached showing
      JIT vs. eager deviation ≤ $10^{-8}$ $E_h$ (see Section 5.2).
- [ ] New TableGen operations have been validated with `mshqc-opt --verify-diagnostics`.
- [ ] Commit messages follow the convention in Section 10.
- [ ] `TROUBLESHOOTING.md` is updated if the change introduces a new known failure mode.

PRs that do not satisfy all checklist items will not be assigned a reviewer.

---

## 8. Code Style & C++20 Conventions

### C++ (Core & Passes)

- Standard: C++20 strictly. Use concepts, ranges, and `std::span` where appropriate.
- No raw owning pointers. Use `std::unique_ptr` / `std::shared_ptr` or RAII wrappers.
- All tensor dimensions must be validated at construction time via `assert` or
  `MSHQC_CHECK` — never at the call site.
- Format with `clang-format` using the project `.clang-format` file before committing:
  ```bash
  find src/ -name "*.cc" -o -name "*.h" | xargs clang-format -i
  ```

### Python Bindings (`nanobind`)

- Binding declarations live exclusively in `src/bindings/bindings.cc`.
- No Python-visible API may allocate heap memory inside the binding layer — all
  allocations belong to the C++ side and are transferred via zero-copy `memref` or
  `numpy` array views.
- Type stubs (`.pyi`) must be updated in `python/mshqc/` for every new binding.

### MLIR / TableGen

- All new ops: follow the naming convention `mshqc.<domain>_<verb>` (e.g.,
  `mshqc.eri_contract`, `mshqc.fock_build`).
- Every op must declare `NoSideEffect` or justify the omission explicitly in the
  TableGen comment.

---

## 9. Test Suite Execution Protocol

Always run the test suite with a single thread to eliminate non-determinism from
thread scheduling before adding parallelism back:

```bash
# Step 1 — single-threaded baseline (mandatory before any submission)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest test/ -v --tb=short

# Step 2 — parallel execution (smoke test for race conditions)
OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=1 pytest test/ -v --tb=short

# Step 3 — precision cross-check (JIT vs. eager)
MSHQC_ENABLE_MLIR=OFF pip install -q -e .
OMP_NUM_THREADS=1 pytest test/precision/ -v --tb=short 2>&1 | tee eager_ref.txt
MSHQC_ENABLE_MLIR=ON  pip install -q -e .
OMP_NUM_THREADS=1 pytest test/precision/ -v --tb=short 2>&1 | tee jit_result.txt
python tools/compare_energies.py eager_ref.txt jit_result.txt --tolerance 1e-8
```

A difference in behavior between Step 1 and Step 2 (results that pass single-threaded
but fail under parallelism) indicates a race condition and must be resolved before
the PR is opened.

---

## 10. Commit Message Convention

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
specification with the HPC-specific scopes listed below:

```
<type>(<scope>): <imperative summary under 72 characters>

[optional body: describe the hardware motivation, cache-level reasoning,
 or precision impact. Reference perf stat output if applicable.]

[optional footer: Closes #<issue>, Breaking-Change: <description>]
```

**Allowed types:** `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `chore`

**HPC-specific scopes:**

| Scope | Target Component |
|---|---|
| `mlir` | `MshqcOps.td`, dialect registration, graph lowering |
| `fusion` | `FusionPass.cc` and cross-kernel fusion logic |
| `tiling` | `TilingPass.cc` and cache-blocking parameters |
| `jit` | LLVM ORCJIT emission and memory mapping |
| `bindings` | `nanobind` interface (`bindings.cc`, `.pyi` stubs) |
| `scf` | RHF / UHF / ROHF / DIIS |
| `mp` | MP2, MP3, OMP2, OMP3 implementations |
| `integrals` | `libcint` interface, Cholesky ERI, Schwarz screening |
| `grad` | Analytical gradient and geometry optimization |
| `bench` | Benchmark scripts in `benchmark/` |
| `docs` | Documentation only |

**Examples:**

```
perf(tiling): reduce L1-dcache-load-misses 23% on H2O/cc-pVTZ MP2

Tile sizes reduced from (64,64,32) to (32,32,16) to fit 32 KB L1d cache
on Intel Alder Lake P-cores. perf stat evidence attached in PR #47.
```

```
fix(fusion): prevent intermediate tensor escape across ERI+Fock boundary

FusionPass was not recognizing the ERI -> Fock density contraction as
fusible because the AffineMap comparison used pointer equality instead
of structural equality. Replaced with AffineMap::isEqual(). Energy
deviation from eager reference drops from 3.2e-6 to 1.1e-11 Eh.

Closes #39
```

---

## Questions & Discussion

For architectural questions about the MLIR dialect design or pass pipeline, open a
GitHub Discussion rather than an Issue. Issues are reserved for confirmed bugs with
reproducible evidence. Feature proposals without a concrete hardware motivation and
profiling baseline will be closed and redirected to Discussions.