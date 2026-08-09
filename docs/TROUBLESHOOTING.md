# Troubleshooting & Compiler Mitigation Guide

This document defines architectural resolutions for build failures, linkage errors, and hardware
anomalies at runtime associated with the C++20 and MLIR/LLVM JIT infrastructure of `mshqc`.
Each section maps a concrete failure mode to its root cause and a validated remediation procedure.

---

## Table of Contents

1. [Hermetic Environment & CMake Path Resolution Leakage](#1-hermetic-environment--cmake-path-resolution-leakage)
2. [Shared Library Linkage & ABI Incompatibilities](#2-shared-library-linkage--abi-incompatibilities)
3. [Runtime Race Conditions & Thread Pool Oversubscription](#3-runtime-race-conditions--thread-pool-oversubscription)
4. [JIT Compiler & Graph Lowering Segmentation Faults](#4-jit-compiler--graph-lowering-segmentation-faults)
5. [Memory Allocator Conflicts (jemalloc vs. glibc malloc)](#5-memory-allocator-conflicts-jemalloc-vs-glibc-malloc)
6. [Cholesky ERI Decomposition Numerical Instability](#6-cholesky-eri-decomposition-numerical-instability)
7. [nanobind Import Failures & Python ABI Mismatch](#7-nanobind-import-failures--python-abi-mismatch)
8. [MKL Threading Layer Conflicts](#8-mkl-threading-layer-conflicts)
9. [HDF5 Checkpoint I/O Corruption](#9-hdf5-checkpoint-io-corruption)
10. [DIIS Divergence in SCF Iterations](#10-diis-divergence-in-scf-iterations)

---

## 1. Hermetic Environment & CMake Path Resolution Leakage

### Root Cause

If CMake fails to locate `llvmdev`, `mlir`, or third-party tensor libraries such as `libcint`
and `tblis`, this indicates **environment leakage** — the OS system compiler (`/usr/bin/c++`)
is overriding the Conda toolchain, causing CMake's `find_package()` to resolve against system
paths rather than the hermetic `conda-forge` prefix.

### Symptom

```text
CMake Error at CMakeLists.txt:
  Could not find a package configuration file provided by "MLIR"
  or "LLVM" with any of the following names:
    MLIRConfig.cmake
    mlir-config.cmake
```

### Resolution

Force absolute injection of Conda environment variables to prevent system path leakage.
Always build from within an activated `mshqc` Conda environment:

```bash
conda activate mshqc

export CMAKE_PREFIX_PATH=$CONDA_PREFIX

cmake -B build \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm \
  -DMLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir \
  -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++ \
  -DCMAKE_BUILD_TYPE=Release
```

> **Note:** On macOS with Apple Silicon, replace `x86_64-conda-linux-gnu-c++` with
> `arm64-apple-darwin20.0.0-c++` or check `$CONDA_PREFIX/bin/` for the exact toolchain
> binary name.

---

## 2. Shared Library Linkage & ABI Incompatibilities

### Root Cause

Late-stage linkage failures (`-l` resolution failures) typically stem from one of two causes:
(a) missing RPATH entries that prevent the dynamic linker from locating `.so` files at runtime,
or (b) **ABI incompatibilities** between objects compiled by different compilers (e.g., mixing
GCC-compiled `libcint` against a Clang-compiled `mshqc` core). Conda packages from `conda-forge`
are ABI-matched; mixing them with system-level `apt`/`dnf` packages breaks this guarantee.

### Symptom

```text
/usr/bin/ld: cannot find -lcint
/usr/bin/ld: undefined reference to `tblis_tensor_contract'
ImportError: libMLIR.so.17: cannot open shared object file: No such file or directory
```

### Resolution

Do not mix system-installed dependencies with the hermetic `conda-forge` environment.
Perform a full purge of Python build caches and CMake artifacts before recompiling:

```bash
pip cache purge
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
rm -rf build/ dist/ *.egg-info/
pip install -v -e .
```

If the linker still cannot resolve `libcint` or `tblis`, verify that the Conda packages
are present and their prefix matches:

```bash
conda list | grep -E "libcint|tblis|llvmdev|mlir"
ls $CONDA_PREFIX/lib/libcint*
ls $CONDA_PREFIX/lib/libtblis*
```

---

## 3. Runtime Race Conditions & Thread Pool Oversubscription

### Root Cause

`mshqc` drives parallel computation through three independent subsystems simultaneously:
quantum chemistry kernels (OpenMP), dense linear algebra (MKL), and dynamic code generation
(LLVM ORCJIT). Without explicit thread affinity constraints, total spawned threads can exceed
the number of physical CPU cores, causing **oversubscription**: L3 cache thrashing, excessive
context switching, and non-deterministic segmentation faults.

### Symptom

- `Segmentation fault (core dumped)` triggered immediately upon initializing an $O(N^5)$
  tensor contraction in the MP2 or MP3 modules.
- CPU utilization reaches 100% but FLOP throughput stalls; the process may freeze entirely.
- Intermittent results across repeated runs with identical inputs (non-deterministic behavior).

### Resolution

Pin thread affinity before launching any Python script. The auto-tuning logic in `bindings.cc`
attempts this at import time, but explicit environment variable declaration guarantees
determinism and overrides any library-internal defaults:

```bash
export OMP_NUM_THREADS=$(nproc)   # Allow OpenMP to use all physical cores
export MKL_NUM_THREADS=1          # Disable MKL's internal threading (defer to OpenMP)
export TBLIS_NUM_THREADS=1        # Disable TBLIS's internal threading
export OPENBLAS_NUM_THREADS=1     # Disable OpenBLAS threading if present
export OMP_PROC_BIND=close        # Bind threads to physically adjacent cores
export OMP_PLACES=cores           # Restrict thread placement to core granularity
```

Add these exports to your shell profile (`~/.bashrc` or `~/.zshrc`) or to a session
activation script to make the settings persistent across all `mshqc` runs.

---

## 4. JIT Compiler & Graph Lowering Segmentation Faults

### Root Cause

JIT execution in `mshqc` depends on dynamic memory allocation for machine code emission
via LLVM ORCJIT. If `FusionPass` or `TilingPass` generates `memref` descriptors with
alignment assumptions that do not match the target CPU's cache line size or SIMD vector
width, the ORCJIT executable memory mapper will raise an access violation during
instruction execution.

### Symptom

```text
LLVM ERROR: Broken function found, compilation aborted.
Segmentation fault (core dumped)
```

Stack traces reference one or more of:

```text
mlir::ExecutionEngine::invoke(...)
llvm::orc::RTDyldObjectLinkingLayer::emit(...)
mlir::linalg::LinalgOp::emitError(...)
```

### Resolution

**Step 1 — Isolate the failure layer.** Recompile with the MLIR JIT backend disabled to
establish a numerical ground truth using pure eager C++ execution (TBLIS/Eigen):

```bash
MSHQC_ENABLE_MLIR=OFF pip install -v -e .
python your_script.py
```

If the eager build runs without error, the fault is isolated to the MLIR transformation
passes and not to the quantum chemistry algorithms themselves.

**Step 2 — Adjust tiling parameters.** Open `src/compiler/Passes/TilingPass.cc` and reduce
the tile sizes to match your CPU's actual L1/L2 cache capacity. A conservative starting
point for a 32 KB L1 / 256 KB L2 architecture:

```cpp
// TilingPass.cc — conservative tile sizes for 32 KB L1 cache
static constexpr int64_t kTileI = 32;
static constexpr int64_t kTileJ = 32;
static constexpr int64_t kTileK = 16;
```

Query your CPU's cache topology before tuning:

```bash
lscpu | grep -E "L1d|L2|L3"
# or
getconf LEVEL1_DCACHE_SIZE
```

**Step 3 — Re-enable MLIR and retest** after adjusting tile sizes:

```bash
MSHQC_ENABLE_MLIR=ON pip install -v -e .
python your_script.py
```

---

## 5. Memory Allocator Conflicts (jemalloc vs. glibc malloc)

### Root Cause

`mshqc` uses `jemalloc` as its primary memory allocator to reduce arena fragmentation during
repeated tensor allocation and deallocation. If `jemalloc` is preloaded alongside another
allocator (e.g., `tcmalloc` from a system-level Google PerfTools installation), allocator
symbol conflicts will cause undefined behavior or a crash at the first large allocation.

### Symptom

```text
jemalloc: FATAL ERROR: detected multiple jemalloc instances
Aborted (core dumped)
```

or silent heap corruption producing NaN values in energy outputs.

### Resolution

Ensure only one allocator is active. Unset any `LD_PRELOAD` entries that inject a competing
allocator and verify the build links exclusively against the Conda-provided `jemalloc`:

```bash
echo $LD_PRELOAD   # should be empty or reference only $CONDA_PREFIX paths
unset LD_PRELOAD

# Confirm jemalloc is the linked allocator
ldd $(python -c "import mshqc; print(mshqc.__file__)") | grep jemalloc
```

---

## 6. Cholesky ERI Decomposition Numerical Instability

### Root Cause

The `CholeskyERI` engine applies a pivoted Cholesky factorization to the four-center
electron repulsion integral (ERI) tensor. For highly diffuse basis sets (e.g., aug-cc-pVTZ
and above) or near-linear-dependent basis sets on heavy atoms, near-zero pivot values
can accumulate floating-point error, causing the decomposition to either fail silently
(truncated rank) or raise a non-positive-definite exception.

### Symptom

```text
mshqc.CholeskyERI: decomposition rank collapsed to 0 at shell block (i=14, j=14)
RuntimeError: Cholesky ERI: non-positive-definite pivot detected (value = -3.2e-14)
```

### Resolution

Tighten the Cholesky threshold or fall back to conventional four-center integrals for
diagnostic purposes:

```python
# Loosen Cholesky threshold (default: 1e-4; try 1e-3 for diffuse basis sets)
chol = mshqc.CholeskyERI(mol, basis, threshold=1e-3)

# Or disable Cholesky entirely and use conventional ERIs
integrals = mshqc.IntegralEngine(mol, basis, use_cholesky=False)
```

If the conventional path yields correct energies, the Cholesky threshold requires tuning
for the specific basis set and molecular system.

---

## 7. nanobind Import Failures & Python ABI Mismatch

### Root Cause

`nanobind` bindings are compiled against a specific CPython ABI tag (e.g., `cp311-cp311`).
If the active Python interpreter version differs from the one used during compilation, the
extension module will refuse to load.

### Symptom

```text
ImportError: /path/to/mshqc/_core.cpython-310-x86_64-linux-gnu.so:
  cannot open shared object file: No such file or directory
ModuleNotFoundError: No module named 'mshqc._core'
```

or, on version mismatch:

```text
ImportError: mshqc/_core.cpython-311-x86_64-linux-gnu.so: wrong ELF class: ELFCLASS32
```

### Resolution

Verify that the Python interpreter inside the active Conda environment matches the
version used to build the extension:

```bash
conda activate mshqc
python --version          # Must match the python= spec in environment.yml
which python              # Must resolve inside $CONDA_PREFIX/bin/

pip install -v -e .       # Rebuild against the active interpreter
```

If multiple Python versions are present in the environment, explicitly pin the version
in `environment.yml`:

```yaml
dependencies:
  - python=3.11   # pin to an exact minor version
```

---

## 8. MKL Threading Layer Conflicts

### Root Cause

Intel MKL ships multiple threading backends (`libmkl_gnu_thread`, `libmkl_intel_thread`,
`libmkl_sequential`). When `conda-forge` resolves MKL alongside `llvm-openmp`, it may
select an incompatible threading layer, causing a deadlock or silent performance regression
where MKL spawns its own thread pool that conflicts with the outer OpenMP region.

### Symptom

- SCF loop hangs indefinitely after the first Fock matrix build.
- `htop` shows all threads pinned at 100% with no forward progress.

### Resolution

Force MKL to use the GNU OpenMP threading layer and disable its internal thread pool:

```bash
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
```

Alternatively, add these to the Conda activation scripts for the environment:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
echo 'export MKL_THREADING_LAYER=GNU' >> $CONDA_PREFIX/etc/conda/activate.d/mshqc_vars.sh
echo 'export MKL_NUM_THREADS=1'       >> $CONDA_PREFIX/etc/conda/activate.d/mshqc_vars.sh
```

---

## 9. HDF5 Checkpoint I/O Corruption

### Root Cause

`mshqc` uses HDF5 for optional wavefunction checkpointing. If a run is interrupted (e.g.,
by a job scheduler wall-time limit or a `SIGKILL`) while an HDF5 dataset is being written,
the file's superblock may be left in an inconsistent state. Subsequent runs that attempt to
read the corrupted checkpoint will fail at the HDF5 layer.

### Symptom

```text
HDF5-DIAG: Error detected in HDF5 (1.12.2) thread 0:
  #000: H5F.c line 512 in H5Fopen(): unable to open file
  major: File accessibility
  minor: Unable to open file
```

### Resolution

Delete the corrupted checkpoint file and restart from scratch:

```bash
rm -f mshqc_checkpoint.h5
python your_script.py
```

To enable HDF5's built-in file integrity check before opening:

```python
import h5py
with h5py.File("mshqc_checkpoint.h5", "r", swmr=True) as f:
    print(list(f.keys()))  # will raise OSError on corruption
```

For long-running jobs, configure `mshqc` to write checkpoints atomically by enabling
the write-staging option in your script:

```python
scf_config = mshqc.SCFConfig()
scf_config.checkpoint_path = "mshqc_checkpoint.h5"
scf_config.checkpoint_atomic = True   # write to .tmp then rename
```

---

## 10. DIIS Divergence in SCF Iterations

### Root Cause

The Direct Inversion in the Iterative Subspace (DIIS) extrapolation requires a well-conditioned
error vector matrix. For strongly correlated systems, open-shell molecules with near-degenerate
orbitals, or an overly aggressive initial guess (e.g., core Hamiltonian for heavy elements),
the DIIS coefficient matrix can become singular, causing the SCF energy to oscillate or diverge.

### Symptom

```text
mshqc.DIIS: WARNING — DIIS subspace matrix condition number = 1.23e+16 (threshold: 1e+8)
SCF iteration 45: ΔE = +0.034291 Eh  (energy increased — DIIS diverging)
```

### Resolution

Reduce the DIIS subspace size, increase the damping factor, or switch to damped SCF
for the first several iterations before enabling DIIS:

```python
scf_config = mshqc.SCFConfig()
scf_config.energy_threshold  = 1e-8
scf_config.diis_start        = 3      # delay DIIS until iteration 3 (default: 1)
scf_config.diis_max_vectors  = 6      # reduce subspace from default 8
scf_config.damping_factor    = 0.40   # apply 40% damping before DIIS kicks in
scf_config.level_shift       = 0.30   # apply level shift (Eh) for near-degenerate cases

rhf = mshqc.RHF(mol, basis, integrals, scf_config)
rhf.compute()
```

For ROHF or UHF calculations on open-shell systems with severe convergence difficulties,
enable the stability analysis after convergence to confirm the solution is a true local minimum:

```python
rohf = mshqc.ROHF(mol, basis, integrals, scf_config)
rohf.compute()
rohf.check_stability()   # returns True if wavefunction is internally stable
```

---

## General Debugging Checklist

Before opening an issue, confirm the following:

1. `conda activate mshqc` is active in the current shell session.
2. `python --version` and `which python` both resolve inside `$CONDA_PREFIX`.
3. `echo $CMAKE_PREFIX_PATH` returns a path pointing to `$CONDA_PREFIX`.
4. Thread affinity environment variables are set as described in [Section 3](#3-runtime-race-conditions--thread-pool-oversubscription).
5. No `LD_PRELOAD` is injecting a competing memory allocator.
6. The build directory is clean (`rm -rf build/ dist/ *.egg-info/`).
7. `MSHQC_ENABLE_MLIR=OFF pip install -v -e .` succeeds and produces correct energies
   before re-enabling the MLIR JIT backend.

If all of the above pass and the failure persists, please open a GitHub Issue and attach:

- The full CMake configure log (`build/CMakeFiles/CMakeOutput.log`).
- The output of `conda list` from the active environment.
- The Python traceback or C-level stack trace (`gdb python core` for segfaults).
- The molecular system specification (geometry, basis set, method).