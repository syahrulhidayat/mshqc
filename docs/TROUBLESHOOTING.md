# Troubleshooting Guide

This document defines resolutions for build failures, linkage errors, and runtime anomalies in the C++20 core of `mshqc` and its library dependencies (`libcint`, `TBLIS`, `MKL`, `HDF5`, `jemalloc`, `nanobind`).

---

## Table of Contents

1. [Hermetic Environment & CMake Path Resolution Leakage](#1-hermetic-environment--cmake-path-resolution-leakage)
2. [Shared Library Linkage & ABI Incompatibilities](#2-shared-library-linkage--abi-incompatibilities)
3. [Runtime Race Conditions & Thread Pool Oversubscription](#3-runtime-race-conditions--thread-pool-oversubscription)
4. [Illegal Instruction / Segfaults from `-march=native`](#4-illegal-instruction--segfaults-from--marchnative)
5. [Memory Allocator Conflicts (jemalloc vs. glibc malloc)](#5-memory-allocator-conflicts-jemalloc-vs-glibc-malloc)
6. [Cholesky & Density Fitting Decomposition Threshold Tuning](#6-cholesky--density-fitting-decomposition-threshold-tuning)
7. [nanobind Import Failures & Python ABI Mismatch](#7-nanobind-import-failures--python-abi-mismatch)
8. [MKL Threading Layer Conflicts](#8-mkl-threading-layer-conflicts)
9. [HDF5 Out-of-Core I/O & File Locking](#9-hdf5-out-of-core-io--file-locking)
10. [DIIS & Orbital Optimization Convergence Tuning](#10-diis--orbital-optimization-convergence-tuning)

---

## 1. Hermetic Environment & CMake Path Resolution Leakage

### Root Cause

If CMake fails to locate third-party libraries such as `libcint`, `tblis`, or `eigen`, the OS system compiler (`/usr/bin/c++`) may be overriding the Conda toolchain.

### Resolution

Always build from within an activated Conda environment and pass `$CONDA_PREFIX` to `CMAKE_PREFIX_PATH`:

```bash
conda activate mshqc_314
export CMAKE_PREFIX_PATH="$HOME/libcint_install:$HOME/tblis_install:$CONDA_PREFIX"
pip install -v -e . --no-build-isolation
```

---

## 2. Shared Library Linkage & ABI Incompatibilities

### Root Cause

Late-stage linkage failures (`cannot find -lcint` or `undefined reference to tblis_tensor_contract`) stem from missing library search paths or mixing system packages with `conda-forge` libraries.

### Resolution

Purge build artifacts and rebuild cleanly against the active Conda environment:

```bash
rm -rf build/ dist/ *.egg-info/ python/*.egg-info/
pip install -v -e . --no-build-isolation
```

---

## 3. Runtime Race Conditions & Thread Pool Oversubscription

### Root Cause

`mshqc` parallelizes outer loops via OpenMP while invoking `MKL` and `TBLIS` kernels. If each library spawns its own thread pool, total threads can exceed physical CPU cores.

### Resolution

`mshqc` automatically locks `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` at import time. For explicit pinning in shell scripts:

```bash
export OMP_NUM_THREADS=4
export TBLIS_NUM_THREADS=6
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

## 4. Illegal Instruction / Segfaults from `-march=native`

### Root Cause

Local builds default to `-march=native`. Running a binary compiled on an AVX-512 host on an older AVX2 CPU will trigger `Illegal instruction (core dumped)`.

### Resolution

Rebuild with portable AVX2 architecture flags:

```bash
CMAKE_ARGS="-DCMAKE_CXX_FLAGS='-O3 -mavx2 -mfma -mno-avx512f -DNDEBUG'" pip install -v -e . --no-build-isolation
```

---

## 5. Memory Allocator Conflicts (jemalloc vs. glibc malloc)

### Root Cause

Preloading a competing allocator (`tcmalloc` or multiple `jemalloc` instances) via `LD_PRELOAD` can cause arena conflicts.

### Resolution

```bash
unset LD_PRELOAD
```

---

## 6. Cholesky & Density Fitting Decomposition Threshold Tuning

### Root Cause

For diffuse basis sets (`aug-cc-pVXZ`) with near-linear dependencies, an overly tight Cholesky threshold may include near-zero pivots, whereas a loose threshold reduces correlation energy accuracy.

### Resolution

Adjust `threshold` in `mshqc.run()` or `cholesky_threshold` / `df_threshold` in `SCFConfig` and `MP2Config`:

```python
import mshqc

scf_conf = mshqc.SCFConfig(eri_method="cholesky", threshold=1e-8)
mp2_conf = mshqc.MP2Config(eri_method="cholesky", threshold=1e-8)
```

---

## 7. nanobind Import Failures & Python ABI Mismatch

### Root Cause

The compiled `_mshqc.*.so` extension is tied to the exact Python minor version used during compilation.

### Resolution

Verify `which python` points to `$CONDA_PREFIX/bin/python` and re-run `pip install -v -e . --no-build-isolation`.

---

## 8. MKL Threading Layer Conflicts

### Root Cause

Incompatible OpenMP runtimes (`libiomp5` vs `libgomp`) can cause deadlocks during Fock builds.

### Resolution

```bash
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
```

---

## 9. HDF5 Out-of-Core I/O & File Locking

### Root Cause

When `scf_type="out_of_core"` is used with Density Fitting, three-center integrals are written to `df_tensor.h5`. On network filesystems (NFS/Lustre), OS file locking can raise an HDF5 access error, or an interrupted run can leave a stale `.h5` file.

### Resolution

Disable HDF5 file locking in the environment and remove any stale temporary `.h5` file before restarting:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
rm -f df_tensor.h5
```

For systems that fit in RAM, keep the default `scf_type="incore"` so that all three-index tensors remain in memory without touching disk.

---

## 10. DIIS & Orbital Optimization Convergence Tuning

### Root Cause

Open-shell radicals or stretched geometries may exhibit slow SCF or orbital-optimization convergence.

### Resolution

Adjust `diis_max_vectors` and `diis_threshold` on `SCFConfig`, or select `opt_method="soscf"` (Steihaug–Toint trust-region SOSCF) vs `"lbfgs"` on `MP2Config`:

```python
import mshqc

scf_conf = mshqc.SCFConfig(eri_method="df", threshold=1e-9, max_iterations=150)
scf_conf.diis_max_vectors = 10
scf_conf.diis_threshold   = 1e-2

mp2_conf = mshqc.MP2Config(
    eri_method="df",
    threshold=1e-9,
    gradient_threshold=1e-5,
    opt_method="soscf",
    max_iterations=50,
)
```
