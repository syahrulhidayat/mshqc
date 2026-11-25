# Troubleshooting Guide (`mshqc` v2.0)

This document covers build failures, linkage issues, basis-set discovery, backend configuration, and convergence tuning in the Pure Thin-Core C++20 (`MSHQC_NO_HDF5=1`) and Python orchestration layers (`libcint`, `TBLIS`, `MKL`/`OpenBLAS`, `nanobind`, optional `PyTorch`).

---

## Table of Contents

1. [Basis Set Not Found (`MSHQC_BASIS_PATH`)](#1-basis-set-not-found-mshqc_basis_path)
2. [Hermetic Environment & CMake Path Resolution Leakage](#2-hermetic-environment--cmake-path-resolution-leakage)
3. [Shared Library Linkage & ABI Incompatibilities](#3-shared-library-linkage--abi-incompatibilities)
4. [Runtime Thread Pool Oversubscription (Linux / macOS / Windows)](#4-runtime-thread-pool-oversubscription-linux--macos--windows)
5. [Illegal Instruction / Segfaults from `-march=native`](#5-illegal-instruction--segfaults-from--marchnative)
6. [Memory Allocator Conflicts (`jemalloc` vs `glibc malloc`)](#6-memory-allocator-conflicts-jemalloc-vs-glibc-malloc)
7. [Cholesky & Density Fitting Threshold Tuning](#7-cholesky--density-fitting-threshold-tuning)
8. [`nanobind` Import Failures & Python ABI Mismatch](#8-nanobind-import-failures--python-abi-mismatch)
9. [MKL Threading Layer Conflicts](#9-mkl-threading-layer-conflicts)
10. [DIIS, Level-Shifting & Orbital-Optimization Convergence Tuning](#10-diis-level-shifting--orbital-optimization-convergence-tuning)
11. [PyTorch / CUDA FP64 Backend Fallback](#11-pytorch--cuda-fp64-backend-fallback)

---

## 1. Basis Set Not Found (`MSHQC_BASIS_PATH`)

### Root Cause

When building from source without cloning `mshqc-basis` into `data/basis/`, `BasisSet("cc-pVDZ", mol)` cannot locate `.gbs` files.

### Resolution

Clone `syahrulhidayat/mshqc-basis` and point `MSHQC_BASIS_PATH` to its `data/basis` folder (or copy the `.gbs` files into `data/basis/`):

```bash
git clone --depth 1 [https://github.com/syahrulhidayat/mshqc-basis.git](https://github.com/syahrulhidayat/mshqc-basis.git) ../mshqc-basis
export MSHQC_BASIS_PATH="$(pwd)/../mshqc-basis/data/basis"
```

---

## 2. Hermetic Environment & CMake Path Resolution Leakage

### Root Cause

If CMake fails to locate `libcint`, `tblis`, or `eigen`, the OS system compiler (`/usr/bin/c++`) may be overriding the Conda toolchain.

### Resolution

Always build inside an activated Conda/Micromamba environment and export `CMAKE_PREFIX_PATH` and `TBLIS_DIR`:

```bash
conda activate mshqc_314
export TBLIS_DIR="$HOME/tblis_install"
export CMAKE_PREFIX_PATH="$HOME/libcint_install:$HOME/tblis_install:$CONDA_PREFIX"
pip install -v -e . --no-build-isolation --no-deps
```

---

## 3. Shared Library Linkage & ABI Incompatibilities

### Root Cause

Late-stage linkage errors (`cannot find -lcint` or `undefined reference to tblis_*`) stem from stale CMake caches or mixing OS libraries with `conda-forge`.

### Resolution

```bash
rm -rf build/ dist/ *.egg-info/ python/*.egg-info/ python/mshqc/*.so
pip install -v -e . --no-build-isolation --no-deps
```

---

## 4. Runtime Thread Pool Oversubscription (Linux / macOS / Windows)

### Root Cause

`mshqc` parallelizes outer loops via OpenMP while invoking `MKL`/`OpenBLAS` and `TBLIS`. Nested thread pools can oversubscribe physical CPU cores.

### Resolution

`mshqc` automatically locks `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` at import time (and `TBLIS_NUM_THREADS=1` on macOS/Windows). For explicit pinning on Linux:

```bash
export OMP_NUM_THREADS=4
export TBLIS_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

## 5. Illegal Instruction / Segfaults from `-march=native`

### Root Cause

Local source builds default to `-march=native`. Moving binaries compiled on an AVX-512 host to an AVX2-only CPU causes `Illegal instruction (core dumped)`.

### Resolution

Build with portable AVX2 flags (official PyPI wheels already use portable AVX2 on Linux/Windows and `apple-m1` on macOS ARM64):

```bash
GITHUB_ACTIONS=true pip install -v -e . --no-build-isolation --no-deps
```

---

## 6. Memory Allocator Conflicts (`jemalloc` vs `glibc malloc`)

### Root Cause

Preloading competing allocators via `LD_PRELOAD` can trigger heap corruption or hangs on shutdown.

### Resolution

```bash
unset LD_PRELOAD
```

---

## 7. Cholesky & Density Fitting Threshold Tuning

### Root Cause

For diffuse basis sets (`aug-cc-pVXZ`) with near-linear dependencies, an overly tight Cholesky threshold may retain near-zero pivots, while a loose threshold degrades correlation accuracy.

### Resolution

Pass `threshold` directly in `mshqc.run()` or `MSHQCCalculator`:

```python
import mshqc

calc = mshqc.MSHQCCalculator(
    mshqc.molecule("H2O"),
    basis_name="aug-cc-pVDZ",
    eri_method="cholesky",
    threshold=1e-8,
)
```

---

## 8. `nanobind` Import Failures & Python ABI Mismatch

### Root Cause

The compiled `_mshqc.*.so` / `_mshqc.*.pyd` extension is bound to a specific Python minor version.

### Resolution

Verify `which python` matches your target environment and rebuild cleanly (`pip install -v -e . --no-build-isolation --no-deps`).

---

## 9. MKL Threading Layer Conflicts

### Root Cause

Mixing `libiomp5` and `libgomp` on Linux can stall Fock or GEMM calls.

### Resolution

```bash
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
```

---

## 10. DIIS, Level-Shifting & Orbital-Optimization Convergence Tuning

### Root Cause

Open-shell radicals, biradicals (`O2`), or stretched bonds may converge slowly during SCF or OMP2/OMP3 macro-iterations.

### Resolution

Enable `level_shift`, tighten `threshold`, or use `kappa-OMP2` to regularize small orbital denominators:

```python
import mshqc

calc = mshqc.MSHQCCalculator(
    mshqc.molecule("O2"),
    basis_name="cc-pVDZ",
    eri_method="df",
    threshold=1e-9,
    level_shift=0.25,
)
scf_res = calc.run_scf()
omp_res = calc.run_kappa_omp2(scf_res, kappa=1.45, grad_thresh=1e-6)
```

---

## 11. PyTorch / CUDA FP64 Backend Fallback

### Root Cause

Consumer GPUs without strong FP64 throughput or missing `torch` installations may fail if strict GPU execution is forced.

### Resolution

Keep `fallback_to_cpp=True` when registering the PyTorch backend so unsupported kernels fall back seamlessly to the C++20 core:

```python
from mshqc.backends import enable_torch_backend, use_backend

enable_torch_backend(device="auto", name="torch", fallback_to_cpp=True)
with use_backend("torch"):
    ...
```
