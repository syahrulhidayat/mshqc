#!/usr/bin/env python3
"""
.modal/gpu_benchmark.py
Serverless NVIDIA CUDA GPU CI & Benchmark Runner untuk mshqc v2.0 di Modal.com.
Jalankan lokal atau dari GitHub Actions:
  MODAL_GPU=T4   modal run .modal/gpu_benchmark.py
  MODAL_GPU=A100 modal run .modal/gpu_benchmark.py --heavy
"""
from __future__ import annotations

import os
from pathlib import Path
import modal

GPU_TYPE = os.environ.get("MODAL_GPU", "T4")

app = modal.App("mshqc-v2-cuda-ci")

# Bangun image Linux + CUDA + PyTorch + deps mshqc (di-cache otomatis oleh Modal setelah build ke-1)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "cmake", "ninja-build", "build-essential", "pkg-config",
        "libopenblas-dev", "liblapacke-dev", "libeigen3-dev", "libomp-dev",
    )
    .pip_install(
        "torch",
        "numpy",
        "scipy",
        "pytest",
        "nanobind>=1.9.0",
        "setuptools>=77.0.0",
        "wheel",
    )
    .run_commands(
        # Compile & install libcint v6.1.2 + TBLIS v1.2.0 ke /usr/local (ter-cache di layer image Modal)
        "git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint",
        "cmake -S /tmp/libcint -B /tmp/libcint/build -GNinja "
        "-DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON "
        "-DCMAKE_C_FLAGS='-O3 -fPIC -std=gnu17' -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=OFF",
        "cmake --build /tmp/libcint/build -j4 && cmake --install /tmp/libcint/build",
        "git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis",
        "cd /tmp/tblis && git submodule update --init --recursive && "
        "sed -i 's/SIGSTKSZ/8192/g' src/external/catch/catch.hpp && "
        "for f in src/external/marray/include/short_vector.hpp src/external/marray/include/range.hpp src/util/thread.h; do sed -i '1i #include <limits>' $f; done && "
        "sed -i '1i #include <stdlib.h>' src/external/tci/tci/task_set.c && "
        "./configure --prefix=/usr/local --enable-static --disable-shared --with-pic --without-hwloc "
        "--enable-config=haswell,zen,core2,reference "
        "CFLAGS='-O3 -fPIC -std=gnu17' CXXFLAGS='-O3 -fPIC -std=c++17' && "
        "make -j4 bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS= && "
        "make install bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS=",
        "rm -rf /tmp/libcint /tmp/tblis",
        # Ambil basis sets mshqc-basis
        "git clone --depth 1 https://github.com/syahrulhidayat/mshqc-basis.git /opt/mshqc-basis",
    )
    .add_local_dir(".", remote_path="/workspace", ignore=[".git", "build", "dist", "__pycache__", ".cache*"])
)


@app.function(image=image, gpu=GPU_TYPE, cpu=4.0, memory=16384, timeout=900)
def run_cuda_suite(heavy: bool = False) -> str:
    import subprocess
    import time
    import torch

    os.chdir("/workspace")
    os.environ["MSHQC_BASIS_PATH"] = "/opt/mshqc-basis/data/basis"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TBLIS_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"

    print(f"=== Modal GPU Info: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) ===", flush=True)
    subprocess.run(["python", "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."], check=True)
    subprocess.run(["python", "-m", "pytest", "tests/test_thin_core_stage12_polish.py", "-v", "-s"], check=True)

    import mshqc
    from mshqc.backends import enable_torch_backend, use_backend

    enable_torch_backend(device="cuda", name="cuda", fallback_to_cpp=True)

    systems = [
        ("C6H6 (114 BF, FC)", mshqc.molecule("C6H6"), "cc-pVDZ", True),
    ]
    if heavy:
        systems.append(("C6H6 (264 BF, FC)", mshqc.molecule("C6H6"), "cc-pVTZ", True))

    lines = [
        f"GPU: {torch.cuda.get_device_name(0)}",
        "| System | Basis | t_RHF CPU | t_RHF GPU | t_LT-SOS CPU | t_LT-SOS GPU | Speedup LT-SOS | |dE| (Ha) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, mol, basis, fc in systems:
        calc_cpu = mshqc.MSHQCCalculator(mol, basis_name=basis, eri_method="df", freeze_core=fc)
        t0 = time.perf_counter()
        scf_c = calc_cpu.run_scf()
        t_rhf_c = time.perf_counter() - t0
        t0 = time.perf_counter()
        lt_c = calc_cpu.run_lt_sos_omp2(scf_c, grad_thresh=1e-5)
        t_lt_c = time.perf_counter() - t0

        calc_gpu = mshqc.MSHQCCalculator(mol, basis_name=basis, eri_method="df", freeze_core=fc)
        # Warm-up CUDA context
        _ = torch.eye(4, dtype=torch.float64, device="cuda") @ torch.eye(4, dtype=torch.float64, device="cuda")
        torch.cuda.synchronize()
        with use_backend("cuda"):
            t0 = time.perf_counter()
            scf_g = calc_gpu.run_scf()
            torch.cuda.synchronize()
            t_rhf_g = time.perf_counter() - t0
            t0 = time.perf_counter()
            lt_g = calc_gpu.run_lt_sos_omp2(scf_g, grad_thresh=1e-5)
            torch.cuda.synchronize()
            t_lt_g = time.perf_counter() - t0

        de = abs(lt_c.energy_total - lt_g.energy_total)
        sp = t_lt_c / max(t_lt_g, 1e-6)
        assert de < 1e-9, f"CUDA parity mismatch: {de}"
        lines.append(
            f"| {label} | {basis} | {t_rhf_c:.2f}s | {t_rhf_g:.2f}s | {t_lt_c:.2f}s | {t_lt_g:.2f}s | {sp:.2f}x | {de:.2e} |"
        )

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    return report


@app.local_entrypoint()
def main(heavy: bool = False) -> None:
    report = run_cuda_suite.remote(heavy=heavy)
    Path("cuda_gpu_benchmark_summary.md").write_text(report + "\n", encoding="utf-8")
