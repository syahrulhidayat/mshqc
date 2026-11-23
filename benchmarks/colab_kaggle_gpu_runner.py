#!/usr/bin/env python3
"""
benchmarks/colab_kaggle_gpu_runner.py
Runner GPU CUDA FP64 mandiri (100% gratis tanpa kartu kredit) untuk Google Colab atau Kaggle.
Cara pakai di 1 sel Colab / Kaggle (setelah Runtime = GPU T4):
  !git clone -b hybrid-v2 --depth 1 https://github.com/syahrulhidayat/mshqc.git
  %cd mshqc
  !python benchmarks/colab_kaggle_gpu_runner.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def ensure_native_deps() -> None:
    if not Path("/usr/local/lib/libtblis.a").exists() or not Path("/usr/local/lib/libcint.a").exists():
        run_cmd(
            "apt-get update -qq && apt-get install -y -qq "
            "cmake ninja-build build-essential pkg-config libopenblas-dev liblapacke-dev libeigen3-dev libomp-dev"
        )
        run_cmd("rm -rf /tmp/libcint && git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint")
        run_cmd(
            "cmake -S /tmp/libcint -B /tmp/libcint/build -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local "
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS='-O3 -fPIC -std=gnu17' "
            "-DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=OFF && "
            "cmake --build /tmp/libcint/build -j4 && cmake --install /tmp/libcint/build"
        )
        run_cmd("rm -rf /tmp/tblis && git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis")
        run_cmd(
            "cd /tmp/tblis && git submodule update --init --recursive && "
            "sed -i 's/SIGSTKSZ/8192/g' src/external/catch/catch.hpp && "
            "for f in src/external/marray/include/short_vector.hpp src/external/marray/include/range.hpp src/util/thread.h; do sed -i '1i #include <limits>' $f; done && "
            "sed -i '1i #include <stdlib.h>' src/external/tci/tci/task_set.c && "
            "./configure --prefix=/usr/local --enable-static --disable-shared --with-pic --without-hwloc "
            "--enable-config=haswell,zen,core2,reference CFLAGS='-O3 -fPIC -std=gnu17' CXXFLAGS='-O3 -fPIC -std=c++17' && "
            "make -j4 bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS= && "
            "make install bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS="
        )
    if not Path("data/basis").exists():
        run_cmd("rm -rf /tmp/mshqc-basis && git clone --depth 1 https://github.com/syahrulhidayat/mshqc-basis.git /tmp/mshqc-basis")
        run_cmd("mkdir -p data/basis && cp -rf /tmp/mshqc-basis/data/basis/* data/basis/")


def main() -> None:
    import torch  # pyright: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU tidak terdeteksi. Aktifkan Runtime -> Change runtime type -> T4 GPU.")

    print(f"=== GPU Terdeteksi: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) ===", flush=True)
    ensure_native_deps()

    py = f'"{sys.executable}"'
    run_cmd(f'{py} -m pip install -q "nanobind>=1.9.0" pytest')
    run_cmd(f"CMAKE_BUILD_PARALLEL_LEVEL=4 {py} -m pip install --no-build-isolation --no-deps -e .")
    os.environ["MSHQC_BASIS_PATH"] = str(Path("data/basis").resolve())

    run_cmd(f"{py} -m pytest tests/test_thin_core_stage12_polish.py -v -s")

    import mshqc
    from mshqc.backends import enable_torch_backend, use_backend

    enable_torch_backend(device="cuda", name="cuda", fallback_to_cpp=True)
    mol = mshqc.molecule("C6H6")

    calc_c = mshqc.MSHQCCalculator(mol, basis_name="cc-pVDZ", eri_method="df", freeze_core=True)
    t0 = time.perf_counter()
    scf_c = calc_c.run_scf()
    t_rhf_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    lt_c = calc_c.run_lt_sos_omp2(scf_c, grad_thresh=1e-5)
    t_lt_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    o3_c = calc_c.run_omp3(scf_c, grad_thresh=1e-5)
    t_o3_c = time.perf_counter() - t0

    calc_g = mshqc.MSHQCCalculator(mol, basis_name="cc-pVDZ", eri_method="df", freeze_core=True)
    with use_backend("cuda"):
        t0 = time.perf_counter()
        scf_g = calc_g.run_scf()
        torch.cuda.synchronize()
        t_rhf_g = time.perf_counter() - t0

        t0 = time.perf_counter()
        lt_g = calc_g.run_lt_sos_omp2(scf_g, grad_thresh=1e-5)
        torch.cuda.synchronize()
        t_lt_g = time.perf_counter() - t0

        t0 = time.perf_counter()
        o3_g = calc_g.run_omp3(scf_g, grad_thresh=1e-5)
        torch.cuda.synchronize()
        t_o3_g = time.perf_counter() - t0

    print("\n=== C6H6 cc-pVDZ (114 BF, FC): CPU vs CUDA FP64 ===")
    print(f"RHF        : CPU={t_rhf_c:.2f}s | GPU={t_rhf_g:.2f}s | |dE|={abs(scf_c.energy_total - scf_g.energy_total):.2e}")
    print(f"LT-SOS-OMP2: CPU={t_lt_c:.2f}s | GPU={t_lt_g:.2f}s | |dE|={abs(lt_c.energy_total - lt_g.energy_total):.2e}")
    print(f"OMP3       : CPU={t_o3_c:.2f}s | GPU={t_o3_g:.2f}s | |dE|={abs(o3_c.energy_total - o3_g.energy_total):.2e}")


if __name__ == "__main__":
    main()