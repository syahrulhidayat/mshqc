#!/usr/bin/env python3
"""
benchmarks/kaggle_cuda_benchmark.py
Runner CUDA FP64 untuk Kaggle GPU (T4 gratis) berbasis PyTorch backend mshqc v2.0.
Menghasilkan cuda_gpu_benchmark_summary.md dan cuda_gpu_benchmark_raw.json
dengan skema identik .modal/gpu_benchmark.py (termasuk FP64 GEMM TFLOPS roofline).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def peak_cpu_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def ensure_native_deps() -> None:
    if not Path("/usr/local/lib/libtblis.a").exists() or not Path("/usr/local/lib/libcint.a").exists():
        run_cmd(
            "apt-get update -qq && apt-get install -y -qq "
            "cmake ninja-build build-essential pkg-config "
            "libopenblas-dev liblapacke-dev libeigen3-dev libomp-dev"
        )
        run_cmd("rm -rf /tmp/libcint && git clone --depth 1 -b v6.1.2 https://github.com/sunqm/libcint.git /tmp/libcint")
        run_cmd(
            "cmake -S /tmp/libcint -B /tmp/libcint/build -GNinja "
            "-DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON "
            "-DCMAKE_C_FLAGS='-O3 -fPIC -std=gnu17' -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DBUILD_SHARED_LIBS=OFF && "
            "cmake --build /tmp/libcint/build -j4 && cmake --install /tmp/libcint/build"
        )
        run_cmd("rm -rf /tmp/tblis && git clone --depth 1 -b v1.2.0 https://github.com/devinamatthews/tblis.git /tmp/tblis")
        run_cmd(
            "cd /tmp/tblis && git submodule update --init --recursive && "
            "sed -i 's/SIGSTKSZ/8192/g' src/external/catch/catch.hpp && "
            "for f in src/external/marray/include/short_vector.hpp src/external/marray/include/range.hpp src/util/thread.h; "
            "do sed -i '1i #include <limits>' $f; done && "
            "sed -i '1i #include <stdlib.h>' src/external/tci/tci/task_set.c && "
            "./configure --prefix=/usr/local --enable-static --disable-shared --with-pic --without-hwloc "
            "--enable-config=haswell,zen,core2,reference CFLAGS='-O3 -fPIC -std=gnu17' CXXFLAGS='-O3 -fPIC -std=c++17' && "
            "make -j4 bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS= && "
            "make install bin_PROGRAMS= noinst_PROGRAMS= check_PROGRAMS="
        )
    if not Path("data/basis").exists() or not any(Path("data/basis").glob("*.gbs")):
        run_cmd("rm -rf /tmp/mshqc-basis && git clone --depth 1 https://github.com/syahrulhidayat/mshqc-basis.git /tmp/mshqc-basis")
        run_cmd("mkdir -p data/basis python/mshqc/basis && cp -rf /tmp/mshqc-basis/data/basis/* data/basis/ && cp -rf /tmp/mshqc-basis/data/basis/* python/mshqc/basis/")


def main() -> None:
    parser = argparse.ArgumentParser(description="mshqc v2.0 Kaggle CUDA FP64 Benchmark Runner")
    parser.add_argument("--mode", choices=["smoke", "heavy", "cpc_scaling"], default="smoke")
    parser.add_argument("--out-dir", default="/kaggle/working")
    parser.add_argument("--git-sha", default=os.environ.get("MSHQC_GIT_SHA", "local"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU tidak terdeteksi di Kaggle. Pastikan enable_gpu=true.")

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TBLIS_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
    print(f"=== Kaggle GPU Info: {gpu_name} ({gpu_total_gb:.1f} GB, CUDA {torch.version.cuda}) ===", flush=True)

    ensure_native_deps()
    os.environ["MSHQC_BASIS_PATH"] = str(Path("data/basis").resolve())

    py = f'"{sys.executable}"'
    run_cmd(f'{py} -m pip install -q "nanobind>=1.9.0" pytest')
    run_cmd(f"CMAKE_BUILD_PARALLEL_LEVEL=4 {py} -m pip install --no-build-isolation --no-deps -e .")
    run_cmd(f"{py} -m pytest tests/test_thin_core_stage12_polish.py -v -s")

    import mshqc
    from mshqc.backends import enable_torch_backend, use_backend

    # Ukur throughput FP64 GEMM 2048x2048 untuk transparansi roofline CPC
    n_gemm = 2048
    a = torch.ones((n_gemm, n_gemm), dtype=torch.float64, device="cuda")
    b = torch.ones((n_gemm, n_gemm), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    tg0 = time.perf_counter()
    _ = a @ b
    torch.cuda.synchronize()
    fp64_tflops = (2.0 * (n_gemm ** 3)) / ((time.perf_counter() - tg0) * 1e12)
    del a, b
    torch.cuda.empty_cache()

    cuda_backend = enable_torch_backend(device="cuda", name="cuda", fallback_to_cpp=True)

    naphthalene = [
        ("C", 0.0000, 0.7100, 0.0), ("C", 0.0000, -0.7100, 0.0),
        ("C", 1.2430, 1.4040, 0.0), ("C", 2.4310, 0.7080, 0.0),
        ("C", 2.4310, -0.7080, 0.0), ("C", 1.2430, -1.4040, 0.0),
        ("C", -1.2430, 1.4040, 0.0), ("C", -2.4310, 0.7080, 0.0),
        ("C", -2.4310, -0.7080, 0.0), ("C", -1.2430, -1.4040, 0.0),
        ("H", 1.2450, 2.4900, 0.0), ("H", 3.3740, 1.2460, 0.0),
        ("H", 3.3740, -1.2460, 0.0), ("H", 1.2450, -2.4900, 0.0),
        ("H", -1.2450, 2.4900, 0.0), ("H", -3.3740, 1.2460, 0.0),
        ("H", -3.3740, -1.2460, 0.0), ("H", -1.2450, -2.4900, 0.0),
    ]
    anthracene = [
        ("C", 0.0000, 1.4050, 0.0), ("C", 0.0000, -1.4050, 0.0),
        ("C", 1.2200, 0.7150, 0.0), ("C", 1.2200, -0.7150, 0.0),
        ("C", -1.2200, 0.7150, 0.0), ("C", -1.2200, -0.7150, 0.0),
        ("C", 2.4650, 1.4050, 0.0), ("C", 2.4650, -1.4050, 0.0),
        ("C", 3.6550, 0.7100, 0.0), ("C", 3.6550, -0.7100, 0.0),
        ("C", -2.4650, 1.4050, 0.0), ("C", -2.4650, -1.4050, 0.0),
        ("C", -3.6550, 0.7100, 0.0), ("C", -3.6550, -0.7100, 0.0),
        ("H", 0.0000, 2.4900, 0.0), ("H", 0.0000, -2.4900, 0.0),
        ("H", 2.4650, 2.4900, 0.0), ("H", 2.4650, -2.4900, 0.0),
        ("H", 4.6000, 1.2500, 0.0), ("H", 4.6000, -1.2500, 0.0),
        ("H", -2.4650, 2.4900, 0.0), ("H", -2.4650, -2.4900, 0.0),
        ("H", -4.6000, 1.2500, 0.0), ("H", -4.6000, -1.2500, 0.0),
    ]

    systems = [
        ("C6H6 (114 BF, FC)", mshqc.molecule("C6H6"), "cc-pVDZ", True, True, True),
    ]
    if args.mode in {"heavy", "cpc_scaling"}:
        systems.append(("C10H8 (180 BF, FC)", mshqc.build_molecule(naphthalene, units="angstrom"), "cc-pVDZ", True, True, True))
    if args.mode == "cpc_scaling":
        systems.append(("C14H10 (246 BF, FC)", mshqc.build_molecule(anthracene, units="angstrom"), "cc-pVDZ", True, True, True))
        systems.append(("C6H6 (264 BF, FC)", mshqc.molecule("C6H6"), "cc-pVTZ", True, True, True))
        systems.append(("C10H8 (412 BF, FC)", mshqc.build_molecule(naphthalene, units="angstrom"), "cc-pVTZ", True, True, False))
    elif args.mode == "heavy":
        systems.append(("C6H6 (264 BF, FC)", mshqc.molecule("C6H6"), "cc-pVTZ", True, True, False))

    lines = [
        f"### mshqc v{mshqc.__version__} CUDA FP64 Benchmark ({gpu_name}, mode={args.mode})",
        f"- **FP64 GEMM 2048 Throughput:** `{fp64_tflops:.3f} TFLOPS` | **CUDA:** `{torch.version.cuda}` | **SHA:** `{args.git_sha[:10]}`",
        "",
        "| System | Basis | Nbf | t_RHF (CPU/GPU) | t_OMP2 (CPU/GPU) | t_LT-SOS (CPU/GPU) | t_OMP3 (CPU/GPU) | Speedup OMP3 | |dE_max| | Peak VRAM (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    records = []
    for label, mol, basis, fc, do_omp2, do_omp3 in systems:
        calc_c = mshqc.MSHQCCalculator(mol, basis_name=basis, eri_method="df", freeze_core=fc)
        nbf = int(calc_c.basis.n_basis_functions())
        t0 = time.perf_counter(); scf_c = calc_c.run_scf(); t_rhf_c = time.perf_counter() - t0
        o2_c, t_o2_c = None, None
        if do_omp2:
            t0 = time.perf_counter(); o2_c = calc_c.run_omp2(scf_c, grad_thresh=1e-5); t_o2_c = time.perf_counter() - t0
        t0 = time.perf_counter(); lt_c = calc_c.run_lt_sos_omp2(scf_c, grad_thresh=1e-5); t_lt_c = time.perf_counter() - t0
        o3_c, t_o3_c = None, None
        if do_omp3:
            t0 = time.perf_counter(); o3_c = calc_c.run_omp3(scf_c, grad_thresh=1e-5); t_o3_c = time.perf_counter() - t0

        torch.cuda.reset_peak_memory_stats()
        calc_g = mshqc.MSHQCCalculator(mol, basis_name=basis, eri_method="df", freeze_core=fc)
        with use_backend("cuda"):
            t0 = time.perf_counter(); scf_g = calc_g.run_scf(); torch.cuda.synchronize(); t_rhf_g = time.perf_counter() - t0
            o2_g, t_o2_g = None, None
            if do_omp2:
                t0 = time.perf_counter(); o2_g = calc_g.run_omp2(scf_g, grad_thresh=1e-5); torch.cuda.synchronize(); t_o2_g = time.perf_counter() - t0
            t0 = time.perf_counter(); lt_g = calc_g.run_lt_sos_omp2(scf_g, grad_thresh=1e-5); torch.cuda.synchronize(); t_lt_g = time.perf_counter() - t0
            o3_g, t_o3_g = None, None
            if do_omp3:
                t0 = time.perf_counter(); o3_g = calc_g.run_omp3(scf_g, grad_thresh=1e-5); torch.cuda.synchronize(); t_o3_g = time.perf_counter() - t0

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        peak_cpu_mb = peak_cpu_rss_mb()

        de_rhf = abs(scf_c.energy_total - scf_g.energy_total)
        de_lt = abs(lt_c.energy_total - lt_g.energy_total)
        de_o2 = abs(o2_c.energy_total - o2_g.energy_total) if (o2_c and o2_g) else 0.0
        de_o3 = abs(o3_c.energy_total - o3_g.energy_total) if (o3_c and o3_g) else 0.0
        de_max = max(de_rhf, de_o2, de_lt, de_o3)

        assert de_rhf < 1e-9, f"RHF CUDA parity mismatch: {de_rhf}"
        assert de_lt < 1e-9, f"LT-SOS-OMP2 CUDA parity mismatch: {de_lt}"
        if o2_c and o2_g:
            assert de_o2 < 1e-9, f"OMP2 CUDA parity mismatch: {de_o2}"
        if o3_c and o3_g:
            assert de_o3 < 1e-9, f"OMP3 CUDA parity mismatch: {de_o3}"

        o2_col = f"{t_o2_c:.2f}s / {t_o2_g:.2f}s" if (t_o2_c is not None and t_o2_g is not None) else "skip"
        if t_o3_c is not None and t_o3_g is not None:
            o3_col = f"{t_o3_c:.2f}s / {t_o3_g:.2f}s"
            sp_o3 = t_o3_c / max(t_o3_g, 1e-6)
            sp_col = f"{sp_o3:.2f}x"
        else:
            o3_col, sp_o3, sp_col = "skip", None, "-"

        lines.append(
            f"| {label} | {basis} | {nbf} | {t_rhf_c:.2f}s / {t_rhf_g:.2f}s | "
            f"{o2_col} | {t_lt_c:.2f}s / {t_lt_g:.2f}s | {o3_col} | {sp_col} | "
            f"{de_max:.2e} | {peak_vram_mb:.1f} |"
        )
        records.append({
            "label": label,
            "basis": basis,
            "nbf": nbf,
            "freeze_core": fc,
            "t_rhf_cpu": t_rhf_c, "t_rhf_gpu": t_rhf_g, "de_rhf": de_rhf,
            "t_omp2_cpu": t_o2_c, "t_omp2_gpu": t_o2_g, "de_omp2": de_o2,
            "t_lt_sos_cpu": t_lt_c, "t_lt_sos_gpu": t_lt_g, "de_lt_sos": de_lt,
            "t_omp3_cpu": t_o3_c, "t_omp3_gpu": t_o3_g, "de_omp3": de_o3,
            "speedup_omp3": sp_o3,
            "speedup_lt_sos": t_lt_c / max(t_lt_g, 1e-6),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "peak_cpu_rss_mb": round(peak_cpu_mb, 2),
        })
        cuda_backend.clear_cache()

    report_md = "\n".join(lines) + "\n"
    payload = {
        "git_sha": args.git_sha,
        "mode": args.mode,
        "provider": "kaggle",
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_total_gb, 2),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "fp64_gemm_2048_tflops": round(fp64_tflops, 3),
        "platform": platform.platform(),
        "records": records,
        "markdown": report_md,
    }
    (out_dir / "cuda_gpu_benchmark_summary.md").write_text(report_md, encoding="utf-8")
    (out_dir / "cuda_gpu_benchmark_raw.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + report_md, flush=True)


if __name__ == "__main__":
    main()
