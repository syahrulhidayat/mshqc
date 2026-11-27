#!/usr/bin/env python3
"""
Runner Modal serverless GPU (FP64) untuk mshqc cabang hybrid-v2.

Selaras dengan arsitektur riil mshqc:
  - Memakai PyTorch CUDA (`torch_backend.py`) untuk akselerasi GPU FP64.
  - Mengompilasi dan menginstal `mshqc` langsung dari source tree `/workspace`
    (BUKAN dari PyPI) agar benchmark selalu memverifikasi commit terbaru.

Penggunaan lokal:
    modal profile activate syahrulhidayat
    modal run scripts/modal_gpu_bench.py --out artifacts/modal_gpu_bench.json
    modal run scripts/modal_gpu_bench.py --smoke-only --out artifacts/modal_gpu_smoke.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import modal

GPU_SKU = os.environ.get("MSHQC_MODAL_GPU", "A100")

app = modal.App("mshqc-cpc-gpu-bench")

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ninja-build",
        "git",
        "pkg-config",
        "gfortran",
        "libopenblas-dev",
        "liblapack-dev",
        "libeigen3-dev",
    )
    .pip_install(
        "numpy",
        "scipy",
        "pytest",
        "psutil",
        "pybind11",
        "setuptools",
        "wheel",
        "scikit-build-core",
    )
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[
            ".git",
            "build",
            "build-*",
            "dist",
            "__pycache__",
            "*.pyc",
            ".venv",
            "artifacts",
            "experimental",
        ],
    )
)


def _run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, combined


@app.function(
    image=cuda_image,
    gpu=GPU_SKU,
    timeout=1800,
)
def run_gpu_benchmark(
    git_sha: str = "local",
    bench_script: str = "benchmarks/run_heavy_large_v2.py",
    smoke_only: bool = False,
) -> dict:
    """Build mshqc dari /workspace, uji FP64 Torch CUDA, lalu jalankan benchmark."""
    os.chdir("/workspace")
    t0 = time.perf_counter()

    gpu_info = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()

    # 1. Verifikasi PyTorch CUDA FP64 hardware throughput
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia di dalam kontainer Modal.")

    n_test = 2048
    a = torch.ones((n_test, n_test), dtype=torch.float64, device="cuda")
    b = torch.ones((n_test, n_test), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    tg0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    tg1 = time.perf_counter()
    fp64_tflops = (2.0 * (n_test**3)) / ((tg1 - tg0) * 1e12)
    fp64_checksum = float(c[0, 0].item())

    # 2. Build & install mshqc langsung dari source tree /workspace (non-PyPI)
    tb0 = time.perf_counter()
    build_rc, build_log = _run_cmd(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."]
    )
    build_seconds = round(time.perf_counter() - tb0, 3)

    mshqc_origin = None
    import_error = None
    if build_rc == 0:
        try:
            import mshqc

            mshqc_origin = str(Path(mshqc.__file__).resolve())
            if "/workspace" not in mshqc_origin:
                raise RuntimeError(
                    f"mshqc terimpor dari luar /workspace ({mshqc_origin}); pastikan bukan paket PyPI."
                )
        except Exception as exc:
            import_error = repr(exc)

    # 3. Jalankan benchmark GPU target (atau lewati jika --smoke-only / build gagal)
    bench_path = Path(bench_script)
    bench_status = "skipped_smoke_only"
    bench_log_tail = ""

    if build_rc != 0 or import_error is not None:
        bench_status = "skipped_build_or_import_failed"
    elif not smoke_only:
        if not bench_path.exists():
            bench_status = f"skipped_not_found:{bench_script}"
        else:
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "4")
            rc, out_log = _run_cmd([sys.executable, str(bench_path)], env=env)
            bench_status = "ok" if rc == 0 else f"failed_rc_{rc}"
            bench_log_tail = out_log[-4000:]

    return {
        "git_sha": git_sha,
        "gpu_sku_requested": GPU_SKU,
        "gpu_hardware": gpu_info,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "host_platform": platform.platform(),
        "fp64_gemm_2048_tflops": round(fp64_tflops, 3),
        "fp64_gemm_checksum": fp64_checksum,
        "source_build_rc": build_rc,
        "source_build_seconds": build_seconds,
        "source_build_log_tail": build_log[-2500:] if build_rc != 0 else "",
        "mshqc_origin": mshqc_origin,
        "mshqc_import_error": import_error,
        "benchmark_script": bench_script,
        "benchmark_status": bench_status,
        "benchmark_log_tail": bench_log_tail,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


@app.local_entrypoint()
def main(
    out: str = "artifacts/modal_gpu_bench.json",
    bench_script: str = "benchmarks/run_heavy_large_v2.py",
    smoke_only: bool = False,
) -> None:
    git_sha = os.environ.get("MSHQC_GIT_SHA", "local")
    result = run_gpu_benchmark.remote(
        git_sha=git_sha,
        bench_script=bench_script,
        smoke_only=smoke_only,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\n[OK] Benchmark artifact saved to: {out_path}")
    if result["source_build_rc"] != 0 or result["mshqc_import_error"] is not None:
        raise SystemExit(1)
