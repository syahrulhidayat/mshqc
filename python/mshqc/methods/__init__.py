# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Lapisan 2: Orkestrasi metode Python modular di atas C++20 `mshqc.kernels`."""

from .tensors import t2_to_4d, t2_from_4d, b3c_to_3d, b3c_from_3d
from .tr_soscf import solve_tr_soscf_rhf, solve_tr_soscf_uhf
from .scf_hybrid import HybridSCFResult, PulayDIIS, run_df_rhf, run_df_uhf
from .omp_hybrid import (
    HybridOMPResult,
    eval_omp_rhf_point,
    eval_omp_uhf_point,
    run_df_omp_rhf,
    run_df_omp_uhf,
)
from .interop import run_from_arrays, run_from_pyscf
from .custom import optimize_orbitals_rhf, optimize_orbitals_uhf

__all__ = [
    "t2_to_4d", "t2_from_4d", "b3c_to_3d", "b3c_from_3d",
    "solve_tr_soscf_rhf", "solve_tr_soscf_uhf",
    "HybridSCFResult", "PulayDIIS", "run_df_rhf", "run_df_uhf",
    "HybridOMPResult",
    "eval_omp_rhf_point", "eval_omp_uhf_point",
    "run_df_omp_rhf", "run_df_omp_uhf",
    "run_from_arrays", "run_from_pyscf",
    "optimize_orbitals_rhf", "optimize_orbitals_uhf",
]
