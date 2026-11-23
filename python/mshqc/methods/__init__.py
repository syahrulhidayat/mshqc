# Copyright 2026 Muhamad Syahrul Hidayat
# Licensed under the Apache License, Version 2.0
"""Lapisan 2: Orkestrasi metode Python modular di atas C++20 `mshqc.kernels`."""

from .tensors import t2_to_4d, t2_from_4d, b3c_to_3d, b3c_from_3d
from .tr_soscf import solve_tr_soscf_rhf, solve_tr_soscf_uhf
from .scf_hybrid import (
    HybridSCFResult,
    PulayDIIS,
    build_sad_guess,
    orthonormalize_mo_guess,
    run_df_rhf,
    run_df_uhf,
    run_df_rohf,
)
from .mp_hybrid import (
    HybridMP2Result,
    HybridMP3Result,
    run_df_mp2_rhf,
    run_df_mp2_uhf,
    run_df_romp2,
    run_df_mp3_rhf,
    run_df_mp3_uhf,
    run_df_romp3,
)
from .omp_hybrid import (
    HybridOMPResult,
    semicanonicalize_subspaces_frozen,
    eval_omp_rhf_point,
    eval_omp_uhf_point,
    run_df_omp_rhf,
    run_df_omp_uhf,
    optimize_orbitals_rhf,
    optimize_orbitals_uhf,
)
from .grad_hybrid import (
    eval_df_scf_gradient_rhf,
    eval_df_scf_gradient_uhf,
    eval_df_scf_gradient_rohf,
    eval_df_omp2_gradient_rhf,
    eval_df_omp2_gradient_uhf,
    compute_df_scf_gradient,
    compute_df_gradient,
)
from .geom_hybrid import (
    ThinCoreGeomResult,
    ThinCoreVibResult,
    optimize_geometry_thin_core,
    compute_frequencies_thin_core,
)
from .kappa_omp2 import run_df_kappa_omp2_rhf, run_df_kappa_omp2_uhf
from .checkpoint import save_orbital_checkpoint, load_orbital_checkpoint
from .interop import run_from_arrays, run_from_pyscf

__all__ = [
    "t2_to_4d", "t2_from_4d", "b3c_to_3d", "b3c_from_3d",
    "solve_tr_soscf_rhf", "solve_tr_soscf_uhf",
    "HybridSCFResult", "PulayDIIS", "build_sad_guess", "orthonormalize_mo_guess",
    "run_df_rhf", "run_df_uhf", "run_df_rohf",
    "HybridMP2Result", "HybridMP3Result",
    "run_df_mp2_rhf", "run_df_mp2_uhf", "run_df_romp2",
    "run_df_mp3_rhf", "run_df_mp3_uhf", "run_df_romp3",
    "HybridOMPResult", "semicanonicalize_subspaces_frozen",
    "eval_omp_rhf_point", "eval_omp_uhf_point",
    "run_df_omp_rhf", "run_df_omp_uhf",
    "optimize_orbitals_rhf", "optimize_orbitals_uhf",
    "eval_df_scf_gradient_rhf", "eval_df_scf_gradient_uhf", "eval_df_scf_gradient_rohf",
    "eval_df_omp2_gradient_rhf", "eval_df_omp2_gradient_uhf",
    "compute_df_scf_gradient", "compute_df_gradient",
    "ThinCoreGeomResult", "ThinCoreVibResult",
    "optimize_geometry_thin_core", "compute_frequencies_thin_core",
    "run_df_kappa_omp2_rhf", "run_df_kappa_omp2_uhf",
    "save_orbital_checkpoint", "load_orbital_checkpoint",
    "run_from_arrays", "run_from_pyscf",
]
