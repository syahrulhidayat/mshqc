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
    run_exact_mp2_rhf,
    run_exact_mp2_uhf,
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
    eval_df_kappa_omp2_gradient_rhf,
    eval_df_kappa_omp2_gradient_uhf,
    compute_df_scf_gradient,
    compute_df_gradient,
)
from .geom_hybrid import (
    ThinCoreGeomResult,
    ThinCoreVibResult,
    optimize_geometry_thin_core,
    ThinCorePESPoint,
    compute_frequencies_thin_core,
    scan_bond_pes_thin_core,
)
from .kappa_omp2 import run_df_kappa_omp2_rhf, run_df_kappa_omp2_uhf
from .lt_sos_omp2 import (
    AdaptiveLaplaceCache,
    build_laplace_quadrature,
    run_df_lt_sos_omp2_rhf,
    run_df_lt_sos_omp2_uhf,
)
from .checkpoint import save_orbital_checkpoint, load_orbital_checkpoint
from .properties import (
    NaturalOrbitalResult,
    PopulationResult,
    build_1rdm_ao,
    scf_density_ao,
    correlated_density_ao,
    compute_natural_orbitals,
    mulliken_and_lowdin_charges,
    write_molden,
)
from .excited import ExcitedStateResult, run_df_cis_rhf, run_df_tdhf_rhf
from .oo_dh_dft import OODoubleHybridResult, run_oo_double_hybrid_rhf
from .interop import run_from_arrays, run_from_pyscf
# [INJECTED_V2_POLISH_SYMMETRY_EXPORT]
from .symmetry import (
    AbelianSymmetryInfo,
    AbelianSALCBlocks,
    build_salc_orthogonalizer,
    align_principal_axes,
    detect_abelian_symmetry,
    build_ao_symmetry_operators,
    symmetrize_ao_matrix,
    label_mo_irreps,
)

__all__ = [
    "t2_to_4d", "t2_from_4d", "b3c_to_3d", "b3c_from_3d",
    "solve_tr_soscf_rhf", "solve_tr_soscf_uhf",
    "HybridSCFResult", "PulayDIIS", "build_sad_guess", "orthonormalize_mo_guess",
    "run_df_rhf", "run_df_uhf", "run_df_rohf",
    "HybridMP2Result", "HybridMP3Result",
    "run_df_mp2_rhf", "run_df_mp2_uhf", "run_df_romp2",
    "run_df_mp3_rhf", "run_df_mp3_uhf", "run_df_romp3", "run_exact_mp2_rhf", "run_exact_mp2_uhf",
    "HybridOMPResult", "semicanonicalize_subspaces_frozen",
    "eval_omp_rhf_point", "eval_omp_uhf_point",
    "run_df_omp_rhf", "run_df_omp_uhf",
    "optimize_orbitals_rhf", "optimize_orbitals_uhf",
    "eval_df_scf_gradient_rhf", "eval_df_scf_gradient_uhf", "eval_df_scf_gradient_rohf",
    "eval_df_omp2_gradient_rhf", "eval_df_omp2_gradient_uhf",
    "eval_df_kappa_omp2_gradient_rhf", "eval_df_kappa_omp2_gradient_uhf",
    "compute_df_scf_gradient", "compute_df_gradient",
    "ThinCoreGeomResult", "ThinCoreVibResult",
    "ThinCorePESPoint", "optimize_geometry_thin_core", "compute_frequencies_thin_core", "scan_bond_pes_thin_core",
    "run_df_kappa_omp2_rhf", "run_df_kappa_omp2_uhf",
    "AdaptiveLaplaceCache", "build_laplace_quadrature", "run_df_lt_sos_omp2_rhf", "run_df_lt_sos_omp2_uhf",
    "save_orbital_checkpoint", "load_orbital_checkpoint",
    "NaturalOrbitalResult", "PopulationResult", "build_1rdm_ao", "scf_density_ao", "correlated_density_ao",
    "compute_natural_orbitals", "mulliken_and_lowdin_charges", "write_molden",
    "ExcitedStateResult", "run_df_cis_rhf", "run_df_tdhf_rhf",
    "OODoubleHybridResult", "run_oo_double_hybrid_rhf",
    "run_from_arrays", "run_from_pyscf",
    "AbelianSymmetryInfo", "AbelianSALCBlocks", "build_salc_orthogonalizer",
    "align_principal_axes", "detect_abelian_symmetry",
    "build_ao_symmetry_operators", "symmetrize_ao_matrix", "label_mo_irreps",
]
