// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "mshqc/basis.h"
#include "mshqc/core/molecule.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/kernels/mp3_kernels.h"
#include "mshqc/kernels/primitives.h"
#include "mshqc/kernels/uhf_kernels.h"

namespace nb = nanobind;

namespace mshqc {

void bind_kernels_module(nb::module_& parent) {
    auto k = parent.def_submodule(
        "kernels",
        "Coarse-grained C++20 HPC primitives (RHF + UHF, MP2/MP3/OMP2/OMP3, LT-SOS & Analytic DF Gradients)");

    nb::class_<kernels::SafeDFEngine>(k, "DensityFittingERI")
        .def(nb::init<const Molecule&, const BasisSet&, const BasisSet&, double>(),
             nb::arg("mol"), nb::arg("primary_basis"), nb::arg("aux_basis"),
             nb::arg("cutoff") = 1e-10)
        .def("compute",
             [](kernels::SafeDFEngine& self, bool save_hdf5) {
                 nb::gil_scoped_release release;
                 self.compute(save_hdf5);
             },
             nb::arg("save_hdf5") = false)
        .def("get_B_mat", &kernels::SafeDFEngine::get_B_mat,
             nb::rv_policy::reference_internal)
        .def("set_verbose", &kernels::SafeDFEngine::set_verbose, nb::arg("verbose"))
        .def("set_hdf5_path", &kernels::SafeDFEngine::set_hdf5_path, nb::arg("path"))
        .def("get_hdf5_path", &kernels::SafeDFEngine::get_hdf5_path);

    k.def("compute_df_ao_factor", &kernels::compute_df_ao_factor,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("mol"), nb::arg("primary_basis"), nb::arg("aux_basis"),
          nb::arg("cutoff") = 1e-10, nb::arg("save_hdf5") = false, nb::arg("verbose") = false);

    // --- Shared AO/MO & RHF Primitives ---
    k.def("is_packed_ao", &integrals::is_packed_ao_matrix, nb::arg("L_mat"), nb::arg("nbf"));
    k.def("build_coulomb_3c", &kernels::build_coulomb_3c,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("L_mat"), nb::arg("P_tot"), nb::arg("nbf"));
    k.def("build_exchange_3c_occ", &kernels::build_exchange_3c_occ,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("nbf"));
    k.def("build_exchange_3c_diag", &kernels::build_exchange_3c_diag,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("L_mat"), nb::arg("d_diag"), nb::arg("nbf"));
    k.def("build_exchange_3c_sym", &kernels::build_exchange_3c_sym,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("L_mat"), nb::arg("D_sym"), nb::arg("nbf"));
    k.def("build_jk_3c_rhf", &kernels::build_jk_3c_rhf,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("nbf"));
    k.def("transform_3c_mo", &kernels::transform_3c_mo,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("L_mat"), nb::arg("C1"), nb::arg("C2"), nb::arg("nbf"));
    k.def("transform_3c_mo_blocks", &kernels::transform_3c_mo_blocks,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("C_vir"), nb::arg("nbf"));
    k.def("transform_3c_mo_blocks_uhf", &kernels::transform_3c_mo_blocks_uhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("Ca_occ"), nb::arg("Ca_vir"),
          nb::arg("Cb_occ"), nb::arg("Cb_vir"), nb::arg("nbf"));
    k.def("semicanonicalize_subspaces", &kernels::semicanonicalize_subspaces,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("C"), nb::arg("F_ao"), nb::arg("n_occ"));

    k.def("eval_rmp2_df", &kernels::eval_rmp2_df,
          nb::call_guard<nb::gil_scoped_release>(), nb::arg("B_ov"), nb::arg("eps_occ"), nb::arg("eps_vir"));
    k.def("eval_omp2_1rdm_rhf", &kernels::eval_omp2_1rdm_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("n_occ"), nb::arg("n_vir"), nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0);
    k.def("eval_omp2_zvec_rhf_df", &kernels::eval_omp2_zvec_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("n_occ"), nb::arg("n_vir"), nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0);
    k.def("eval_pp_ladder_rhf_df", &kernels::eval_pp_ladder_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_vv"), nb::arg("n_occ"), nb::arg("n_vir"));
    k.def("eval_rmp3_df", &kernels::eval_rmp3_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("eps_occ"), nb::arg("eps_vir"));
    k.def("eval_omp3_1rdm_rhf", &kernels::eval_omp3_1rdm_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("L2"), nb::arg("n_occ"), nb::arg("n_vir"));
    k.def("eval_omp3_zvec_rhf_df", &kernels::eval_omp3_zvec_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("L2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("n_occ"), nb::arg("n_vir"));
    k.def("apply_mo_df_hessian_rhf", &kernels::apply_mo_df_hessian_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("kappa_vo"), nb::arg("eps_occ"), nb::arg("eps_vir"),
          nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"));

    // --- Open-Shell (UHF) DF Primitives ---
    k.def("build_jk_3c_uhf", &kernels::build_jk_3c_uhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C_occ_a"), nb::arg("C_occ_b"), nb::arg("nbf"));
    k.def("eval_ump2_df", &kernels::eval_ump2_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("B_ov_a"), nb::arg("B_ov_b"),
          nb::arg("eps_oa"), nb::arg("eps_va"), nb::arg("eps_ob"), nb::arg("eps_vb"));
    k.def("eval_omp2_1rdm_uhf", &kernels::eval_omp2_1rdm_uhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2_aa"), nb::arg("T2_bb"), nb::arg("T2_ab"),
          nb::arg("na"), nb::arg("va"), nb::arg("nb"), nb::arg("vb"),
          nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0);
    k.def("eval_omp2_zvec_uhf_df", &kernels::eval_omp2_zvec_uhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2_aa"), nb::arg("T2_bb"), nb::arg("T2_ab"),
          nb::arg("B_oo_a"), nb::arg("B_ov_a"), nb::arg("B_vv_a"),
          nb::arg("B_oo_b"), nb::arg("B_ov_b"), nb::arg("B_vv_b"),
          nb::arg("na"), nb::arg("va"), nb::arg("nb"), nb::arg("vb"),
          nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0);
    k.def("eval_ump3_df", &kernels::eval_ump3_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2_aa"), nb::arg("T2_bb"), nb::arg("T2_ab"),
          nb::arg("B_oo_a"), nb::arg("B_ov_a"), nb::arg("B_vv_a"),
          nb::arg("B_oo_b"), nb::arg("B_ov_b"), nb::arg("B_vv_b"),
          nb::arg("eps_oa"), nb::arg("eps_va"), nb::arg("eps_ob"), nb::arg("eps_vb"));
    k.def("eval_omp3_1rdm_uhf", &kernels::eval_omp3_1rdm_uhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2_aa"), nb::arg("T2_bb"), nb::arg("T2_ab"),
          nb::arg("L2_aa"), nb::arg("L2_bb"), nb::arg("L2_ab"),
          nb::arg("na"), nb::arg("va"), nb::arg("nb"), nb::arg("vb"));
    k.def("eval_omp3_zvec_uhf_df", &kernels::eval_omp3_zvec_uhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2_aa"), nb::arg("T2_bb"), nb::arg("T2_ab"),
          nb::arg("L2_aa"), nb::arg("L2_bb"), nb::arg("L2_ab"),
          nb::arg("B_oo_a"), nb::arg("B_ov_a"), nb::arg("B_vv_a"),
          nb::arg("B_oo_b"), nb::arg("B_ov_b"), nb::arg("B_vv_b"),
          nb::arg("na"), nb::arg("va"), nb::arg("nb"), nb::arg("vb"));
    k.def("apply_mo_df_hessian_uhf", &kernels::apply_mo_df_hessian_uhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("kappa_vo_a"), nb::arg("kappa_vo_b"),
          nb::arg("eps_oa"), nb::arg("eps_va"), nb::arg("eps_ob"), nb::arg("eps_vb"),
          nb::arg("B_oo_a"), nb::arg("B_ov_a"), nb::arg("B_vv_a"),
          nb::arg("B_oo_b"), nb::arg("B_ov_b"), nb::arg("B_vv_b"));

    // --- Analytic DF Nuclear Gradient & O(N^4) LT-SOS Primitives ---
    k.def("contract_ao_sym", &kernels::contract_ao_sym,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("M_sym"), nb::arg("nbf"));
    k.def("compute_gamma2c", &kernels::compute_gamma2c,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("Gamma_full"), nb::arg("L_mat"), nb::arg("nbf"));
    k.def("eval_df_nuclear_gradient", &kernels::eval_df_nuclear_gradient,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("mol"), nb::arg("primary_basis"), nb::arg("aux_basis"),
          nb::arg("P_eff_ao"), nb::arg("W_eff_ao"),
          nb::arg("Gamma_3c"), nb::arg("gamma_2c"),
          nb::arg("cutoff") = 1e-12);
    k.def("eval_lt_sos_intermediates_rhf_df", &kernels::eval_lt_sos_intermediates_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("B_ov"), nb::arg("d_ov"), nb::arg("weights"), nb::arg("nodes"),
          nb::arg("c_os") = 1.3);
    k.def("eval_lt_sos_intermediates_uhf_df", &kernels::eval_lt_sos_intermediates_uhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("B_ov_a"), nb::arg("B_ov_b"), nb::arg("da"), nb::arg("db"),
          nb::arg("weights"), nb::arg("nodes"), nb::arg("c_os") = 1.3);
}

} // namespace mshqc
