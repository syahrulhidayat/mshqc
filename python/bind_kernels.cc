// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE2_BIND_KERNELS]

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "mshqc/basis.h"
#include "mshqc/core/fock_builder.h"
#include "mshqc/core/molecule.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/kernels/mp3_kernels.h"
#include "mshqc/kernels/primitives.h"

namespace nb = nanobind;

namespace mshqc {

void bind_kernels_module(nb::module_& parent) {
    auto k = parent.def_submodule(
        "kernels",
        "Coarse-grained C++20 HPC primitives (Phase 1 + Phase 2) for hybrid Python/C++ methods");

    // Safe RAII DensityFittingERI wrapper (auto-builds combined primary+aux basis)
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
        .def("set_hdf5_path", &kernels::SafeDFEngine::set_hdf5_path, nb::arg("path"))
        .def("get_hdf5_path", &kernels::SafeDFEngine::get_hdf5_path);

    k.def("compute_df_ao_factor", &kernels::compute_df_ao_factor,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("mol"), nb::arg("primary_basis"), nb::arg("aux_basis"),
          nb::arg("cutoff") = 1e-10, nb::arg("save_hdf5") = false,
          "Safely compute packed 3-index AO Density Fitting factor B_mat");

    nb::class_<FockBuilder>(k, "FockBuilder")
        .def("__init__",
             [](FockBuilder* self,
                IntegralEngine* integrals,
                const BasisSet& basis,
                const Eigen::MatrixXd& H_core,
                const Eigen::MatrixXd& schwarz) {
                 auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
                 new (self) FockBuilder(sp_int, basis, H_core, schwarz);
             },
             nb::arg("integrals"), nb::arg("basis"), nb::arg("H_core"), nb::arg("schwarz"),
             nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def("reset", &FockBuilder::reset)
        .def("compute",
             [](FockBuilder& self, const Eigen::MatrixXd& P_alpha, const Eigen::MatrixXd& P_beta) {
                 Eigen::MatrixXd F_alpha, F_beta;
                 {
                     nb::gil_scoped_release release;
                     self.compute(P_alpha, P_beta, F_alpha, F_beta);
                 }
                 return std::make_pair(std::move(F_alpha), std::move(F_beta));
             },
             nb::arg("P_alpha"), nb::arg("P_beta"));

    // --- Phase 1: AO/MO 3-index & OMP2 primitives ---
    k.def("is_packed_ao", &integrals::is_packed_ao_matrix,
          nb::arg("L_mat"), nb::arg("nbf"));

    k.def("build_coulomb_3c", &kernels::build_coulomb_3c,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("P_tot"), nb::arg("nbf"));

    k.def("build_exchange_3c_occ", &kernels::build_exchange_3c_occ,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("nbf"));

    k.def("build_exchange_3c_sym", &kernels::build_exchange_3c_sym,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("D_sym"), nb::arg("nbf"));

    k.def("build_jk_3c_rhf", &kernels::build_jk_3c_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("nbf"));

    k.def("transform_3c_mo", &kernels::transform_3c_mo,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C1"), nb::arg("C2"), nb::arg("nbf"));

    k.def("semicanonicalize_subspaces", &kernels::semicanonicalize_subspaces,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("C"), nb::arg("F_ao"), nb::arg("n_occ"));

    k.def("eval_rmp2_df", &kernels::eval_rmp2_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("B_ov"), nb::arg("eps_occ"), nb::arg("eps_vir"));

    k.def("eval_omp2_1rdm_rhf", &kernels::eval_omp2_1rdm_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("n_occ"), nb::arg("n_vir"),
          nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0);

    k.def("eval_omp2_zvec_rhf_df", &kernels::eval_omp2_zvec_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("n_occ"), nb::arg("n_vir"),
          nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0);

    // --- Phase 2: O(V^4)-free MP3/OMP3 & MO-DF TR-SOSCF Hessian-vector kernels ---
    k.def("eval_pp_ladder_rhf_df", &kernels::eval_pp_ladder_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_vv"), nb::arg("n_occ"), nb::arg("n_vir"));

    k.def("eval_rmp3_df", &kernels::eval_rmp3_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("eps_occ"), nb::arg("eps_vir"),
          "Evaluate DF-RMP3 returning (e_mp3, L2) with O(V^4)-free adaptive ladder");

    k.def("eval_omp3_1rdm_rhf", &kernels::eval_omp3_1rdm_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("L2"), nb::arg("n_occ"), nb::arg("n_vir"),
          "Evaluate streaming zero-copy RHF OMP3 1-RDM (gamma_oo, gamma_vv)");

    k.def("eval_omp3_zvec_rhf_df", &kernels::eval_omp3_zvec_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("L2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("n_occ"), nb::arg("n_vir"),
          "Evaluate O(V^4)-free RHF OMP3 Z-vector via chunked Y_ooov intermediate");

    k.def("apply_mo_df_hessian_rhf", &kernels::apply_mo_df_hessian_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("kappa_vo"), nb::arg("eps_occ"), nb::arg("eps_vir"),
          nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          "Apply MO-space DF approximate orbital Hessian to trial step kappa_vo");
}

} // namespace mshqc
