// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_HYBRID_PHASE1_BIND_KERNELS]

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "mshqc/basis.h"
#include "mshqc/core/fock_builder.h"
#include "mshqc/core/molecule.h"
#include "mshqc/integrals/df_eri.h"
#include "mshqc/integrals/packed_ao_utils.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/kernels/primitives.h"

namespace nb = nanobind;

namespace mshqc {

void bind_kernels_module(nb::module_& parent) {
    auto k = parent.def_submodule(
        "kernels",
        "Coarse-grained C++20 HPC primitives for hybrid Python/C++ method development");

    nb::class_<integrals::DensityFittingERI>(k, "DensityFittingERI")
        .def("__init__",
            [](integrals::DensityFittingERI* self,
               const BasisSet& primary_basis,
               const BasisSet& aux_basis,
               IntegralEngine* integrals,
               double cutoff) {
                auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
                new (self) integrals::DensityFittingERI(primary_basis, aux_basis, sp_int, cutoff);
            },
            nb::arg("primary_basis"), nb::arg("aux_basis"), nb::arg("integrals"),
            nb::arg("cutoff") = 1e-10,
            nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
        .def("__init__",
            [](integrals::DensityFittingERI* self,
               const Molecule& mol,
               const BasisSet& primary_basis,
               const BasisSet& aux_basis,
               double cutoff) {
                auto sp_int = std::make_shared<IntegralEngine>(mol, primary_basis);
                new (self) integrals::DensityFittingERI(primary_basis, aux_basis, sp_int, cutoff);
            },
            nb::arg("mol"), nb::arg("primary_basis"), nb::arg("aux_basis"),
            nb::arg("cutoff") = 1e-10,
            nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
        .def("compute",
            [](integrals::DensityFittingERI& self, bool save_hdf5) {
                nb::gil_scoped_release release;
                self.compute(save_hdf5);
            },
            nb::arg("save_hdf5") = false)
        .def("get_B_mat", &integrals::DensityFittingERI::get_B_mat,
             nb::rv_policy::reference_internal)
        .def("set_hdf5_path", &integrals::DensityFittingERI::set_hdf5_path, nb::arg("path"))
        .def("get_hdf5_path", &integrals::DensityFittingERI::get_hdf5_path);

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

    k.def("is_packed_ao", &integrals::is_packed_ao_matrix,
          nb::arg("L_mat"), nb::arg("nbf"),
          "Check whether 3-index AO factor uses packed lower-triangular storage");

    k.def("build_coulomb_3c", &kernels::build_coulomb_3c,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("P_tot"), nb::arg("nbf"),
          "Compute Coulomb matrix J[P_tot] from 3-index AO factor (packed or full)");

    k.def("build_exchange_3c_occ", &kernels::build_exchange_3c_occ,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("nbf"),
          "Compute Exchange matrix K[C_occ * C_occ^T] in O(N_aux * N_occ * N_bf^2)");

    k.def("build_exchange_3c_sym", &kernels::build_exchange_3c_sym,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("D_sym"), nb::arg("nbf"),
          "Compute Exchange matrix K[D_sym] for a general symmetric density matrix");

    k.def("build_jk_3c_rhf", &kernels::build_jk_3c_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C_occ"), nb::arg("nbf"),
          "Compute (J[2*C*C^T], K[C*C^T]) for closed-shell RHF (F = H_core + J - K)");

    k.def("transform_3c_mo", &kernels::transform_3c_mo,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("L_mat"), nb::arg("C1"), nb::arg("C2"), nb::arg("nbf"),
          "Transform 3-index AO factor to MO block B(d1*d2, n_aux)");

    k.def("semicanonicalize_subspaces", &kernels::semicanonicalize_subspaces,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("C"), nb::arg("F_ao"), nb::arg("n_occ"),
          "Diagonalize Occ-Occ and Vir-Vir Fock blocks and return (C_semi, eps, F_vo)");

    k.def("eval_rmp2_df", &kernels::eval_rmp2_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("B_ov"), nb::arg("eps_occ"), nb::arg("eps_vir"),
          "Evaluate closed-shell DF-MP2 returning (e_os, e_ss, T2[nv*nv, no*no])");

    k.def("eval_pp_ladder_rhf_df", &kernels::eval_pp_ladder_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_vv"), nb::arg("n_occ"), nb::arg("n_vir"),
          "O(V^4)-free occupied-pair sliced (i<=j) particle-particle ladder contraction");

    k.def("eval_omp2_1rdm_rhf", &kernels::eval_omp2_1rdm_rhf,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("n_occ"), nb::arg("n_vir"),
          nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0,
          "Evaluate spin-alpha 1-RDM blocks (gamma_oo, gamma_vv) supporting SCS scaling");

    k.def("eval_omp2_zvec_rhf_df", &kernels::eval_omp2_zvec_rhf_df,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("T2"), nb::arg("B_oo"), nb::arg("B_ov"), nb::arg("B_vv"),
          nb::arg("n_occ"), nb::arg("n_vir"),
          nb::arg("c_os") = 1.0, nb::arg("c_ss") = 1.0,
          "Evaluate OMP2 Z-vector Z_vo via 3-index DF factors supporting SCS scaling");
}

} // namespace mshqc
