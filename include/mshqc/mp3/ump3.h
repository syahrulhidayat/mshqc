 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //     http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 // ==============================================================================

#ifndef MSHQC_UMP3_H
#define MSHQC_UMP3_H

#include "mshqc/scf/scf.h"
#include "mshqc/mp2/ump2.h"
#include "mshqc/mp2/mp2.h"
#include "mshqc/mp3/ump3_memory.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/symmetry/point_group.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

struct UMP3Result {
    double e_uhf = 0.0;
    double e_mp2 = 0.0;
    double e_mp3 = 0.0;

    double e3_aa = 0.0;
    double e3_bb = 0.0;
    double e3_ab = 0.0;

    double e_corr_total = 0.0;
    double e_total = 0.0;
    int n_occ_alpha;
    int n_occ_beta;
    int n_virt_alpha;
    int n_virt_beta;

    Eigen::Tensor<double, 2> t1_a_2;
    Eigen::Tensor<double, 2> t1_b_2;

    Eigen::Tensor<double, 4> t2_aa_1;
    Eigen::Tensor<double, 4> t2_bb_1;
    Eigen::Tensor<double, 4> t2_ab_1;

    Eigen::Tensor<double, 4> t2_aa_2;
    Eigen::Tensor<double, 4> t2_bb_2;
    Eigen::Tensor<double, 4> t2_ab_2;
};

class UMP3 {
public:

    UMP3(const SCFResult& uhf,
         const UMP2Result& ump2,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         std::shared_ptr<PointGroup> pg = nullptr);

    UMP3Result compute();

private:
    UMP3Workspace workspace_;

    void transform_integrals();
    void compute_mp2_amplitudes();
    void compute_mp3_energy(UMP3Result& r);

    double calc_pp_aa();
    double calc_pp_bb();
    double calc_hh_aa();
    double calc_hh_bb();
    double calc_ph_aa();
    double calc_ph_bb();

    double calc_pp_ab();
    double calc_hh_ab();
    double calc_ph_ab();

    double tensor_norm(const Eigen::Tensor<double, 4>& t);
    Eigen::Tensor<double, 4> manual_transform_chem(
        const Eigen::Tensor<double, 4>&, const Eigen::MatrixXd&, int,
        const Eigen::MatrixXd&, int, const Eigen::MatrixXd&, int,
        const Eigen::MatrixXd&, int, int);

    SCFResult uhf_;
    UMP2Result ump2_;
    BasisSet basis_;
    std::shared_ptr<IntegralEngine> integrals_;

    std::shared_ptr<PointGroup> pg_;
    std::vector<int> irreps_occ_a_;
    std::vector<int> irreps_vir_a_;
    std::vector<int> irreps_occ_b_;
    std::vector<int> irreps_vir_b_;

    int nbf_;
    int nbf;
    int nocc_a_, nocc_b_;
    int nvir_a_, nvir_b_;

    Eigen::Tensor<double, 4> g_oooo_aa_, g_oooo_bb_, g_oooo_ab_;
    Eigen::Tensor<double, 4> g_vvvv_aa_, g_vvvv_bb_, g_vvvv_ab_;
    Eigen::Tensor<double, 4> g_ovov_aa_, g_ovov_bb_, g_ovov_ab_;
    Eigen::Tensor<double, 4> g_oovv_ab_;

    Eigen::Tensor<double, 4> t2_aa_, t2_bb_, t2_ab_;
};

}

#endif
