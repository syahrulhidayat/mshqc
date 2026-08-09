 // ==============================================================================
 // Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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

#ifndef MSHQC_MPN_HIERARCHY_H
#define MSHQC_MPN_HIERARCHY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#ifdef I
#undef I
#endif

namespace mshqc {

struct MPnHierarchyResult {

    double e0_hf;

    double e1;

    double e2_mp2;
    double e2_aa;

    double e2_bb;

    double e2_ab;

    double e3_mp3;
    double e3_aa;

    double e3_bb;

    double e3_ab;

    double e4_mp4;
    double e4_s;

    double e4_d;

    double e4_t;

    double e4_q;

    double e5_mp5;
    double e5_t;

    double e5_q;

    double e5_p;

    double e_total_mp0;

    double e_total_mp1;

    double e_total_mp2;

    double e_total_mp3;

    double e_total_mp4;

    double e_total_mp5;

    Eigen::Tensor<double, 4> t2_aa_1;

    Eigen::Tensor<double, 4> t2_bb_1;

    Eigen::Tensor<double, 4> t2_ab_1;

    Eigen::Tensor<double, 2> t1_a_2;

    Eigen::Tensor<double, 2> t1_b_2;

    Eigen::Tensor<double, 4> t2_aa_2;

    Eigen::Tensor<double, 4> t2_bb_2;

    Eigen::Tensor<double, 4> t2_ab_2;

    Eigen::Tensor<double, 2> t1_a_3;

    Eigen::Tensor<double, 2> t1_b_3;

    Eigen::Tensor<double, 4> t2_aa_3;

    Eigen::Tensor<double, 4> t2_bb_3;

    Eigen::Tensor<double, 4> t2_ab_3;

    Eigen::Tensor<double, 6> t3_aaa_2;

    Eigen::Tensor<double, 6> t3_bbb_2;

    Eigen::Tensor<double, 6> t3_aab_2;

    Eigen::Tensor<double, 6> t3_abb_2;

    int n_occ_alpha;

    int n_occ_beta;

    int n_virt_alpha;

    int n_virt_beta;

    int n_basis;

    double norm_t2_1;

    double norm_t1_2;

    double norm_t2_2;

    double norm_t3_2;

    double norm_t1_3;

    double norm_t2_3;

    bool mp2_computed;

    bool mp3_computed;

    bool mp4_computed;

    bool mp5_computed;

    bool psi1_computed;

    bool psi3_computed;

    bool psi4_computed;

    std::string basis_name;

    std::string molecule;

    void print_energy_table() const;

    void print_wavefunction_summary() const;

    void print() const;
};

MPnHierarchyResult build_mpn_hierarchy(
    const struct SCFResult& uhf_result,
    const struct UMP2Result& ump2_result,
    const struct UMP3Result* ump3_result = nullptr,
    const void* ump4_result = nullptr,

    const void* ump5_result = nullptr

);

}

#endif
