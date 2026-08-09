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

#ifndef MSHQC_INTEGRALS_H
#define MSHQC_INTEGRALS_H

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace mshqc {

using ERITensor = Eigen::Tensor<double, 4>;

class IntegralEngine {
public:
    IntegralEngine(const Molecule& mol, const BasisSet& basis);
    ~IntegralEngine();

    IntegralEngine(const IntegralEngine&) = delete;
    IntegralEngine& operator=(const IntegralEngine&) = delete;
    IntegralEngine(IntegralEngine&&) = default;
    IntegralEngine& operator=(IntegralEngine&&) = default;

    Eigen::MatrixXd compute_overlap();
    Eigen::MatrixXd compute_kinetic();
    Eigen::MatrixXd compute_nuclear();
    Eigen::MatrixXd compute_core_hamiltonian();

    const Eigen::Tensor<double, 4>& compute_eri();

    size_t nbasis() const { return nbasis_; }

    Eigen::Tensor<double, 3> compute_3center_eri(const BasisSet& aux_basis);
    Eigen::MatrixXd compute_2center_eri(const BasisSet& aux_basis);
    double compute_single_eri(int mu, int nu, int lam, int sig);
    Eigen::VectorXd compute_eri_diagonal();
    Eigen::VectorXd compute_eri_column(int pivot_index);

    std::vector<double> compute_2c2e_block(int sh_P, int sh_Q);

    std::vector<double> compute_3c2e_block(int sh_i, int sh_j, int sh_P);

    const double* compute_shell_block_ptr(int sh_a, int sh_b, int sh_c, int sh_d, size_t& size_out);
    const std::vector<double>& compute_shell_block(int sh_a, int sh_b, int sh_c, int sh_d);
    const std::vector<int>& get_bas() const { return bas_; }
    const std::vector<int>& get_atm() const { return atm_; }
    const std::vector<double>& get_env() const { return env_; }

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    size_t nbasis_;
    bool cache_valid = false;
    Eigen::Tensor<double, 4> cached_eri;
    std::vector<int> atm_;
    std::vector<int> bas_;
    std::vector<double> env_;

    void* opt_ = nullptr;

    void convert_basis_to_libcint();

    int find_atom_index(const std::array<double, 3>& center);
};

}

#endif
