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


#ifndef MSHQC_SCF_H
#define MSHQC_SCF_H

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/symmetry/point_group.h"
#include "mshqc/symmetry/petite_list.h"
#include "mshqc/symmetry/molecule_sym.h"
#include "mshqc/core/fock_builder.h"
#include "mshqc/integrals/cholesky_eri.h"
#ifdef I
#undef I
#endif

namespace mshqc {

struct SCFConfig {
    int max_iterations = 100;
    double energy_threshold = 1e-10;
    double density_threshold = 1e-8;
    int print_level = 1;
    double diis_threshold = 1e-2;
    int diis_max_vectors = 8;
    std::string scf_type = "direct";
    std::string eri_method = "exact";
    double df_threshold = 1e-10;
    double cholesky_threshold = 1e-9;
    bool use_df = false;
    std::string aux_basis_name = "cc-pVDZ-JKFIT";
};

struct SCFResult {
    double energy_total = 0.0;
    double energy_electronic = 0.0;
    double energy_nuclear = 0.0;
    int iterations = 0;
    bool converged = false;
    Eigen::MatrixXd C_alpha, C_beta;
    Eigen::MatrixXd P_alpha, P_beta;
    Eigen::MatrixXd F_alpha, F_beta;
    Eigen::VectorXd orbital_energies_alpha, orbital_energies_beta;
    std::vector<int> irreps_alpha;
    std::vector<int> irreps_beta;
    int n_occ_alpha = 0;
    int n_occ_beta = 0;
    Eigen::MatrixXd L_mat;
};

class BaseSCF {
protected:

    Molecule mol_;
    BasisSet basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    std::shared_ptr<PointGroup> pg_;
    std::shared_ptr<PetiteList> pl_;
    std::unique_ptr<BasisSymmetrizer> symmetrizer_;
    SCFConfig config_;

    int nbasis_;
    int n_alpha_;
    int n_beta_;
    double energy_ = 0.0;
    double energy_old_ = 0.0;
    int iter_scf_ = 0;

    Eigen::MatrixXd S_, H_, X_, schwarz_;
    Eigen::MatrixXd C_alpha_, C_beta_;
    Eigen::MatrixXd P_alpha_, P_beta_;
    Eigen::MatrixXd F_alpha_, F_beta_;
    Eigen::VectorXd eps_alpha_, eps_beta_;
    Eigen::VectorXd occ_numbers_alpha_, occ_numbers_beta_;
    std::vector<int> salc_irreps_;

    std::vector<int> shell_starts_, shell_sizes_;
    std::vector<std::pair<int, int>> row_map_;
    std::vector<double> J_val_, K_val_;
    std::vector<int> J_ind_, K_ind_;
    std::vector<size_t> J_ptr_, K_ptr_;
    Eigen::MatrixXd schwarz_basis_;

    std::unique_ptr<integrals::CholeskyERI> internal_cholesky_;
    std::vector<Eigen::MatrixXd> L_vecs_;
    Eigen::MatrixXd L_mat_;

    std::unique_ptr<FockBuilder> fock_engine_;

    void init_integrals();
    void init_integrals_incore();
    void init_integrals_cholesky();
    Eigen::MatrixXd precompute_shell_schwarz();
    Eigen::VectorXd smear_electrons(const Eigen::VectorXd& eps, int target_electrons);
    void solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps);
    void print_final(const SCFResult& r);

    virtual void initial_guess() = 0;
    virtual void update_densities() = 0;
    virtual void build_fock_matrix() = 0;
    virtual double compute_energy() = 0;

public:
    BaseSCF(const Molecule& mol, const BasisSet& basis,
            std::shared_ptr<IntegralEngine> integrals,
            std::shared_ptr<PointGroup> pg,
            std::shared_ptr<PetiteList> pl,
            int n_alpha, int n_beta,
            const SCFConfig& config);

    virtual ~BaseSCF() = default;

    virtual SCFResult compute();

    double energy() const { return energy_ + mol_.nuclear_repulsion_energy(); }
    const Molecule& molecule() const { return mol_; }
    const BasisSet& basis() const { return basis_; }
    std::shared_ptr<PetiteList> get_petite_list() const { return pl_; }
};

class RHF : public BaseSCF {
public:
    SCFResult compute() override;

public:
    RHF(const Molecule& mol, const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        std::shared_ptr<PointGroup> pg,
        std::shared_ptr<PetiteList> pl,
        const SCFConfig& config);

    RHF(const Molecule& mol, const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const SCFConfig& config)
        : RHF(mol, basis, integrals, nullptr, nullptr, config) {}

protected:
    void initial_guess() override;
    void update_densities() override;
    void build_fock_matrix() override;
    double compute_energy() override;

private:
    Eigen::MatrixXd G_J_accum_;
    Eigen::MatrixXd G_accum_;
    Eigen::MatrixXd P_old_;

};

class ROHF : public BaseSCF {
public:
    SCFResult compute() override;

public:
    ROHF(const Molecule& mol, const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         std::shared_ptr<PointGroup> pg,
         std::shared_ptr<PetiteList> pl,
         int n_alpha, int n_beta, const SCFConfig& config);

    ROHF(const Molecule& mol, const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         int n_alpha, int n_beta, const SCFConfig& config)
         : ROHF(mol, basis, integrals, nullptr, nullptr, n_alpha, n_beta, config) {}

    SCFResult run() { return compute(); }

protected:
    void initial_guess() override;
    void update_densities() override;
    void build_fock_matrix() override;
    double compute_energy() override;

private:
    Eigen::MatrixXd P_alpha_old_, P_beta_old_;
    Eigen::MatrixXd G_accum_a_, G_accum_b_;
    Eigen::MatrixXd G_J_accum_a_;
    Eigen::MatrixXd G_J_accum_b_;
    Eigen::MatrixXd C_;
};

class UHF : public BaseSCF {
public:
    SCFResult compute() override;

public:
    UHF(const Molecule& mol, const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        std::shared_ptr<PointGroup> pg,
        std::shared_ptr<PetiteList> pl,
        int n_alpha, int n_beta, const SCFConfig& config);

    UHF(const Molecule& mol, const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        int n_alpha, int n_beta, const SCFConfig& config)
        : UHF(mol, basis, integrals, nullptr, nullptr, n_alpha, n_beta, config) {}

protected:
    void initial_guess() override;
    void update_densities() override;
    void build_fock_matrix() override;
    double compute_energy() override;

private:
    Eigen::MatrixXd P_alpha_old_, P_beta_old_;
    Eigen::MatrixXd G_accum_a_, G_accum_b_;
    Eigen::MatrixXd G_J_accum_a_;
    Eigen::MatrixXd G_J_accum_b_;
    Eigen::MatrixXd C_;
    double compute_s2(const Eigen::MatrixXd& Ca, const Eigen::MatrixXd& Cb, const Eigen::MatrixXd& S);
};

}

#endif
