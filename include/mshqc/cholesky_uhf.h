#ifndef MSHQC_CHOLESKY_UHF_H
#define MSHQC_CHOLESKY_UHF_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/scf.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace mshqc {

struct CholeskyUHFConfig {
    double energy_threshold = 1e-8;
    double cholesky_threshold = 1e-6;
    int max_iterations = 100;
    int diis_max_vectors = 6;
    int print_level = 1;
};

class CholeskyUHF {
public:
    CholeskyUHF(const Molecule& mol, const BasisSet& basis,
                std::shared_ptr<IntegralEngine> integrals,
                int n_alpha, int n_beta, const CholeskyUHFConfig& config);

    SCFResult compute();
    
    // [OPTIMISASI PENTING] Fungsi untuk inject vektor Cholesky dari luar
    void set_cholesky_vectors(const std::vector<Eigen::VectorXd>& vectors);
    double compute_s_squared() const;

private:
    Molecule mol_;
    BasisSet basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    CholeskyUHFConfig config_;
    
    int n_alpha_, n_beta_;
    int nbasis_;
    
    // Cholesky Data
    std::unique_ptr<integrals::CholeskyERI> cholesky_;
    std::vector<Eigen::VectorXd> L_vectors_;
    int n_cholesky_vectors_ = 0;
    bool is_cholesky_external_ = false; // Flag optimasi

    // SCF variables
    Eigen::MatrixXd H_, S_, X_;
    Eigen::MatrixXd F_a_, F_b_;
    Eigen::MatrixXd P_a_, P_b_;
    Eigen::MatrixXd C_a_, C_b_;
    Eigen::VectorXd eps_a_, eps_b_;
    
    double energy_ = 0.0;
    double energy_old_ = 0.0;
    
    // DIIS
    DIIS diis_a_;
    DIIS diis_b_;

    // Helpers
    void init_integrals_and_cholesky();
    void initial_guess();
    void build_fock_cholesky();
    void solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps);
    Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ);
    double compute_energy();
    
};

} // namespace mshqc

#endif