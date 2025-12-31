/**
 * @file src/scf/cholesky_uhf.cc
 * @brief UHF with Cholesky Decomposition (OPTIMIZED SPEED & MEMORY)
 */
#include "mshqc/cholesky_uhf.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

// Constructor
CholeskyUHF::CholeskyUHF(const Molecule& mol, const BasisSet& basis,
                         std::shared_ptr<IntegralEngine> integrals,
                         int n_alpha, int n_beta, const CholeskyUHFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals), config_(config),
      n_alpha_(n_alpha), n_beta_(n_beta),
      diis_a_(config.diis_max_vectors), diis_b_(config.diis_max_vectors) 
{
    nbasis_ = basis.n_basis_functions();
    energy_ = 0.0;
    is_cholesky_external_ = false; // Default: hitung sendiri
}

// [OPTIMISASI] Implementasi Setter
void CholeskyUHF::set_cholesky_vectors(const std::vector<Eigen::VectorXd>& vectors) {
    L_vectors_ = vectors;
    n_cholesky_vectors_ = vectors.size();
    is_cholesky_external_ = true; // Tandai bahwa vektor sudah ada
}

void CholeskyUHF::init_integrals_and_cholesky() {
    // 1. Integrals Standard (Overlap & Core Hamiltonian)
    S_ = integrals_->compute_overlap();
    H_ = integrals_->compute_kinetic() + integrals_->compute_nuclear();
    
    // 2. Orthogonalization (S^-1/2)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    X_ = es.operatorInverseSqrt();
    
    // 3. Cholesky Decomposition Logic
    if (is_cholesky_external_) {
        // [FAST PATH] Gunakan vektor yang di-inject dari luar
        if (config_.print_level > 0) {
            std::cout << "  > Using " << n_cholesky_vectors_ << " external Cholesky vectors (Skipping ERI calc).\n";
        }
    } 
    else {
        // [SLOW PATH] Hitung sendiri jika tidak ada injeksi
        if (L_vectors_.empty()) {
            if (config_.print_level > 0) std::cout << "  > Decomposing ERIs (Internal)..." << std::flush;
            
            auto eri_full = integrals_->compute_eri(); 
            cholesky_ = std::make_unique<integrals::CholeskyERI>(config_.cholesky_threshold);
            auto res = cholesky_->decompose(eri_full);
            
            L_vectors_ = cholesky_->get_L_vectors();
            n_cholesky_vectors_ = res.n_vectors;
            
            if (config_.print_level > 0) std::cout << " Done (" << n_cholesky_vectors_ << " vectors)\n";
        }
    }
}

// ... (Sisa fungsi initial_guess, build_fock, compute, dll SAMA SEPERTI SEBELUMNYA) ...
// (Untuk menghemat tempat, pastikan copy paste sisa fungsi dari file cholesky_uhf.cc Anda yang lama di sini)

void CholeskyUHF::initial_guess() {
    Eigen::MatrixXd H_bias = H_; 
    for(int i=0; i<nbasis_; i++) H_bias(i,i) -= 0.5;
    solve_fock(H_bias, C_a_, eps_a_);
    solve_fock(H_bias, C_b_, eps_b_);
    P_a_ = build_density(C_a_, n_alpha_);
    P_b_ = build_density(C_b_, n_beta_);
}

void CholeskyUHF::build_fock_cholesky() {
    F_a_ = H_; F_b_ = H_;
    Eigen::MatrixXd P_tot = P_a_ + P_b_;
    
    #pragma omp parallel 
    {
        Eigen::MatrixXd F_a_loc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd F_b_loc = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd T_a(nbasis_, nbasis_), T_b(nbasis_, nbasis_);

        #pragma omp for schedule(static)
        for (int K = 0; K < n_cholesky_vectors_; K++) {
            Eigen::Map<const Eigen::MatrixXd> L_K(L_vectors_[K].data(), nbasis_, nbasis_);
            double J_val = (P_tot.cwiseProduct(L_K)).sum();
            F_a_loc += J_val * L_K;
            F_b_loc += J_val * L_K;
            
            T_a.noalias() = P_a_ * L_K;
            F_a_loc.noalias() -= L_K * T_a;

            T_b.noalias() = P_b_ * L_K;
            F_b_loc.noalias() -= L_K * T_b;
        }
        #pragma omp critical
        { F_a_ += F_a_loc; F_b_ += F_b_loc; }
    }
}

void CholeskyUHF::solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps) {
    Eigen::MatrixXd Fp = X_.transpose() * F * X_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
    eps = es.eigenvalues();
    C = X_ * es.eigenvectors();
}

Eigen::MatrixXd CholeskyUHF::build_density(const Eigen::MatrixXd& C, int n_occ) {
    if (n_occ == 0) return Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    return C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
}

double CholeskyUHF::compute_energy() {
    return 0.5 * ((P_a_ * (H_ + F_a_)).trace() + (P_b_ * (H_ + F_b_)).trace());
}

SCFResult CholeskyUHF::compute() {
    init_integrals_and_cholesky();
    initial_guess();
    
    int iter = 0;
    for (iter = 1; iter <= config_.max_iterations; iter++) {
        energy_old_ = energy_;
        build_fock_cholesky();
        
        Eigen::MatrixXd err_a = F_a_*P_a_*S_ - S_*P_a_*F_a_;
        Eigen::MatrixXd err_b = F_b_*P_b_*S_ - S_*P_b_*F_b_;
        diis_a_.add_iteration(F_a_, err_a);
        diis_b_.add_iteration(F_b_, err_b);
        
        if(iter > 2) { F_a_ = diis_a_.extrapolate(); F_b_ = diis_b_.extrapolate(); }

        solve_fock(F_a_, C_a_, eps_a_);
        solve_fock(F_b_, C_b_, eps_b_);
        
        Eigen::MatrixXd P_a_new = build_density(C_a_, n_alpha_);
        Eigen::MatrixXd P_b_new = build_density(C_b_, n_beta_);
        
        if (iter > 1) { P_a_ = 0.7 * P_a_new + 0.3 * P_a_; P_b_ = 0.7 * P_b_new + 0.3 * P_b_; } 
        else { P_a_ = P_a_new; P_b_ = P_b_new; }
        
        energy_ = compute_energy();
        if (std::abs(energy_ - energy_old_) < config_.energy_threshold) break;
    }
    
    SCFResult res;
    res.energy_total = energy_ + mol_.nuclear_repulsion_energy();
    res.energy_electronic = energy_;
    res.iterations = iter;
    res.n_occ_alpha = n_alpha_; res.n_occ_beta = n_beta_;
    res.C_alpha = C_a_; res.C_beta = C_b_;
    res.orbital_energies_alpha = eps_a_; res.orbital_energies_beta = eps_b_;
    res.P_alpha = P_a_; res.P_beta = P_b_;
    return res;
}

double CholeskyUHF::compute_s_squared() const {
    double s_z = 0.5 * (n_alpha_ - n_beta_);
    double s2_ideal = s_z * (s_z + 1.0);
    Eigen::MatrixXd S_ab = C_a_.leftCols(n_alpha_).transpose() * S_ * C_b_.leftCols(n_beta_);
    return s2_ideal + n_beta_ - S_ab.squaredNorm();
}

} // namespace mshqc