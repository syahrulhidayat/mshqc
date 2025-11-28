/**
 * @file uhf.cc
 * @brief Unrestricted Hartree-Fock (UHF) implementation for open-shell systems
 * 
 * Implementation of UHF with separate α and β spin Fock matrices.
 * Includes DIIS convergence acceleration and spin contamination analysis.
 * 
 * Theory References:
 *   - J. A. Pople & R. K. Nesbet, J. Chem. Phys. 22, 571 (1954)
 *     [Original UHF formulation]
 *   - G. Berthier, C. R. Acad. Sci. Paris 238, 91 (1954)
 *     [Independent UHF development]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [Section 3.8.5, pp. 108-110: UHF equations and spin contamination]
 *   - P.-O. Löwdin, J. Chem. Phys. 18, 365 (1950)
 *     [Symmetric orthogonalization method]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       Fock matrix construction follows Szabo & Ostlund Eq. (3.198).
 *       Spin contamination computed via Eq. (3.199).
 */

#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

UHF::UHF(const Molecule& mol, 
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         int n_alpha,
         int n_beta,
         const SCFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals), config_(config),
      n_alpha_(n_alpha), n_beta_(n_beta) {
    
    nbasis_ = basis.n_basis_functions();
    
    // Check valid electron count
    if (n_alpha_ < n_beta_) {
        throw std::runtime_error("UHF: n_alpha must be >= n_beta");
    }
}

void UHF::init_integrals() {
    // Compute 1-electron integrals
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.153)
    S_ = integrals_->compute_overlap();
    Eigen::MatrixXd T = integrals_->compute_kinetic();
    Eigen::MatrixXd V = integrals_->compute_nuclear();
    H_ = T + V;  // Core Hamiltonian
    
    // Löwdin symmetric orthogonalization: X = S^(-1/2)
    // REFERENCE: Löwdin (1950), J. Chem. Phys. 18, 365
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    Eigen::VectorXd s_vals = es.eigenvalues();
    Eigen::MatrixXd s_vecs = es.eigenvectors();
    
    // S^(-1/2) = U * s^(-1/2) * U^T
    Eigen::VectorXd s_inv_sqrt = s_vals.array().rsqrt();
    X_ = s_vecs * s_inv_sqrt.asDiagonal() * s_vecs.transpose();
}

void UHF::initial_guess() {
    // Core Hamiltonian guess
    // REFERENCE: Szabo & Ostlund (1996), Section 3.4.4
    
    // Diagonalize H for both spins (same initially)
    solve_fock(H_, C_a_, eps_a_);
    solve_fock(H_, C_b_, eps_b_);
    
    // Build initial densities
    P_a_ = build_density(C_a_, n_alpha_);
    P_b_ = build_density(C_b_, n_beta_);
    
    energy_ = 0.0;
    energy_old_ = 0.0;
}

void UHF::build_fock() {
    // Build separate Fock for α and β
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.198)
    // F^α = H + J[P^α + P^β] - K[P^α]
    // F^β = H + J[P^α + P^β] - K[P^β]
    
    F_a_ = H_;
    F_b_ = H_;
    
    // Get ERI tensor
    auto eri = integrals_->compute_eri();
    
    // Total density for Coulomb
    Eigen::MatrixXd P_tot = P_a_ + P_b_;
    
    // Build G matrices (2-electron part)
    for (size_t mu = 0; mu < nbasis_; mu++) {
        for (size_t nu = 0; nu < nbasis_; nu++) {
            double g_a = 0.0, g_b = 0.0;
            
            // Coulomb: J[P_tot]
            // K^α: K[P^α], K^β: K[P^β]
            for (size_t lam = 0; lam < nbasis_; lam++) {
                for (size_t sig = 0; sig < nbasis_; sig++) {
                    double eri_val = eri(mu, nu, lam, sig);
                    
                    // J term (same for both)
                    g_a += P_tot(lam, sig) * eri_val;
                    g_b += P_tot(lam, sig) * eri_val;
                    
                    // K term (exchange - different for each spin)
                    double eri_ex = eri(mu, lam, nu, sig);
                    g_a -= P_a_(lam, sig) * eri_ex;
                    g_b -= P_b_(lam, sig) * eri_ex;
                }
            }
            
            F_a_(mu, nu) += g_a;
            F_b_(mu, nu) += g_b;
        }
    }
}

Eigen::MatrixXd UHF::build_density(const Eigen::MatrixXd& C, int n) {
    // P_μν = Σ_i^occ C_μi C_νi
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.145)
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    for (int i = 0; i < n; i++) {
        P += C.col(i) * C.col(i).transpose();
    }
    
    return P;
}

double UHF::compute_energy() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.184)
    // E = ½ Tr[(P^α + P^β)(H + H)] + ½[Tr(P^α F^α) + Tr(P^β F^β)]
    //   = Tr[(P^α + P^β)H] + ½[Tr(P^α F^α) + Tr(P^β F^β)]
    // Simplifies to: E = ½[Tr(P^α(H+F^α)) + Tr(P^β(H+F^β))]
    
    double e_a = 0.5 * (P_a_ * (H_ + F_a_)).trace();
    double e_b = 0.5 * (P_b_ * (H_ + F_b_)).trace();
    
    return e_a + e_b;
}

void UHF::solve_fock(const Eigen::MatrixXd& F,
                     Eigen::MatrixXd& C,
                     Eigen::VectorXd& eps) {
    // Transform to orthogonal basis and diagonalize
    // REFERENCE: Szabo & Ostlund (1996), Section 3.4.5
    // F' = X^T F X, F' C' = C' ε, C = X C'
    
    Eigen::MatrixXd Fp = X_.transpose() * F * X_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
    
    eps = es.eigenvalues();
    Eigen::MatrixXd Cp = es.eigenvectors();
    C = X_ * Cp;
}

bool UHF::check_convergence() {
    double de = std::abs(energy_ - energy_old_);
    return de < config_.energy_threshold;
}

void UHF::print_iter(int iter, double de, double dp) {
    if (config_.print_level < 1) return;
    
    std::cout << std::setw(4) << iter
              << std::setw(18) << std::fixed << std::setprecision(10) << energy_
              << std::setw(14) << std::scientific << std::setprecision(4) << de
              << std::setw(14) << dp << "\n";
}

void UHF::print_final(const SCFResult& r) {
    if (config_.print_level < 1) return;
    
    std::cout << "\n=== UHF Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Nuclear repulsion: " << r.energy_nuclear << " Ha\n";
    std::cout << "Electronic energy: " << r.energy_electronic << " Ha\n";
    std::cout << "Total energy:      " << r.energy_total << " Ha\n";
    std::cout << "Iterations:        " << r.iterations << "\n";
    
    // Compute and show spin contamination
    double s2 = compute_s_squared(r);
    double s2_exact = 0.25 * (n_alpha_ - n_beta_) * (n_alpha_ - n_beta_ + 2);
    std::cout << "<S²>:              " << s2 << " (exact: " << s2_exact << ")\n";
    std::cout << "Spin contam:       " << (s2 - s2_exact) << "\n";
}

double UHF::compute_s_squared(const SCFResult& result) {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.199)
    // <S²> = S(S+1) + N_β - Σ_ij |<φ^α_i|φ^β_j>|²
    
    int na = result.n_occ_alpha;
    int nb = result.n_occ_beta;
    
    // Exact value for pure spin state
    double s_exact = 0.25 * (na - nb) * (na - nb + 2);
    
    // Overlap matrix S_ij = <φ^α_i|φ^β_j> = C^α^T S C^β
    Eigen::MatrixXd C_a_occ = result.C_alpha.leftCols(na);
    Eigen::MatrixXd C_b_occ = result.C_beta.leftCols(nb);
    Eigen::MatrixXd S_ab = C_a_occ.transpose() * S_ * C_b_occ;
    
    // Sum of squared overlaps
    double overlap_sum = S_ab.array().abs2().sum();
    
    // <S²> includes contamination from overlap
    return s_exact + nb - overlap_sum;
}

SCFResult UHF::compute() {
    std::cout << "\n====================================\n";
    std::cout << "  Unrestricted Hartree-Fock (UHF)\n";
    std::cout << "====================================\n";
    std::cout << "Basis functions: " << nbasis_ << "\n";
    std::cout << "Alpha electrons: " << n_alpha_ << "\n";
    std::cout << "Beta electrons:  " << n_beta_ << "\n";
    std::cout << "====================================\n";
    
    // Initialize
    init_integrals();
    initial_guess();
    
    // Setup DIIS
    DIIS diis_a(config_.diis_max_vectors);
    DIIS diis_b(config_.diis_max_vectors);
    
    if (config_.print_level > 0) {
        std::cout << "\nSCF Iterations:\n";
        std::cout << std::setw(4) << "Iter"
                  << std::setw(18) << "Energy (Ha)"
                  << std::setw(14) << "ΔE"
                  << std::setw(14) << "||ΔP||\n";
        std::cout << std::string(50, '-') << "\n";
    }
    
    bool converged = false;
    int iter;
    
    for (iter = 1; iter <= config_.max_iterations; iter++) {
        energy_old_ = energy_;
        
        // Build Fock
        build_fock();
        
        // DIIS acceleration
        // Error: e = [F,P]S = FPS - SPF
        Eigen::MatrixXd e_a = F_a_ * P_a_ * S_ - S_ * P_a_ * F_a_;
        Eigen::MatrixXd e_b = F_b_ * P_b_ * S_ - S_ * P_b_ * F_b_;
        double err_norm = std::max(e_a.norm(), e_b.norm());
        
        bool diis_used = false;
        if (err_norm < config_.diis_threshold) {
            diis_a.add_iteration(F_a_, e_a);
            diis_b.add_iteration(F_b_, e_b);
            
            if (diis_a.can_extrapolate()) {
                F_a_ = diis_a.extrapolate();
                F_b_ = diis_b.extrapolate();
                diis_used = true;
            }
        }
        
        // Diagonalize Fock
        solve_fock(F_a_, C_a_, eps_a_);
        solve_fock(F_b_, C_b_, eps_b_);
        
        // Update densities
        Eigen::MatrixXd P_a_old = P_a_;
        Eigen::MatrixXd P_b_old = P_b_;
        P_a_ = build_density(C_a_, n_alpha_);
        P_b_ = build_density(C_b_, n_beta_);
        
        // Compute energy
        energy_ = compute_energy();
        
        // Check convergence
        double de = std::abs(energy_ - energy_old_);
        double dp_a = (P_a_ - P_a_old).norm();
        double dp_b = (P_b_ - P_b_old).norm();
        double dp = std::max(dp_a, dp_b);
        
        print_iter(iter, de, dp);
        
        if (check_convergence() && dp < config_.density_threshold) {
            converged = true;
            break;
        }
    }
    
    // Prepare result
    SCFResult result;
    result.energy_nuclear = mol_.nuclear_repulsion_energy();
    result.energy_electronic = energy_;
    result.energy_total = energy_ + result.energy_nuclear;
    result.orbital_energies_alpha = eps_a_;
    result.orbital_energies_beta = eps_b_;
    result.C_alpha = C_a_;
    result.C_beta = C_b_;
    result.P_alpha = P_a_;
    result.P_beta = P_b_;
    result.F_alpha = F_a_;
    result.F_beta = F_b_;
    result.iterations = iter;
    result.converged = converged;
    result.n_occ_alpha = n_alpha_;
    result.n_occ_beta = n_beta_;
    
    print_final(result);
    
    return result;
}

} // namespace mshqc
