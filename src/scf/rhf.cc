/**
 * @file rhf.cc
 * @brief Restricted Hartree-Fock (RHF) implementation for closed-shell systems
 * 
 * Implementation of RHF with single Fock matrix for both spin-up and spin-down electrons.
 * Simpler than UHF/ROHF - all electrons occupy same spatial orbitals with opposite spins.
 * 
 * Theory References:
 *   - C. C. J. Roothaan, Rev. Mod. Phys. 23, 69 (1951)
 *     [Original RHF equations - Roothaan equations]
 *   - R. K. Nesbet, Proc. R. Soc. London A 230, 312 (1955)
 *     [Self-consistent field theory]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [Section 3.4, pp. 138-146: RHF equations and derivation]
 *   - P.-O. Löwdin, J. Chem. Phys. 18, 365 (1950)
 *     [Symmetric orthogonalization method]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-11
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       Fock matrix construction follows Szabo & Ostlund Eq. (3.154).
 *       Density matrix: Eq. (3.145), Energy: Eq. (3.184).
 */

#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

RHF::RHF(const Molecule& mol,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         const SCFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals), config_(config),
      diis_(config.diis_max_vectors) {
    
    nbasis_ = basis.n_basis_functions();
    
    // Calculate number of electrons and check for closed-shell
    int n_electrons = mol.n_electrons();
    if (n_electrons % 2 != 0) {
        throw std::runtime_error("RHF: Molecule must have even number of electrons (closed-shell)");
    }
    
    n_occ_ = n_electrons / 2;  // Number of doubly-occupied orbitals
    
    if (config_.print_level >= 1) {
        std::cout << "\n=== RHF Calculation ===\n";
        std::cout << "Basis functions: " << nbasis_ << "\n";
        std::cout << "Electrons: " << n_electrons << "\n";
        std::cout << "Occupied orbitals: " << n_occ_ << "\n\n";
    }
}

void RHF::init_integrals() {
    // Compute 1-electron integrals
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.153)
    // H_μν = T_μν + V_μν
    S_ = integrals_->compute_overlap();
    Eigen::MatrixXd T = integrals_->compute_kinetic();
    Eigen::MatrixXd V = integrals_->compute_nuclear();
    H_ = T + V;  // Core Hamiltonian
    
    // Löwdin symmetric orthogonalization: X = S^(-1/2)
    // REFERENCE: Löwdin (1950), J. Chem. Phys. 18, 365, Eq. (6)
    // S = U s U^T → S^(-1/2) = U s^(-1/2) U^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    Eigen::VectorXd s_vals = es.eigenvalues();
    Eigen::MatrixXd s_vecs = es.eigenvectors();
    
    // Compute s^(-1/2)
    Eigen::VectorXd s_inv_sqrt = s_vals.array().rsqrt();
    X_ = s_vecs * s_inv_sqrt.asDiagonal() * s_vecs.transpose();
    
    if (config_.print_level >= 2) {
        std::cout << "Orthogonalization (Löwdin symmetric):\n";
        std::cout << "  Smallest eigenvalue of S: " << s_vals(0) << "\n";
        std::cout << "  Largest eigenvalue of S: " << s_vals(s_vals.size()-1) << "\n\n";
    }
}

void RHF::initial_guess() {
    // Core Hamiltonian guess
    // REFERENCE: Szabo & Ostlund (1996), Section 3.4.4, p. 143
    // Initial guess: diagonalize core Hamiltonian H
    
    // Transform H to orthogonal basis and diagonalize
    Eigen::MatrixXd H_prime = X_.transpose() * H_ * X_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_prime);
    
    eps_ = es.eigenvalues();
    Eigen::MatrixXd C_prime = es.eigenvectors();
    C_ = X_ * C_prime;  // Back-transform to AO basis
    
    // Build initial density
    P_ = build_density();
    
    energy_ = 0.0;
    energy_old_ = 0.0;
    
    if (config_.print_level >= 2) {
        std::cout << "Initial guess (core Hamiltonian):\n";
        std::cout << "  HOMO energy: " << eps_(n_occ_-1) << " Ha\n";
        std::cout << "  LUMO energy: " << eps_(n_occ_) << " Ha\n";
        std::cout << "  Gap: " << (eps_(n_occ_) - eps_(n_occ_-1)) << " Ha\n\n";
    }
}

Eigen::MatrixXd RHF::build_density() {
    // Build density matrix from MO coefficients
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.145), p. 139
    // P_μν = 2 Σ_i^{N/2} C_μi C_νi (factor of 2 for doubly-occupied)
    
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    for (int i = 0; i < n_occ_; i++) {
        P += 2.0 * C_.col(i) * C_.col(i).transpose();
    }
    
    return P;
}

void RHF::build_fock() {
    // Build Fock matrix F = H + G
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.154), p. 141
    // G_μν = Σ_λσ P_λσ [(μν|λσ) - 0.5(μλ|νσ)]
    //      = Σ_λσ P_λσ [J_μν - 0.5 K_μν]
    // where J is Coulomb, K is exchange
    
    F_ = H_;
    
    // Get ERI tensor
    auto eri = integrals_->compute_eri();
    
    // Build 2-electron part (G matrix)
    for (size_t mu = 0; mu < nbasis_; mu++) {
        for (size_t nu = 0; nu < nbasis_; nu++) {
            double g = 0.0;
            
            for (size_t lam = 0; lam < nbasis_; lam++) {
                for (size_t sig = 0; sig < nbasis_; sig++) {
                    // Coulomb: J = (μν|λσ)
                    double j_term = eri(mu, nu, lam, sig);
                    
                    // Exchange: K = (μλ|νσ)
                    double k_term = eri(mu, lam, nu, sig);
                    
                    // G = J - 0.5 K (factor of 0.5 from averaging spins)
                    g += P_(lam, sig) * (j_term - 0.5 * k_term);
                }
            }
            
            F_(mu, nu) += g;
        }
    }
}

double RHF::compute_energy() {
    // Compute electronic energy
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.184), p. 150
    // E_elec = Σ_μν P_μν (H_μν + F_μν) / 2
    //        = Tr[P(H + F)] / 2
    
    double e = 0.5 * (P_ * (H_ + F_)).trace();
    return e;
}

void RHF::solve_fock() {
    // Solve generalized eigenvalue problem: F C = S C ε
    // REFERENCE: Szabo & Ostlund (1996), Section 3.4.5, pp. 143-145
    // 
    // Method: Orthogonalize with X = S^(-1/2)
    // 1. Transform: F' = X^T F X
    // 2. Diagonalize: F' C' = C' ε
    // 3. Back-transform: C = X C'
    
    // Transform to orthogonal basis
    Eigen::MatrixXd F_prime = X_.transpose() * F_ * X_;
    
    // Diagonalize
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_prime);
    eps_ = es.eigenvalues();
    Eigen::MatrixXd C_prime = es.eigenvectors();
    
    // Back-transform to AO basis
    C_ = X_ * C_prime;
}

bool RHF::check_convergence() {
    double de = std::abs(energy_ - energy_old_);
    return de < config_.energy_threshold;
}

void RHF::print_iter(int iter, double de, double dp) {
    if (config_.print_level < 1) return;
    
    std::cout << std::setw(4) << iter
              << std::setw(18) << std::fixed << std::setprecision(10) << energy_
              << std::setw(14) << std::scientific << std::setprecision(4) << de
              << std::setw(14) << dp << "\n";
}

void RHF::print_final(const SCFResult& result) {
    if (config_.print_level < 1) return;
    
    std::cout << "\n=== RHF Results ===\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Nuclear repulsion: " << result.energy_nuclear << " Ha\n";
    std::cout << "Electronic energy: " << result.energy_electronic << " Ha\n";
    std::cout << "Total energy:      " << result.energy_total << " Ha\n";
    std::cout << "Iterations:        " << result.iterations << "\n";
    std::cout << "Converged:         " << (result.converged ? "Yes" : "No") << "\n";
    
    // Print orbital energies
    if (config_.print_level >= 2) {
        std::cout << "\nOrbital Energies (Ha):\n";
        for (int i = 0; i < std::min(n_occ_ + 5, (int)nbasis_); i++) {
            std::string label = (i < n_occ_) ? "occ" : "vir";
            std::cout << "  " << std::setw(3) << i+1 
                      << " (" << label << "): " 
                      << std::setw(12) << eps_(i) << "\n";
        }
        std::cout << "\n";
        std::cout << "HOMO: " << eps_(n_occ_-1) << " Ha\n";
        std::cout << "LUMO: " << eps_(n_occ_) << " Ha\n";
        std::cout << "Gap:  " << (eps_(n_occ_) - eps_(n_occ_-1)) << " Ha\n";
    }
}

SCFResult RHF::compute() {
    // Initialize
    init_integrals();
    initial_guess();
    
    if (config_.print_level >= 1) {
        std::cout << "Starting RHF SCF iterations...\n";
        std::cout << std::setw(4) << "Iter"
                  << std::setw(18) << "Energy (Ha)"
                  << std::setw(14) << "dE"
                  << std::setw(14) << "dP" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }
    
    // SCF loop
    bool converged = false;
    int iter;
    
    for (iter = 1; iter <= config_.max_iterations; iter++) {
        energy_old_ = energy_;
        Eigen::MatrixXd P_old = P_;
        
        // Build Fock matrix
        build_fock();
        
        // DIIS extrapolation (after few iterations)
        if (iter > 2 && diis_.can_extrapolate()) {
            // Compute DIIS error: e = FPS - SPF
            Eigen::MatrixXd FPS = F_ * P_ * S_;
            Eigen::MatrixXd SPF = S_ * P_ * F_;
            Eigen::MatrixXd error = FPS - SPF;
            
            double error_norm = error.norm();
            
            // Add to DIIS and extrapolate if error small enough
            if (error_norm < config_.diis_threshold) {
                diis_.add_iteration(F_, error);
                if (diis_.can_extrapolate()) {
                    F_ = diis_.extrapolate();
                }
            }
        }
        
        // Solve Fock equation
        solve_fock();
        
        // Build new density
        P_ = build_density();
        
        // Compute energy
        energy_ = compute_energy();
        
        // Check convergence
        double de = std::abs(energy_ - energy_old_);
        double dp = (P_ - P_old).norm();
        
        print_iter(iter, de, dp);
        
        if (check_convergence() && dp < config_.density_threshold) {
            converged = true;
            break;
        }
    }
    
    // Prepare result
    SCFResult result;
    result.energy_electronic = energy_;
    result.energy_nuclear = mol_.nuclear_repulsion_energy();
    result.energy_total = energy_ + result.energy_nuclear;
    result.orbital_energies_alpha = eps_;
    result.orbital_energies_beta = eps_;  // Same for RHF
    result.C_alpha = C_;
    result.C_beta = C_;  // Same for RHF
    result.P_alpha = 0.5 * P_;  // Half density for each spin
    result.P_beta = 0.5 * P_;
    result.F_alpha = F_;
    result.F_beta = F_;  // Same for RHF
    result.iterations = iter;
    result.converged = converged;
    result.n_occ_alpha = n_occ_;
    result.n_occ_beta = n_occ_;
    
    print_final(result);
    
    return result;
}

} // namespace mshqc
