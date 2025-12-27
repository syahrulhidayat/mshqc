/**
 * @file rohf.cc
 * @brief Restricted Open-shell Hartree-Fock (ROHF) implementation
 * 
 * Implementation of ROHF for systems with unpaired electrons.
 * Uses separate α and β spin densities with common spatial orbitals.
 * Effective Fock operator approach for open-shell coupling.
 * 
 * Theory References:
 *   - C. C. J. Roothaan, Rev. Mod. Phys. 32, 179 (1960)
 *     [Original ROHF formulation for doublet and higher multiplets]
 *   - R. McWeeny & G. Diercksen, J. Chem. Phys. 49, 4852 (1968)
 *     [Effective Fock operator method]
 *   - A. Szabo & N. S. Ostlund, "Modern Quantum Chemistry" (1996)
 *     [Section 3.8.7, pp. 108-110: ROHF equations]
 *   - R. McWeeny, "Methods of Molecular Quantum Mechanics" (1989)
 *     [Comprehensive ROHF theory and implementation]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-29
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note This is an original implementation derived from published theory.
 *       No code was copied from existing quantum chemistry software.
 *       Fock matrices built separately for α and β spins (Eq. 3.184).
 *       Energy computed via Szabo & Ostlund Eq. (3.184).
 */

#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

ROHF::ROHF(const Molecule& mol,
           const BasisSet& basis,
           int n_alpha,
           int n_beta,
           const SCFConfig& config)
    : mol_(mol), basis_(basis), nbasis_(basis.n_basis_functions()),
      n_alpha_(n_alpha), n_beta_(n_beta), config_(config),
      energy_(0.0), energy_old_(0.0) {
    
    // Validate electron configuration
    if (n_alpha < n_beta) {
        throw std::runtime_error("ROHF requires n_alpha >= n_beta");
    }
    
    if (n_alpha > static_cast<int>(nbasis_) || n_beta > static_cast<int>(nbasis_)) {
        throw std::runtime_error("Number of electrons exceeds basis functions");
    }
    
    // Initialize matrices
    C_alpha_.resize(nbasis_, nbasis_);
    C_beta_.resize(nbasis_, nbasis_);
    P_alpha_.resize(nbasis_, nbasis_);
    P_beta_.resize(nbasis_, nbasis_);
    F_alpha_.resize(nbasis_, nbasis_);
    F_beta_.resize(nbasis_, nbasis_);
    eps_alpha_.resize(nbasis_);
    eps_beta_.resize(nbasis_);
    
    C_alpha_.setZero();
    C_beta_.setZero();
    P_alpha_.setZero();
    P_beta_.setZero();
    
    // Initialize ERI tensor (for small-to-medium systems)
    // Memory usage: nbasis^4 × 8 bytes
    //   cc-pVTZ (30):      6.5 MB
    //   aug-cc-pV5Z (127): 2.08 GB
    //   Limit (150):       4.05 GB
    if (nbasis_ < 150) {
        ERI_ = Eigen::Tensor<double, 4>(nbasis_, nbasis_, nbasis_, nbasis_);
        ERI_.setZero();
    }
}

void ROHF::initialize_integrals() {
    /**
     * Compute all required integrals using IntegralEngine
     */
    if (config_.print_level > 0) {
        std::cout << "\nComputing one- and two-electron integrals...\n";
    }
    
    integrals_ = std::make_unique<IntegralEngine>(mol_, basis_);
    
    // One-electron integrals
    S_ = integrals_->compute_overlap();
    H_ = integrals_->compute_core_hamiltonian();
    
    // Two-electron integrals (ERI tensor)
    // For small-to-medium systems (nbasis < 150), precompute full ERI tensor
    // Examples:
    //   Li/STO-3G:        5^4 = 625 elements (~5 KB)
    //   Li/cc-pVTZ:      30^4 = 810K elements (~6.5 MB)
    //   Li/aug-cc-pV5Z: 127^4 = 260M elements (~2.08 GB)
    if (nbasis_ < 150) {
        size_t num_elements = nbasis_ * nbasis_ * nbasis_ * nbasis_;
        double size_mb = (num_elements * sizeof(double)) / (1024.0 * 1024.0);
        
        if (config_.print_level > 0) {
            std::cout << "  Computing ERI tensor (" << nbasis_ << "^4 = " 
                      << num_elements << " elements, ";
            if (size_mb > 1024) {
                std::cout << std::fixed << std::setprecision(2) 
                          << (size_mb / 1024.0) << " GB)...\n";
            } else {
                std::cout << std::fixed << std::setprecision(1) 
                          << size_mb << " MB)...\n";
            }
        }
        ERI_ = integrals_->compute_eri();
        if (config_.print_level > 0) {
            std::cout << "  ERI computed successfully.\n";
        }
    } else {
        throw std::runtime_error("Large systems (nbasis >= 150) not yet supported. Need on-the-fly ERI.");
    }
    
    if (config_.print_level > 0) {
        std::cout << "  Basis functions: " << nbasis_ << "\n";
        std::cout << "  All integrals computed successfully.\n";
    }
}

void ROHF::initial_guess() {
    // REFERENCE: Szabo & Ostlund (1996), Section 3.4.4, p. 143
    // Core Hamiltonian guess: H C = S C ε
    
    if (config_.print_level > 0) {
        std::cout << "\nForming initial guess (core Hamiltonian)...\n";
    }
    
    // Solve H C = S C ε for both spins
    solve_fock(H_, C_alpha_, eps_alpha_);
    solve_fock(H_, C_beta_, eps_beta_);
    
    // Build initial density matrices
    P_alpha_ = build_density(C_alpha_, n_alpha_);
    P_beta_ = build_density(C_beta_, n_beta_);
    
    if (config_.print_level > 1) {
        std::cout << "  Initial alpha orbital energies (Ha):\n";
        for (int i = 0; i < std::min(10, static_cast<int>(nbasis_)); i++) {
            std::cout << "    " << i+1 << ": " << std::setw(12) << std::fixed 
                      << std::setprecision(6) << eps_alpha_(i);
            if (i < n_alpha_) std::cout << " (occ)";
            std::cout << "\n";
        }
    }
}

void ROHF::build_fock() {
    // ROHF using effective Fock operator approach
    // REFERENCE: McWeeny & Diercksen (1968), J. Chem. Phys. 49, 4852
    // REFERENCE: Szabo & Ostlund (1996), Section 3.8.7, p. 108-110
    // Build separate F^α, F^β then average for canonical orbitals
    
    Eigen::MatrixXd Pt = P_alpha_ + P_beta_;  // total density
    
    // Build F^α and F^β separately
    // F^α = H + J[P^tot] - K[P^α]
    // F^β = H + J[P^tot] - K[P^β]
    F_alpha_ = H_;
    F_beta_ = H_;
    
    for (size_t p = 0; p < nbasis_; p++) {
        for (size_t q = 0; q < nbasis_; q++) {
            double ga = 0.0, gb = 0.0;
            
            for (size_t r = 0; r < nbasis_; r++) {
                for (size_t s = 0; s < nbasis_; s++) {
                    double pqrs = compute_eri_element(p,q,r,s);
                    double prqs = compute_eri_element(p,r,q,s);
                    
                    // Coulomb from total density
                    double j_contrib = Pt(r,s) * pqrs;
                    ga += j_contrib;
                    gb += j_contrib;
                    
                    // Exchange from respective spin
                    ga -= P_alpha_(r,s) * prqs;
                    gb -= P_beta_(r,s) * prqs;
                }
            }
            
            F_alpha_(p,q) += ga;
            F_beta_(p,q) += gb;
        }
    }
}

double ROHF::compute_eri_element(size_t p, size_t q, size_t r, size_t s) {
    // Return precomputed (pq|rs) from ERI tensor
    return ERI_(p, q, r, s);
}

Eigen::MatrixXd ROHF::build_density(const Eigen::MatrixXd& C, int n_occ) {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.145), p. 139
    // Density: P_μν = Σ_i C_μi C_νi (sum over occupied)
    
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    for (int i = 0; i < n_occ; i++) {
        for (size_t mu = 0; mu < nbasis_; mu++) {
            for (size_t nu = 0; nu < nbasis_; nu++) {
                P(mu, nu) += C(mu, i) * C(nu, i);
            }
        }
    }
    
    return P;
}

double ROHF::compute_energy() {
    // REFERENCE: Szabo & Ostlund (1996), Eq. (3.184), p. 150
    // E_elec = (1/2) Σ_μν [P^α (H + F^α) + P^β (H + F^β)]
    
    double E_elec = 0.0;
    
    for (size_t mu = 0; mu < nbasis_; mu++) {
        for (size_t nu = 0; nu < nbasis_; nu++) {
            E_elec += P_alpha_(mu,nu) * (H_(mu,nu) + F_alpha_(mu,nu));
            E_elec += P_beta_(mu,nu) * (H_(mu,nu) + F_beta_(mu,nu));
        }
    }
    
    E_elec *= 0.5;
    
    return E_elec;
}

void ROHF::solve_fock(const Eigen::MatrixXd& F,
                      Eigen::MatrixXd& C,
                      Eigen::VectorXd& eps) {
    /**
     * Solve generalized eigenvalue problem: F C = S C ε
     * 
     * ALGORITHM:
     * 1. Compute X = S^{-1/2} (orthogonalization matrix)
     * 2. Transform Fock: F' = X^T F X
     * 3. Diagonalize: F' C' = C' ε
     * 4. Back-transform: C = X C'
     * 
     * REFERENCE:
     * Szabo & Ostlund (1996), Section 3.4.5, pp. 143-145
     */
    
    // Step 1: Compute S^{-1/2}
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s_solver(S_);
    Eigen::VectorXd s_vals = s_solver.eigenvalues();
    Eigen::MatrixXd s_vecs = s_solver.eigenvectors();
    
    // S^{-1/2} = U s^{-1/2} U^T
    Eigen::MatrixXd s_inv_sqrt = s_vecs * s_vals.cwiseSqrt().cwiseInverse().asDiagonal() * s_vecs.transpose();
    
    // Step 2: Transform Fock matrix
    Eigen::MatrixXd F_prime = s_inv_sqrt.transpose() * F * s_inv_sqrt;
    
    // Step 3: Diagonalize
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> f_solver(F_prime);
    eps = f_solver.eigenvalues();
    Eigen::MatrixXd C_prime = f_solver.eigenvectors();
    
    // Step 4: Back-transform
    C = s_inv_sqrt * C_prime;
}

bool ROHF::check_convergence() {
    /**
     * Check SCF convergence
     * 
     * Criteria:
     * 1. |E_new - E_old| < energy_threshold
     * 2. ||P_new - P_old|| < density_threshold
     */
    
    // Energy convergence
    double dE = std::abs(energy_ - energy_old_);
    
    // For first iteration, not converged
    if (energy_old_ == 0.0) {
        return false;
    }
    
    return (dE < config_.energy_threshold);
}

void ROHF::print_iteration(int iter, double dE, double dP) {
    if (config_.print_level == 0) return;
    
    std::cout << std::setw(6) << iter
              << std::setw(20) << std::fixed << std::setprecision(10) << energy_
              << std::setw(16) << std::scientific << std::setprecision(4) << dE
              << std::setw(16) << dP
              << "\n";
}

void ROHF::print_results(const SCFResult& result) {
    if (config_.print_level == 0) return;
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "       ROHF SCF RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "\nConvergence: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "\nEnergies (Hartree):\n";
    std::cout << "  Nuclear repulsion:  " << std::setw(16) << std::fixed 
              << std::setprecision(10) << result.energy_nuclear << "\n";
    std::cout << "  Electronic energy:  " << std::setw(16) << result.energy_electronic << "\n";
    std::cout << "  Total energy:       " << std::setw(16) << result.energy_total << "\n";
    
    std::cout << "\nOrbital energies (first 10):\n";
    std::cout << "  Alpha orbitals:\n";
    for (int i = 0; i < std::min(10, static_cast<int>(nbasis_)); i++) {
        std::cout << "    " << std::setw(3) << i+1 << ": " 
                  << std::setw(12) << std::setprecision(6) 
                  << result.orbital_energies_alpha(i);
        if (i < n_alpha_) std::cout << " (occ)";
        std::cout << "\n";
    }
    
    std::cout << "  Beta orbitals:\n";
    for (int i = 0; i < std::min(10, static_cast<int>(nbasis_)); i++) {
        std::cout << "    " << std::setw(3) << i+1 << ": " 
                  << std::setw(12) << std::setprecision(6) 
                  << result.orbital_energies_beta(i);
        if (i < n_beta_) std::cout << " (occ)";
        std::cout << "\n";
    }
    
    std::cout << "========================================\n\n";
}

SCFResult ROHF::run() {
    /**
     * Main SCF procedure
     */
    
    if (config_.print_level > 0) {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "     ROHF SCF CALCULATION\n";
        std::cout << "========================================\n";
        std::cout << "\nSystem:\n";
        std::cout << "  Atoms: " << mol_.n_atoms() << "\n";
        std::cout << "  Basis functions: " << nbasis_ << "\n";
        std::cout << "  Alpha electrons: " << n_alpha_ << "\n";
        std::cout << "  Beta electrons: " << n_beta_ << "\n";
        std::cout << "  Multiplicity: " << (n_alpha_ - n_beta_ + 1) << "\n";
        std::cout << "\nConvergence thresholds:\n";
        std::cout << "  Energy: " << std::scientific << config_.energy_threshold << "\n";
        std::cout << "  Density: " << config_.density_threshold << "\n";
        std::cout << "========================================\n";
    }
    
    // Initialize
    initialize_integrals();
    initial_guess();
    
    // Initialize DIIS
    DIIS diis_alpha(config_.diis_max_vectors);
    DIIS diis_beta(config_.diis_max_vectors);
    bool use_diis = (config_.diis_max_vectors > 0);
    
    // SCF header
    if (config_.print_level > 0) {
        std::cout << "\nSCF Iterations:\n";
        std::cout << std::setw(6) << "Iter"
                  << std::setw(20) << "Energy (Ha)"
                  << std::setw(16) << "ΔE"
                  << std::setw(16) << "||ΔP||"
                  << std::setw(10) << "DIIS"
                  << "\n";
        std::cout << std::string(68, '-') << "\n";
    }
    
    // SCF loop
    bool converged = false;
    int iter;
    
    for (iter = 1; iter <= config_.max_iterations; iter++) {
        // Save old energy
        energy_old_ = energy_;
        
        // Build Fock matrices
        build_fock();
        
        // Compute DIIS error vectors: e = F*P*S - S*P*F
        Eigen::MatrixXd error_alpha = F_alpha_ * P_alpha_ * S_ - S_ * P_alpha_ * F_alpha_;
        Eigen::MatrixXd error_beta = F_beta_ * P_beta_ * S_ - S_ * P_beta_ * F_beta_;
        double error_norm = std::max(error_alpha.norm(), error_beta.norm());
        
        // Apply DIIS extrapolation if enabled and error is small enough
        bool diis_applied = false;
        if (use_diis && error_norm < config_.diis_threshold) {
            diis_alpha.add_iteration(F_alpha_, error_alpha);
            diis_beta.add_iteration(F_beta_, error_beta);
            
            if (diis_alpha.can_extrapolate()) {
                F_alpha_ = diis_alpha.extrapolate();
                F_beta_ = diis_beta.extrapolate();
                diis_applied = true;
            }
        }
        
        // Average F^α and F^β for canonical orbitals
        // REFERENCE: McWeeny & Diercksen (1968), J. Chem. Phys. 49, 4852
        // F_eff = (n_α F^α + n_β F^β) / (n_α + n_β)
        double na = static_cast<double>(n_alpha_);
        double nb = static_cast<double>(n_beta_);
        Eigen::MatrixXd F_eff = (na * F_alpha_ + nb * F_beta_) / (na + nb);
        
        // Solve for canonical spatial orbitals
        solve_fock(F_eff, C_alpha_, eps_alpha_);
        C_beta_ = C_alpha_;
        eps_beta_ = eps_alpha_;
        
        // Update densities
        Eigen::MatrixXd P_alpha_old = P_alpha_;
        Eigen::MatrixXd P_beta_old = P_beta_;
        
        P_alpha_ = build_density(C_alpha_, n_alpha_);
        P_beta_ = build_density(C_beta_, n_beta_);
        
        // Compute energy
        energy_ = compute_energy();
        
        // Check convergence
        double dE = std::abs(energy_ - energy_old_);
        double dP_alpha = (P_alpha_ - P_alpha_old).norm();
        double dP_beta = (P_beta_ - P_beta_old).norm();
        double dP = std::max(dP_alpha, dP_beta);
        
        // Print iteration info
        if (config_.print_level > 0) {
            std::cout << std::setw(6) << iter
                      << std::setw(20) << std::fixed << std::setprecision(10) << energy_
                      << std::setw(16) << std::scientific << std::setprecision(4) << dE
                      << std::setw(16) << dP
                      << std::setw(10) << (diis_applied ? "YES" : "NO")
                      << "\n";
        }
        
        if (check_convergence() && dP < config_.density_threshold) {
            converged = true;
            break;
        }
    }
    
    // Prepare results
    SCFResult result;
    result.energy_nuclear = mol_.nuclear_repulsion_energy();
    result.energy_electronic = energy_;
    result.energy_total = energy_ + result.energy_nuclear;
    result.orbital_energies_alpha = eps_alpha_;
    result.orbital_energies_beta = eps_beta_;
    result.C_alpha = C_alpha_;
    result.C_beta = C_beta_;
    result.P_alpha = P_alpha_;
    result.P_beta = P_beta_;
    result.F_alpha = F_alpha_;  // Store Fock for MP2
    result.F_beta = F_beta_;
    result.iterations = iter;
    result.converged = converged;
    result.gradient_norm = 0.0;  // TODO: Implement gradient
    result.n_occ_alpha = n_alpha_;
    result.n_occ_beta = n_beta_;
    
    // Print results
    print_results(result);
    
    return result;
}

} // namespace mshqc