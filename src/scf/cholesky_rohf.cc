/**
 * @file cholesky_rohf.cc
 * @brief Implementation of Cholesky ROHF
 * @details FIXED: Print Total Energy (Elec + Nuc) during iterations to avoid confusion.
 */

#include "mshqc/cholesky_rohf.h"
#include "mshqc/scf.h" // Pastikan DIIS class definition terlihat jika ada di header terpisah
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace mshqc {

// ============================================================================
// CONSTRUCTOR 1: FROM SCRATCH (DECOMPOSE)
// ============================================================================
CholeskyROHF::CholeskyROHF(const Molecule& mol, const BasisSet& basis,
                           std::shared_ptr<IntegralEngine> integrals,
                           int n_alpha, int n_beta,
                           const CholeskyROHFConfig& config)
    : mol_(mol), basis_(basis), integrals_(integrals),
      config_(config), n_alpha_(n_alpha), n_beta_(n_beta), own_cholesky_(true)
{
    nbasis_ = basis.n_basis_functions();
    // Inisialisasi object Cholesky kosong, nanti didecompose di init_integrals
    cholesky_ = std::make_unique<integrals::CholeskyERI>(config_.cholesky_threshold);
}

// ============================================================================
// CONSTRUCTOR 2: REUSE VECTORS (EFFICIENT)
// ============================================================================
CholeskyROHF::CholeskyROHF(const Molecule& mol, const BasisSet& basis,
                           std::shared_ptr<IntegralEngine> integrals,
                           int n_alpha, int n_beta,
                           const CholeskyROHFConfig& config,
                           const integrals::CholeskyERI& existing_cholesky)
    : mol_(mol), basis_(basis), integrals_(integrals),
      config_(config), n_alpha_(n_alpha), n_beta_(n_beta), own_cholesky_(true)
{
    nbasis_ = basis.n_basis_functions();
    
    if (config_.print_level > 0) {
        std::cout << "\n[CholeskyROHF] Initializing with REUSED vectors.\n";
        std::cout << "  Vectors provided: " << existing_cholesky.n_vectors() << "\n";
    }

    // Copy objek cholesky (shallow copy vektor internal std::vector aman)
    cholesky_ = std::make_unique<integrals::CholeskyERI>(existing_cholesky);
}

// ============================================================================
// INITIALIZATION
// ============================================================================
void CholeskyROHF::init_integrals() {
    // 1. One-Electron Integrals
    S_ = integrals_->compute_overlap();
    H_ = integrals_->compute_core_hamiltonian();
    
    // 2. Cholesky Decomposition (If not reused)
    if (cholesky_->n_vectors() == 0) {
        if (config_.print_level > 0) std::cout << "[CholeskyROHF] Decomposing Integrals...\n";
        auto eri_full = integrals_->compute_eri();
        cholesky_->decompose(eri_full);
    }
    
    // 3. Orthogonalizer (S^-1/2)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    X_ = es.eigenvectors() * es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
    
    // Resize matrices
    F_alpha_.resize(nbasis_, nbasis_);
    F_beta_.resize(nbasis_, nbasis_);
}

void CholeskyROHF::initial_guess() {
    // Core Hamiltonian Guess
    solve_fock(H_, C_alpha_, eps_alpha_);
    solve_fock(H_, C_beta_, eps_beta_);
    
    P_alpha_ = build_density(C_alpha_, n_alpha_);
    P_beta_  = build_density(C_beta_, n_beta_);
}

// ============================================================================
// FOCK BUILD (THE KEY PART)
// ============================================================================
void CholeskyROHF::build_fock_cholesky() {
    // Reset Fock matrices ke Core Hamiltonian
    F_alpha_ = H_;
    F_beta_  = H_;
    
    // Total Density
    Eigen::MatrixXd P_tot = P_alpha_ + P_beta_;
    
    const auto& L_vecs = cholesky_->get_L_vectors();
    int n_chol = L_vecs.size();
    
    // Ambil koefisien occupied untuk Exchange
    Eigen::MatrixXd Ca_occ = C_alpha_.leftCols(n_alpha_);
    Eigen::MatrixXd Cb_occ = C_beta_.leftCols(n_beta_);

    // Parallel Loop over Cholesky Vectors
    
    Eigen::MatrixXd J_tot = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    Eigen::MatrixXd K_a   = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    Eigen::MatrixXd K_b   = Eigen::MatrixXd::Zero(nbasis_, nbasis_);

    #pragma omp parallel
    {
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Ka_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd Kb_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        
        #pragma omp for schedule(dynamic)
        for (int K = 0; K < n_chol; ++K) {
            // 1. Map Vector ke Matrix (Tanpa Copy!)
            Eigen::Map<const Eigen::MatrixXd> L(L_vecs[K].data(), nbasis_, nbasis_);
            
            // COULOMB (J)
            double scalar_J = (L.cwiseProduct(P_tot)).sum();
            J_local += scalar_J * L;
            
            // EXCHANGE (K) Alpha
            Eigen::MatrixXd La_occ = L * Ca_occ; 
            Ka_local += La_occ * La_occ.transpose();
            
            // EXCHANGE (K) Beta
            if (n_beta_ > 0) {
                Eigen::MatrixXd Lb_occ = L * Cb_occ;
                Kb_local += Lb_occ * Lb_occ.transpose();
            }
        }
        
        #pragma omp critical
        {
            J_tot += J_local;
            K_a   += Ka_local;
            K_b   += Kb_local;
        }
    }
    
    // Final Assembly
    F_alpha_ += J_tot - K_a;
    F_beta_  += J_tot - K_b;
}

// ============================================================================
// STANDARD SCF ROUTINES
// ============================================================================

Eigen::MatrixXd CholeskyROHF::build_density(const Eigen::MatrixXd& C, int n_occ) {
    if (n_occ == 0) return Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    return C_occ * C_occ.transpose();
}

double CholeskyROHF::compute_energy() {
    double E = 0.0;
    // E = 0.5 * Tr[ P_a(H + F_a) + P_b(H + F_b) ]
    E += 0.5 * (P_alpha_.cwiseProduct(H_ + F_alpha_)).sum();
    E += 0.5 * (P_beta_.cwiseProduct(H_ + F_beta_)).sum();
    return E;
}

void CholeskyROHF::solve_fock(const Eigen::MatrixXd& F, Eigen::MatrixXd& C, Eigen::VectorXd& eps) {
    // F' = X^T F X
    Eigen::MatrixXd F_prime = X_.transpose() * F * X_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_prime);
    eps = es.eigenvalues();
    C = X_ * es.eigenvectors();
}

bool CholeskyROHF::check_convergence() {
    double dE = std::abs(energy_ - energy_old_);
    return (dE < config_.energy_threshold);
}

// ============================================================================
// MAIN COMPUTE
// ============================================================================

SCFResult CholeskyROHF::compute() {
    if (config_.print_level > 0) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "  Cholesky-ROHF SCF Calculation\n";
        std::cout << "  Vectors: " << (own_cholesky_ ? "Internal/Reused" : "External") << "\n";
        std::cout << std::string(60, '=') << "\n";
    }
    
    init_integrals();
    initial_guess();
    
    DIIS diis_a(config_.diis_max_vectors);
    DIIS diis_b(config_.diis_max_vectors);
    
    // [FIX] Pre-compute Nuclear Repulsion untuk display yang konsisten
    double Enuc = mol_.nuclear_repulsion_energy();

    int iter = 0;
    bool converged = false;
    
    // SCF LOOP
    while (iter < config_.max_iterations) {
        iter++;
        energy_old_ = energy_;
        
        // 1. Build Fock (Cholesky Optimized)
        build_fock_cholesky();
        
        // 2. Compute Energy (Electronic)
        energy_ = compute_energy();
        
        // 3. DIIS
        Eigen::MatrixXd err_a = F_alpha_ * P_alpha_ * S_ - S_ * P_alpha_ * F_alpha_;
        Eigen::MatrixXd err_b = F_beta_ * P_beta_ * S_ - S_ * P_beta_ * F_beta_;
        
        if (err_a.norm() < config_.diis_threshold) {
            diis_a.add_iteration(F_alpha_, err_a);
            if (diis_a.can_extrapolate()) F_alpha_ = diis_a.extrapolate();
        }
        if (err_b.norm() < config_.diis_threshold) {
            diis_b.add_iteration(F_beta_, err_b);
            if (diis_b.can_extrapolate()) F_beta_ = diis_b.extrapolate();
        }
        
        // 4. ROHF Averaging for Canonical Orbitals
        // F_eff = (Na * Fa + Nb * Fb) / (Na + Nb)
        double Na = static_cast<double>(n_alpha_);
        double Nb = static_cast<double>(n_beta_);
        Eigen::MatrixXd F_eff = (Na * F_alpha_ + Nb * F_beta_) / (Na + Nb);
        
        // 5. Diagonalize
        solve_fock(F_eff, C_alpha_, eps_alpha_);
        C_beta_ = C_alpha_;     // ROHF shares spatial orbitals
        eps_beta_ = eps_alpha_;
        
        // 6. Update Density
        P_alpha_ = build_density(C_alpha_, n_alpha_);
        P_beta_  = build_density(C_beta_, n_beta_);
        
        // 7. Print & Check
        double dE = energy_ - energy_old_;
        if (config_.print_level > 0) {
            // [FIX] Tampilkan Energy Total (Elec + Nuc) agar user tidak bingung
            double E_print = energy_ + Enuc;
            
            std::cout << "Iter " << std::setw(3) << iter 
                      << " E = " << std::fixed << std::setprecision(10) << E_print
                      << " dE = " << std::scientific << std::setprecision(4) << dE << "\n";
        }
        
        if (check_convergence() && iter > 1) {
            converged = true;
            break;
        }
    }
    
    // Result
    SCFResult res;
    res.energy_nuclear = Enuc;
    res.energy_electronic = energy_;
    res.energy_total = energy_ + Enuc; // Konsisten dengan yang dicetak
    res.C_alpha = C_alpha_;
    res.C_beta = C_beta_;
    res.P_alpha = P_alpha_;
    res.P_beta = P_beta_;
    res.orbital_energies_alpha = eps_alpha_;
    res.orbital_energies_beta = eps_beta_;
    res.converged = converged;
    res.iterations = iter;
    res.n_occ_alpha = n_alpha_;
    res.n_occ_beta = n_beta_;
    
    return res;
}

} // namespace mshqc