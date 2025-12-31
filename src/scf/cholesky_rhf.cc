/**
 * @file src/scf/cholesky_rhf.cc
 * @brief Cholesky-Decomposed Restricted Hartree-Fock (RHF) Implementation
 * @details Added Density Damping to prevent SCF oscillation in small-gap systems.
 */

#include "mshqc/cholesky_rhf.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

// ============================================================================
// CONSTRUCTOR 1: STANDARD
// ============================================================================
CholeskyRHF::CholeskyRHF(const Molecule& mol,
                         const BasisSet& basis,
                         std::shared_ptr<IntegralEngine> integrals,
                         const CholeskyRHFConfig& config) 
    : mol_(mol), basis_(basis), integrals_(integrals), config_(config),
      diis_(config.diis_max_vectors) 
{
    nbasis_ = basis.n_basis_functions();
    int n_electrons = mol.n_electrons();
    
    if (n_electrons % 2 != 0) {
        throw std::runtime_error("CholeskyRHF: Molecule must be closed-shell (even electrons).");
    }
    n_occ_ = n_electrons / 2;

    if (config_.print_level > 0) {
        std::cout << "\n[CholeskyRHF] Performing internal decomposition...\n";
    }
    internal_cholesky_ = std::make_unique<integrals::CholeskyERI>(config_.cholesky_threshold);
    auto eri_tensor = integrals_->compute_eri();
    internal_cholesky_->decompose(eri_tensor);
    cholesky_ptr_ = internal_cholesky_.get();
}

// ============================================================================
// CONSTRUCTOR 2: REUSE VECTORS
// ============================================================================
CholeskyRHF::CholeskyRHF(const Molecule& mol,
                         const BasisSet& basis,
                         std::shared_ptr<IntegralEngine> integrals,
                         const CholeskyRHFConfig& config, 
                         const integrals::CholeskyERI& existing_cholesky)
    : mol_(mol), basis_(basis), integrals_(integrals), config_(config),
      cholesky_ptr_(&existing_cholesky), diis_(config.diis_max_vectors)
{
    nbasis_ = basis.n_basis_functions();
    int n_electrons = mol.n_electrons();

    if (n_electrons % 2 != 0) {
        throw std::runtime_error("CholeskyRHF: Molecule must be closed-shell (even electrons).");
    }
    n_occ_ = n_electrons / 2;

    if (config_.print_level > 0) {
        std::cout << "\n[CholeskyRHF] Initializing with REUSED vectors.\n";
        std::cout << "  Vectors provided: " << cholesky_ptr_->n_vectors() << "\n";
    }
}

// ============================================================================
// INITIALIZATION & GUESS
// ============================================================================
void CholeskyRHF::init_integrals() {
    S_ = integrals_->compute_overlap();
    Eigen::MatrixXd T = integrals_->compute_kinetic();
    Eigen::MatrixXd V = integrals_->compute_nuclear();
    H_ = T + V;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_);
    Eigen::VectorXd s_inv_sqrt = es.eigenvalues().array().rsqrt();
    X_ = es.eigenvectors() * s_inv_sqrt.asDiagonal() * es.eigenvectors().transpose();
}

void CholeskyRHF::initial_guess() {
    Eigen::MatrixXd H_prime = X_.transpose() * H_ * X_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_prime);
    
    eps_ = es.eigenvalues();
    C_ = X_ * es.eigenvectors(); 
    P_ = build_density();        
    energy_ = 0.0;
}

Eigen::MatrixXd CholeskyRHF::build_density() {
    Eigen::MatrixXd C_occ = C_.leftCols(n_occ_);
    Eigen::MatrixXd P = 2.0 * (C_occ * C_occ.transpose());
    return P;
}

// ============================================================================
// CHOLESKY FOCK BUILD
// ============================================================================
void CholeskyRHF::build_fock() {
    F_ = H_; 
    
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
    
    const auto& L_vecs = cholesky_ptr_->get_L_vectors();
    int n_vec = L_vecs.size();

    #pragma omp parallel
    {
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        Eigen::MatrixXd K_local = Eigen::MatrixXd::Zero(nbasis_, nbasis_);
        
        #pragma omp for schedule(dynamic)
        for (int P = 0; P < n_vec; ++P) {
            Eigen::Map<const Eigen::MatrixXd> L_P(L_vecs[P].data(), nbasis_, nbasis_);
            
            double trace_LP = (L_P.cwiseProduct(P_)).sum();
            J_local += trace_LP * L_P;
            
            if (config_.screen_exchange) {
                if (L_P.norm() > 1e-8) {
                    K_local += L_P * P_ * L_P;
                }
            } else {
                K_local += L_P * P_ * L_P;
            }
        }
        
        #pragma omp critical
        {
            J += J_local;
            K += K_local;
        }
    }

    F_ += J - 0.5 * K;
}

// ============================================================================
// SCF UTILITIES
// ============================================================================
double CholeskyRHF::compute_energy() {
    return 0.5 * (P_ * (H_ + F_)).trace();
}

void CholeskyRHF::solve_fock() {
    Eigen::MatrixXd F_prime = X_.transpose() * F_ * X_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_prime);
    
    eps_ = es.eigenvalues();
    C_ = X_ * es.eigenvectors(); 
}

// ============================================================================
// MAIN COMPUTE LOOP
// ============================================================================
SCFResult CholeskyRHF::compute() {
    init_integrals();
    initial_guess();
    
    double e_nuc = mol_.nuclear_repulsion_energy();

    if (config_.print_level > 0) {
        std::cout << "\n============================================================\n";
        std::cout << "  Cholesky-RHF SCF Calculation\n";
        std::cout << "  Vectors: " << (internal_cholesky_ ? "Internal" : "Reused") << "\n";
        std::cout << "  Nuclear Repulsion: " << std::fixed << std::setprecision(6) << e_nuc << " Ha\n";
        std::cout << "============================================================\n";
    }

    bool converged = false;
    int iter = 0;
    double energy_old = 0.0;
    energy_ = 0.0;
    
    // [BARU] Simpan densitas lama untuk Damping
    Eigen::MatrixXd P_prev = P_;

    for (iter = 1; iter <= config_.max_iterations; ++iter) {
        energy_old = energy_;
        
        // Simpan P sebelum update untuk damping di iterasi berikutnya (jika perlu)
        P_prev = P_; 
        
        build_fock();
        
        // DIIS Extrapolation
        if (iter > 1 && diis_.can_extrapolate()) {
            Eigen::MatrixXd error = F_ * P_ * S_ - S_ * P_ * F_;
            if (error.norm() < config_.diis_threshold) {
                diis_.add_iteration(F_, error);
                if (diis_.can_extrapolate()) {
                    F_ = diis_.extrapolate();
                }
            }
        }

        solve_fock();
        
        // Hitung Densitas Baru
        Eigen::MatrixXd P_new = build_density();
        
        // [LOGIKA DAMPING]
        // Campur densitas baru dengan densitas lama untuk mencegah osilasi.
        // alpha = 0.7 artinya 70% baru, 30% lama.
        double alpha = 0.7; 
        
        // Hanya lakukan damping jika DIIS belum aktif sepenuhnya atau osilasi terdeteksi
        // Untuk amannya, kita terapkan selalu di awal, atau jika DIIS belum 'matang'.
        P_ = alpha * P_new + (1.0 - alpha) * P_prev;
        
        // Hitung Energi Total
        double e_elec = compute_energy();
        energy_ = e_elec + e_nuc;

        double de = 0.0;
        if (iter > 1) {
            de = energy_ - energy_old;
        } else {
            de = energy_;
        }
        
        if (config_.print_level > 0) {
            std::cout << "Iter " << std::setw(3) << iter 
                      << " E(Tot) = " << std::fixed << std::setprecision(10) << energy_ 
                      << " dE = " << std::scientific << std::setprecision(4) << de << "\n";
        }

        if (std::abs(de) < config_.energy_threshold && iter > 1) {
            converged = true;
            break;
        }
    }

    SCFResult res;
    res.energy_total = energy_;
    res.energy_nuclear = e_nuc;
    res.energy_electronic = energy_ - e_nuc;
    
    res.converged = converged;
    res.iterations = iter;
    res.C_alpha = C_; 
    res.C_beta = C_; 
    res.P_alpha = 0.5 * P_; 
    res.P_beta = 0.5 * P_;
    res.F_alpha = F_;
    res.F_beta = F_;
    res.orbital_energies_alpha = eps_;
    res.orbital_energies_beta = eps_;
    res.n_occ_alpha = n_occ_;
    res.n_occ_beta = n_occ_;

    return res;
}

} // namespace mshqc