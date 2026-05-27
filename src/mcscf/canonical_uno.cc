/**
 * @file src/mcscf/canonical_uno.cc
 * @brief Implementation of Canonical UNO (Robust for Linear Dependence)
 * Replaces Cholesky with S eigenvalue decomposition.
 */

#include "mshqc/mcscf/canonical_uno.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include <omp.h> // <--- WAJIB ADA UNTUK OPENMP
#ifdef I
#undef I
#endif

// Pastikan I tidak terdefinisi (konflik dengan complex.h kadang terjadi)
namespace mshqc {
namespace mcscf {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

CanonicalUNO::CanonicalUNO(const SCFResult& uhf_result,
                           std::shared_ptr<IntegralEngine> integrals,
                           int n_basis)
    : uhf_res_(uhf_result), integrals_(integrals), nbasis_(n_basis)
{
}

// ============================================================================
// ENTROPY HELPER
// ============================================================================
double CanonicalUNO::calculate_entropy(const Eigen::VectorXd& n) const {
    double S = 0.0;
    for (int i = 0; i < n.size(); ++i) {
        double ni = n(i);
        double p = ni / 2.0; // Normalized probability
        
        if (p > 1e-12 && p < (1.0 - 1e-12)) {
            S -= p * std::log(p) + (1.0 - p) * std::log(1.0 - p);
        }
    }
    return S;
}

// ============================================================================
// MAIN COMPUTE FUNCTION
// ============================================================================

UNOResult CanonicalUNO::compute() {
    // 1. Compute Integrals
    Eigen::MatrixXd S = integrals_->compute_overlap();
    // [Energy Sorting] H_core needed to distinguish degenerate virtuals
    Eigen::MatrixXd H = integrals_->compute_kinetic() + integrals_->compute_nuclear();

    // 2. Canonical Orthogonalization (Robust handling of S)
    //    Decompose S = U * s * U^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_S(S);
    Eigen::VectorXd s_vals = es_S.eigenvalues();
    Eigen::MatrixXd U = es_S.eigenvectors();

    //    Build X = S^(-1/2) and Y = S^(1/2)
    //    Filter eigenvalues to handle linear dependence (e.g., 5Z basis)
    double threshold = 1.0e-6; 
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(nbasis_, nbasis_); // Transform to Ortho
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(nbasis_, nbasis_); // Transform Density

    int n_kept = 0;

    // --- OPTIMIZED PARALLEL CONSTRUCTION ---
    // Siapkan container thread-local untuk menghindari race condition pada X += ...
    int n_threads = omp_get_max_threads();
    std::vector<Eigen::MatrixXd> X_priv(n_threads, Eigen::MatrixXd::Zero(nbasis_, nbasis_));
    std::vector<Eigen::MatrixXd> Y_priv(n_threads, Eigen::MatrixXd::Zero(nbasis_, nbasis_));

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int local_kept = 0;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < nbasis_; i++) {
            if (s_vals(i) >= threshold) {
                local_kept++;
                double val_inv_sqrt = 1.0 / std::sqrt(s_vals(i));
                double val_sqrt = std::sqrt(s_vals(i));
                
                // Outer product u * u^T
                Eigen::MatrixXd UiUiT = U.col(i) * U.col(i).transpose();
                
                X_priv[tid] += UiUiT * val_inv_sqrt;
                Y_priv[tid] += UiUiT * val_sqrt;
            }
        }
        
        #pragma omp atomic
        n_kept += local_kept;
    }

    // Reduksi (Gabungkan hasil thread ke matriks utama)
    for (int t = 0; t < n_threads; ++t) {
        X += X_priv[t];
        Y += Y_priv[t];
    }
    // ---------------------------------------

    if (n_kept < nbasis_) {
        std::cout << "  [UNO WARNING] Linear dependence in basis! Removed " 
                  << (nbasis_ - n_kept) << " vectors.\n";
    }

    // 3. Form Total Density Matrix (AO Basis)
    Eigen::MatrixXd P_tot = uhf_res_.P_alpha + uhf_res_.P_beta;

    // 4. Transform Density to Orthogonal Basis
    //    P_ortho = S^(1/2) * P_AO * S^(1/2)
    //    Using Y constructed above which represents S^(1/2) projected on valid space
    Eigen::MatrixXd P_ortho = Y * P_tot * Y;

    // 5. Diagonalize P_ortho to get Natural Orbitals
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_P(P_ortho);
    Eigen::VectorXd occ_raw = es_P.eigenvalues();      
    Eigen::MatrixXd C_ortho = es_P.eigenvectors();     

    // 6. Back-transform Coefficients to AO Basis
    //    C_AO = S^(-1/2) * C_ortho = X * C_ortho
    Eigen::MatrixXd C_temp = X * C_ortho;

    // --- [ENERGY SORTING LOGIC] ---
    // Calculate orbital energies for secondary sorting criteria
    Eigen::VectorXd orb_energies(nbasis_);
    
    // Paralelisasi perhitungan energi orbital
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<nbasis_; ++i) {
        // E_i = <psi_i | H_core | psi_i>
        // C_temp is already in AO basis
        orb_energies(i) = C_temp.col(i).dot(H * C_temp.col(i));
    }

    // 7. Sort Indices
    std::vector<int> idx(nbasis_);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        double diff = std::abs(occ_raw(i) - occ_raw(j));
        
        // Primary: Occupation Number (descending)
        if (diff > 1.0e-4) {
            return occ_raw(i) > occ_raw(j);
        }
        
        // Secondary: Orbital Energy (ascending) for degenerate occupations
        // Important for mixing Virtuals (2p vs 3s) correctly
        return orb_energies(i) < orb_energies(j);
    });

    // 8. Store Final Results
    result_.C_uno = Eigen::MatrixXd(nbasis_, nbasis_);
    result_.occupations = Eigen::VectorXd(nbasis_);

    for (int i = 0; i < nbasis_; ++i) {
        int sorted_i = idx[i];
        result_.occupations(i) = occ_raw(sorted_i);
        result_.C_uno.col(i) = C_temp.col(sorted_i);
    }

    // 9. Analysis
    result_.entropy = calculate_entropy(result_.occupations);
    analyze_active_space(0.02);

    computed_ = true;
    return result_;
}

// ============================================================================
// ACTIVE SPACE ANALYSIS
// ============================================================================
void CanonicalUNO::analyze_active_space(double threshold) {
    result_.active_indices.clear();
    double n_elec_active_sum = 0.0;

    for (int i = 0; i < nbasis_; ++i) {
        double n = result_.occupations(i);

        // Occupations between threshold and 2.0-threshold are active
        if (n > threshold && n < (2.0 - threshold)) {
            result_.active_indices.push_back(i);
            n_elec_active_sum += n;
        }
    }

    result_.suggested_n_active = static_cast<int>(result_.active_indices.size());
    result_.suggested_n_electrons = static_cast<int>(std::round(n_elec_active_sum));
}

// ============================================================================
// PRINT REPORT
// ============================================================================
void CanonicalUNO::print_report(double threshold) const {
    if (!computed_) {
        std::cout << "  [UNO] Error: Run compute() first.\n";
        return;
    }

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "  CANONICAL UNO ANALYSIS (Robust Orthogonalization)\n";
    std::cout << std::string(65, '=') << "\n";
    std::cout << "  * Threshold for Active Space: " << std::fixed << std::setprecision(4) << threshold << " - " << (2.0 - threshold) << "\n";
    std::cout << "  * von Neumann Entropy (S):    " << std::fixed << std::setprecision(6) << result_.entropy << "\n";
    
    std::cout << "  Indices: { ";
    for (int idx : result_.active_indices) std::cout << idx << " ";
    std::cout << "}\n";
    std::cout << "  Suggested CAS(" << result_.suggested_n_electrons 
              << ", " << result_.suggested_n_active << ")\n";
    std::cout << std::string(65, '=') << "\n\n";
}

// ============================================================================
// SAVE ORBITALS
// ============================================================================
void CanonicalUNO::save_orbitals(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    out << "# Canonical UNO Orbitals Generated by mshqc\n";
    out << "# NBasis: " << nbasis_ << "\n";
    for (int i = 0; i < nbasis_; ++i) {
        out << i << " " << std::fixed << std::setprecision(6) << result_.occupations(i) << " ";
        for (int j = 0; j < nbasis_; ++j) {
            out << std::scientific << std::setprecision(8) << result_.C_uno(j, i) << " ";
        }
        out << "\n";
    }
    out.close();
}

} // namespace mcscf
} // namespace mshqc