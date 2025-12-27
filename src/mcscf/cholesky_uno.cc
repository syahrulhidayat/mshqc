/**
 * @file src/mcscf/cholesky_uno.cc
 * @brief Implementation of Cholesky-UNO Generator (FIXED ORTHOGONALITY)
 * * Corrections:
 * 1. Uses S^(-1/2) for back-transformation to ensure C^T * S * C = I
 * 2. Uses S^(1/2) for Density matrix transformation into orthogonal basis.
 */

#include "mshqc/mcscf/cholesky_uno.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>

namespace mshqc {
namespace mcscf {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

CholeskyUNO::CholeskyUNO(const SCFResult& uhf_result,
                         std::shared_ptr<IntegralEngine> integrals,
                         int n_basis)
    : uhf_res_(uhf_result), integrals_(integrals), nbasis_(n_basis)
{
}

// ============================================================================
// ENTROPY HELPER
// ============================================================================
double CholeskyUNO::calculate_entropy(const Eigen::VectorXd& n) const {
    double S = 0.0;
    for (int i = 0; i < n.size(); ++i) {
        double ni = n(i);
        // Normalized probability p = ni / 2.0
        double p = ni / 2.0;
        
        // Handle numerical limits (0 log 0 = 0)
        if (p > 1e-12 && p < (1.0 - 1e-12)) {
            S -= p * std::log(p) + (1.0 - p) * std::log(1.0 - p);
        }
    }
    return S;
}

// ============================================================================
// MAIN COMPUTE FUNCTION
// ============================================================================



UNOResult CholeskyUNO::compute() {
    // 1. Get Overlap Matrix (S) & Kinetic + Nuclear (H)
    Eigen::MatrixXd S = integrals_->compute_overlap();
    Eigen::MatrixXd H = integrals_->compute_kinetic() + integrals_->compute_nuclear(); // [NEW] Needed for energy sorting
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_S(S);
    Eigen::MatrixXd X_inv = es_S.operatorInverseSqrt(); 
    Eigen::MatrixXd X_reg = es_S.operatorSqrt();

    // 2. Form Total Density Matrix
    Eigen::MatrixXd P_tot = uhf_res_.P_alpha + uhf_res_.P_beta;

    // 3. Transform Density to Orthogonal Basis
    Eigen::MatrixXd P_ortho = X_reg * P_tot * X_reg;
    
    // 4. Diagonalize P to get Natural Orbitals
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_P(P_ortho);
    Eigen::VectorXd occ_raw = es_P.eigenvalues();      
    Eigen::MatrixXd C_ortho = es_P.eigenvectors();     

    // --- [FIX START] PRE-CALCULATE ENERGIES FOR SORTING ---
    // Kita hitung ekspektasi energi 1-elektron: E_i = <psi_i | H_core | psi_i>
    // Ini cukup akurat untuk membedakan 2p vs 3s di virtual space.
    
    // Transform C_ortho back to AO first for energy calculation
    Eigen::MatrixXd C_temp = X_inv * C_ortho; // Coefficients in AO basis
    Eigen::VectorXd orb_energies(nbasis_);
    
    for(int i=0; i<nbasis_; ++i) {
        // E = C_i^T * H_core * C_i
        orb_energies(i) = C_temp.col(i).dot(H * C_temp.col(i));
    }
    // --- [FIX END] ---

    // 5. Sort Orbitals (Modified Logic)
    std::vector<int> idx(nbasis_);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        double diff = std::abs(occ_raw(i) - occ_raw(j));
        
        // Jika perbedaan okupansi signifikan, prioritas okupansi tinggi (Core/Active)
        if (diff > 1e-4) {
            return occ_raw(i) > occ_raw(j);
        }
        
        // Jika okupansi sama (Virtual vs Virtual), prioritas ENERGI RENDAH (2p < 3s)
        return orb_energies(i) < orb_energies(j);
    });

    // 6. Back-transform Coefficients to AO Basis (Sorted)
    result_.C_uno = Eigen::MatrixXd(nbasis_, nbasis_);
    result_.occupations = Eigen::VectorXd(nbasis_);

    for (int i = 0; i < nbasis_; ++i) {
        int sorted_i = idx[i];
        result_.occupations(i) = occ_raw(sorted_i);
        result_.C_uno.col(i) = C_temp.col(sorted_i); // Use pre-calculated C_temp
    }

    // 7. Calculate Entropy & Analyze
    result_.entropy = calculate_entropy(result_.occupations);
    analyze_active_space(0.02);

    computed_ = true;
    return result_;
}

// ============================================================================
// ACTIVE SPACE ANALYSIS
// ============================================================================
void CholeskyUNO::analyze_active_space(double threshold) {
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
void CholeskyUNO::print_report(double threshold) const {
    if (!computed_) {
        std::cout << "  [UNO] Error: Run compute() first.\n";
        return;
    }

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "  UHF NATURAL ORBITALS (UNO) ANALYSIS\n";
    std::cout << std::string(65, '=') << "\n";
    std::cout << "  * Threshold for Active Space: " << std::fixed << std::setprecision(4) << threshold << " - " << (2.0 - threshold) << "\n";
    std::cout << "  * von Neumann Entropy (S):    " << std::fixed << std::setprecision(6) << result_.entropy << "\n";
    
    if (result_.entropy < 0.1) {
        std::cout << "    (Low S: System is likely Single-Reference dominated)\n";
    } else {
        std::cout << "    (High S: System has Multi-Reference/Open-Shell character)\n";
    }
    std::cout << "\n";

    std::cout << "  Orbital   Occupation    Classification    Character\n";
    std::cout << "  -------   ----------    --------------    ---------\n";

    for (int i = 0; i < nbasis_; ++i) {
        double n = result_.occupations(i);
        std::string type;
        std::string character;

        if (n >= (2.0 - threshold)) {
            type = "Inactive";
            character = "Core/Closed";
        } else if (n <= threshold) {
            type = "Secondary";
            character = "Virtual";
        } else {
            type = "ACTIVE";
            if (n > 1.5) character = "Hole-like";
            else if (n < 0.5) character = "Particle-like";
            else character = "Biradical/Open";
        }

        // Smart printing to avoid flooding terminal
        bool print_it = false;
        if (type == "ACTIVE") print_it = true;
        else if (i < 5) print_it = true; 
        else if (i >= nbasis_ - 5) print_it = true; 
        
        if (!result_.active_indices.empty()) {
            int first = result_.active_indices.front();
            int last = result_.active_indices.back();
            if (i >= first - 2 && i <= last + 2) print_it = true;
        }

        if (print_it) {
            std::cout << "  " << std::setw(5) << i 
                      << "   " << std::fixed << std::setprecision(6) << n 
                      << "    " << std::setw(14) << type
                      << "    " << character << "\n";
        } else if (i == 6 && nbasis_ > 20) {
             if (!result_.active_indices.empty() && i < result_.active_indices.front())
                std::cout << "   ...       ...           ...\n";
        }
    }

    std::cout << "\n  SUGGESTED ACTIVE SPACE: CAS(" 
              << result_.suggested_n_electrons << ", " 
              << result_.suggested_n_active << ")\n";
    
    std::cout << "  Indices: { ";
    for (int idx : result_.active_indices) std::cout << idx << " ";
    std::cout << "}\n";
    std::cout << std::string(65, '=') << "\n\n";
}

// ============================================================================
// SAVE ORBITALS
// ============================================================================
void CholeskyUNO::save_orbitals(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    out << "# UNO Orbitals Generated by mshqc\n";
    out << "# NBasis: " << nbasis_ << "\n";
    for (int i = 0; i < nbasis_; ++i) {
        out << i << " " << std::fixed << std::setprecision(6) << result_.occupations(i) << " ";
        for (int j = 0; j < nbasis_; ++j) {
            out << std::scientific << std::setprecision(8) << result_.C_uno(j, i) << " ";
        }
        out << "\n";
    }
    out.close();
    std::cout << "  [UNO] Orbitals saved to " << filename << "\n";
}

} // namespace mcscf
} // namespace mshqc