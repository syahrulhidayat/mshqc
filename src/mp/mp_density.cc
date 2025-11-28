/**
 * @file mp_density.cc
 * @brief Implementation of density matrix analysis for MP wavefunctions
 * 
 * Theory References:
 * - 1-RDM from PT: R. J. Bartlett, Ann. Rev. Phys. Chem. **32**, 359 (1981)
 * - Natural Orbitals: P.-O. Löwdin, Phys. Rev. **97**, 1474 (1955)
 * - OPDM Contractions: T. Helgaker et al., "Molecular Electronic Structure Theory" (2000)
 * - Diagnostics: T. J. Lee & P. R. Taylor, Int. J. Quantum Chem. Symp. **23**, 199 (1989)
 * - Coupled-Cluster Theory: R. J. Bartlett & M. Musiał, Rev. Mod. Phys. **79**, 291 (2007)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * 
 * @note Original implementation from theory papers. This module computes the one-particle
 *       density matrix (1-RDM) for Møller-Plesset wavefunctions from cluster amplitudes.
 *       The 1-RDM is then diagonalized to obtain natural orbitals and correlation measures.
 * 
 * @copyright MIT License
 */

#include "mshqc/mp/mp_density.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace mshqc {
namespace mp {

// ============================================================================
// Helper: Compute HF contribution to 1-RDM
// ============================================================================
Eigen::MatrixXd MPDensityMatrix::compute_hf_density(
    const SCFResult& uhf_result,
    int n_orb,
    int n_alpha,
    int n_beta
) {
    // Theory: HF density is diagonal with occupations
    // γ_pq^(HF) = δ_pq × n_p where n_p = 2 (doubly occ), 1 (singly occ), 0 (virtual)
    // Reference: Löwdin, Phys. Rev. 97, 1474 (1955)
    
    Eigen::MatrixXd opdm = Eigen::MatrixXd::Zero(n_orb, n_orb);
    
    // Fill diagonal: alpha + beta occupations
    for (int i = 0; i < n_alpha; ++i) {
        opdm(i, i) += 1.0;  // Alpha electron
    }
    for (int i = 0; i < n_beta; ++i) {
        opdm(i, i) += 1.0;  // Beta electron
    }
    
    return opdm;
}

// ============================================================================
// Helper: Add T1 contribution to 1-RDM
// ============================================================================
void MPDensityMatrix::add_t1_contribution(
    Eigen::MatrixXd& opdm,
    const Eigen::Tensor<double, 2>& t1_alpha,
    const Eigen::Tensor<double, 2>& t1_beta,
    int n_occ_alpha,
    int n_occ_beta
) {
    // Theory: T1 contribution to 1-RDM (particle-particle block)
    // γ_ab += Σ_i t_i^a t_i^b  (virtual-virtual block)
    // γ_ij -= Σ_a t_i^a t_j^a  (occupied-occupied block, depletion)
    // 
    // Reference: Bartlett, Ann. Rev. Phys. Chem. 32, 359 (1981)
    
    int n_virt_alpha = t1_alpha.dimension(1);
    int n_virt_beta = t1_beta.dimension(1);
    
    // Virtual-virtual block (particle-particle)
    // γ_ab += Σ_i t_i^a t_i^b
    for (int a = 0; a < n_virt_alpha; ++a) {
        for (int b = 0; b < n_virt_alpha; ++b) {
            double sum = 0.0;
            for (int i = 0; i < n_occ_alpha; ++i) {
                sum += t1_alpha(i, a) * t1_alpha(i, b);
            }
            opdm(n_occ_alpha + a, n_occ_alpha + b) += sum;
        }
    }
    
    // Beta spin
    for (int a = 0; a < n_virt_beta; ++a) {
        for (int b = 0; b < n_virt_beta; ++b) {
            double sum = 0.0;
            for (int i = 0; i < n_occ_beta; ++i) {
                sum += t1_beta(i, a) * t1_beta(i, b);
            }
            opdm(n_occ_beta + a, n_occ_beta + b) += sum;
        }
    }
    
    // Occupied-occupied block (hole-hole, depletion)
    // γ_ij -= Σ_a t_i^a t_j^a
    for (int i = 0; i < n_occ_alpha; ++i) {
        for (int j = 0; j < n_occ_alpha; ++j) {
            double sum = 0.0;
            for (int a = 0; a < n_virt_alpha; ++a) {
                sum += t1_alpha(i, a) * t1_alpha(j, a);
            }
            opdm(i, j) -= sum;
        }
    }
    
    // Beta spin
    for (int i = 0; i < n_occ_beta; ++i) {
        for (int j = 0; j < n_occ_beta; ++j) {
            double sum = 0.0;
            for (int a = 0; a < n_virt_beta; ++a) {
                sum += t1_beta(i, a) * t1_beta(j, a);
            }
            opdm(i, j) -= sum;
        }
    }
}

// ============================================================================
// Helper: Add T2 contribution to 1-RDM (simplified)
// ============================================================================
void MPDensityMatrix::add_t2_contribution(
    Eigen::MatrixXd& opdm,
    const Eigen::Tensor<double, 4>& t2_aa,
    const Eigen::Tensor<double, 4>& t2_bb,
    const Eigen::Tensor<double, 4>& t2_ab,
    int n_occ_alpha,
    int n_occ_beta,
    int n_virt_alpha,
    int n_virt_beta
) {
    // Theory: T2 contribution to 1-RDM (complex contractions)
    // γ_pq += Σ_ijab t_ij^ab × [matrix element]
    // 
    // Full formula involves many terms, here we compute main contributions:
    // 1. Virtual block: γ_ac += Σ_ijb t_ij^ab t_ij^cb
    // 2. Occupied block: γ_ki -= Σ_jab t_ij^ab t_kj^ab
    // 
    // Reference: Helgaker et al., "Molecular Electronic Structure Theory" (2000), Ch. 10
    
    // Alpha-Alpha contribution
    // Virtual-virtual block: γ_ac += 0.5 * Σ_ijb t_ij^ab t_ij^cb
    for (int a = 0; a < n_virt_alpha; ++a) {
        for (int c = 0; c < n_virt_alpha; ++c) {
            double sum = 0.0;
            for (int i = 0; i < n_occ_alpha; ++i) {
                for (int j = 0; j < n_occ_alpha; ++j) {
                    for (int b = 0; b < n_virt_alpha; ++b) {
                        sum += t2_aa(i, j, a, b) * t2_aa(i, j, c, b);
                    }
                }
            }
            opdm(n_occ_alpha + a, n_occ_alpha + c) += 0.5 * sum;
        }
    }
    
    // Occupied-occupied block: γ_ki -= 0.5 * Σ_jab t_ij^ab t_kj^ab
    for (int k = 0; k < n_occ_alpha; ++k) {
        for (int i = 0; i < n_occ_alpha; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n_occ_alpha; ++j) {
                for (int a = 0; a < n_virt_alpha; ++a) {
                    for (int b = 0; b < n_virt_alpha; ++b) {
                        sum += t2_aa(i, j, a, b) * t2_aa(k, j, a, b);
                    }
                }
            }
            opdm(k, i) -= 0.5 * sum;
        }
    }
    
    // Beta-Beta contribution (similar structure)
    for (int a = 0; a < n_virt_beta; ++a) {
        for (int c = 0; c < n_virt_beta; ++c) {
            double sum = 0.0;
            for (int i = 0; i < n_occ_beta; ++i) {
                for (int j = 0; j < n_occ_beta; ++j) {
                    for (int b = 0; b < n_virt_beta; ++b) {
                        sum += t2_bb(i, j, a, b) * t2_bb(i, j, c, b);
                    }
                }
            }
            opdm(n_occ_beta + a, n_occ_beta + c) += 0.5 * sum;
        }
    }
    
    for (int k = 0; k < n_occ_beta; ++k) {
        for (int i = 0; i < n_occ_beta; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n_occ_beta; ++j) {
                for (int a = 0; a < n_virt_beta; ++a) {
                    for (int b = 0; b < n_virt_beta; ++b) {
                        sum += t2_bb(i, j, a, b) * t2_bb(k, j, a, b);
                    }
                }
            }
            opdm(k, i) -= 0.5 * sum;
        }
    }
    
    // Alpha-Beta mixed contribution (simplified)
    // This is an approximation - full formula more complex
    for (int a = 0; a < n_virt_alpha; ++a) {
        for (int c = 0; c < n_virt_alpha; ++c) {
            double sum = 0.0;
            for (int i = 0; i < n_occ_alpha; ++i) {
                for (int j = 0; j < n_occ_beta; ++j) {
                    for (int b = 0; b < n_virt_beta; ++b) {
                        sum += t2_ab(i, j, a, b) * t2_ab(i, j, c, b);
                    }
                }
            }
            opdm(n_occ_alpha + a, n_occ_alpha + c) += sum;
        }
    }
}

// ============================================================================
// Compute 1-RDM from UMP2
// ============================================================================
Eigen::MatrixXd MPDensityMatrix::compute_opdm_mp2(
    const UMP2Result& ump2_result,
    const SCFResult& uhf_result
) {
    // TODO: UMP2Result does not store T2 amplitudes, only energies
    // This function requires T2Amplitudes structure to be passed separately
    // For now, return HF density as placeholder
    
    // Note: To fully implement, need to modify UMP2 to return amplitudes
    // or pass T2Amplitudes separately
    
    std::cerr << "Warning: MP2 natural orbital analysis not yet implemented\n";
    std::cerr << "         UMP2Result does not store T2 amplitudes.\n";
    std::cerr << "         Returning HF density only.\n";
    
    int n_orb = uhf_result.C_alpha.cols();
    
    // Return HF-only density (placeholder)
    // TODO: Implement when T2 amplitudes available
    return compute_hf_density(uhf_result, n_orb, 0, 0);  // Needs n_alpha/n_beta fix
}

// ============================================================================
// Compute 1-RDM from UMP3
// ============================================================================
Eigen::MatrixXd MPDensityMatrix::compute_opdm_mp3(
    const UMP3Result& ump3_result,
    const SCFResult& uhf_result
) {
    // Theory: MP3 includes T1^(2), T2^(1), T2^(2)
    // γ = γ^(HF) + γ^(T1^(2)) + γ^(T2^(1)) + γ^(T2^(2))
    // Reference: Bartlett, Ann. Rev. Phys. Chem. 32, 359 (1981)
    
    int n_orb = uhf_result.C_alpha.cols();
    
    // Start with HF density
    Eigen::MatrixXd opdm = compute_hf_density(
        uhf_result, n_orb, 
        ump3_result.n_occ_alpha, ump3_result.n_occ_beta
    );
    
    // Add T1^(2) contribution (2nd order singles)
    add_t1_contribution(
        opdm,
        ump3_result.t1_a_2,
        ump3_result.t1_b_2,
        ump3_result.n_occ_alpha,
        ump3_result.n_occ_beta
    );
    
    // Add T2^(1) contribution (1st order doubles, from MP2)
    add_t2_contribution(
        opdm,
        ump3_result.t2_aa_1,
        ump3_result.t2_bb_1,
        ump3_result.t2_ab_1,
        ump3_result.n_occ_alpha,
        ump3_result.n_occ_beta,
        ump3_result.n_virt_alpha,
        ump3_result.n_virt_beta
    );
    
    // Add T2^(2) contribution (2nd order doubles)
    add_t2_contribution(
        opdm,
        ump3_result.t2_aa_2,
        ump3_result.t2_bb_2,
        ump3_result.t2_ab_2,
        ump3_result.n_occ_alpha,
        ump3_result.n_occ_beta,
        ump3_result.n_virt_alpha,
        ump3_result.n_virt_beta
    );
    
    // Note: T3^(2) contribution omitted for now (very expensive, small contribution to 1-RDM)
    
    return opdm;
}

// ============================================================================
// Compute natural orbitals from 1-RDM
// ============================================================================
MPNaturalOrbitalResult MPDensityMatrix::compute_natural_orbitals(
    const Eigen::MatrixXd& opdm,
    const std::string& level,
    int n_electrons,
    int n_orbitals
) {
    // Theory: Diagonalize 1-RDM to get natural orbitals
    // γ|φ_i⟩ = n_i|φ_i⟩
    // Reference: Löwdin, Phys. Rev. 97, 1474 (1955)
    
    MPNaturalOrbitalResult result;
    result.level = level;
    result.n_electrons = n_electrons;
    result.n_orbitals = n_orbitals;
    
    // Diagonalize 1-RDM (symmetric matrix)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(opdm);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Failed to diagonalize 1-RDM\n";
        return result;
    }
    
    // Extract eigenvalues (occupations) and eigenvectors (natural orbitals)
    result.occupations = solver.eigenvalues();
    result.orbitals = solver.eigenvectors();
    
    // Sort by occupation (descending order)
    std::vector<std::pair<double, int>> occ_idx;
    for (int i = 0; i < n_orbitals; ++i) {
        occ_idx.push_back({result.occupations(i), i});
    }
    std::sort(occ_idx.begin(), occ_idx.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Reorder
    Eigen::VectorXd occ_sorted(n_orbitals);
    Eigen::MatrixXd orb_sorted(n_orbitals, n_orbitals);
    for (int i = 0; i < n_orbitals; ++i) {
        int idx = occ_idx[i].second;
        occ_sorted(i) = result.occupations(idx);
        orb_sorted.col(i) = result.orbitals.col(idx);
    }
    result.occupations = occ_sorted;
    result.orbitals = orb_sorted;
    
    // Compute correlation measure
    // Theory: Deviation from HF occupations (2.0 or 0.0)
    result.total_correlation = 0.0;
    int n_occ = n_electrons / 2;  // Approximate
    for (int i = 0; i < n_occ; ++i) {
        result.total_correlation += (2.0 - result.occupations(i));
    }
    for (int i = n_occ; i < n_orbitals; ++i) {
        result.total_correlation += result.occupations(i);
    }
    
    // Check multi-reference character
    result.is_multi_reference = check_multi_reference(
        result.occupations, n_electrons, 
        result.multi_ref_score, result.multi_ref_reason
    );
    
    return result;
}

// ============================================================================
// Analyze MP wavefunction
// ============================================================================
MPNaturalOrbitalResult MPDensityMatrix::analyze_wavefunction(
    const UMP3Result& ump3_result,
    MPNaturalOrbitalResult no_result
) {
    // Theory: Compute T1 and T2 diagnostics
    // Reference: Lee & Taylor, Int. J. Quantum Chem. Symp. 23, 199 (1989)
    
    int n_occ_alpha = ump3_result.n_occ_alpha;
    int n_occ_beta = ump3_result.n_occ_beta;
    
    // T1 diagnostic: ||T1|| / √(n_occ)
    double t1_norm_sq = 0.0;
    for (int i = 0; i < n_occ_alpha; ++i) {
        for (int a = 0; a < ump3_result.n_virt_alpha; ++a) {
            t1_norm_sq += ump3_result.t1_a_2(i, a) * ump3_result.t1_a_2(i, a);
        }
    }
    for (int i = 0; i < n_occ_beta; ++i) {
        for (int a = 0; a < ump3_result.n_virt_beta; ++a) {
            t1_norm_sq += ump3_result.t1_b_2(i, a) * ump3_result.t1_b_2(i, a);
        }
    }
    no_result.t1_diagnostic = std::sqrt(t1_norm_sq) / std::sqrt(n_occ_alpha + n_occ_beta);
    
    // T2 diagnostic: ||T2|| / √(n_occ²)
    double t2_norm_sq = 0.0;
    auto& t2_aa = ump3_result.t2_aa_1;
    auto& t2_bb = ump3_result.t2_bb_1;
    auto& t2_ab = ump3_result.t2_ab_1;
    
    for (int i = 0; i < n_occ_alpha; ++i) {
        for (int j = 0; j < n_occ_alpha; ++j) {
            for (int a = 0; a < ump3_result.n_virt_alpha; ++a) {
                for (int b = 0; b < ump3_result.n_virt_alpha; ++b) {
                    t2_norm_sq += t2_aa(i,j,a,b) * t2_aa(i,j,a,b);
                }
            }
        }
    }
    int n_occ_pairs = (n_occ_alpha + n_occ_beta) * (n_occ_alpha + n_occ_beta - 1) / 2;
    no_result.t2_diagnostic = std::sqrt(t2_norm_sq) / std::sqrt(n_occ_pairs);
    
    // Find largest amplitudes
    no_result.largest_t1_amplitude = 0.0;
    no_result.largest_t2_amplitude = 0.0;
    
    for (int i = 0; i < n_occ_alpha; ++i) {
        for (int a = 0; a < ump3_result.n_virt_alpha; ++a) {
            double val = std::abs(ump3_result.t1_a_2(i, a));
            if (val > no_result.largest_t1_amplitude) {
                no_result.largest_t1_amplitude = val;
            }
        }
    }
    
    for (int i = 0; i < n_occ_alpha; ++i) {
        for (int j = 0; j < n_occ_alpha; ++j) {
            for (int a = 0; a < ump3_result.n_virt_alpha; ++a) {
                for (int b = 0; b < ump3_result.n_virt_alpha; ++b) {
                    double val = std::abs(t2_aa(i,j,a,b));
                    if (val > no_result.largest_t2_amplitude) {
                        no_result.largest_t2_amplitude = val;
                    }
                }
            }
        }
    }
    
    return no_result;
}

// ============================================================================
// Check multi-reference character
// ============================================================================
bool MPDensityMatrix::check_multi_reference(
    const Eigen::VectorXd& occupations,
    int n_electrons,
    double& score,
    std::string& reason
) {
    // Theory: Check for fractional occupations
    // Strong deviation from 0.0/2.0 indicates multi-reference character
    // Reference: Lee & Taylor, Int. J. Quantum Chem. Symp. 23, 199 (1989)
    
    int n_occ = n_electrons / 2;
    int n_orb = occupations.size();
    
    // Compute score: deviation from ideal HF occupations
    score = 0.0;
    for (int i = 0; i < n_occ; ++i) {
        score += std::abs(2.0 - occupations(i));
    }
    for (int i = n_occ; i < n_orb; ++i) {
        score += std::abs(occupations(i));
    }
    score /= n_electrons;  // Normalize
    
    // Check thresholds
    if (score > 0.15) {
        reason = "Strong multi-reference character (large NO deviations)";
        return true;
    }
    
    // Check for fractional occupations in frontier orbitals
    double lumo_occ = (n_occ < n_orb) ? occupations(n_occ) : 0.0;
    double homo_occ = (n_occ > 0) ? occupations(n_occ - 1) : 0.0;
    
    if (lumo_occ > 0.1) {
        reason = "Significant LUMO occupation (" + std::to_string(lumo_occ) + ")";
        return true;
    }
    
    if (homo_occ < 1.8) {
        reason = "Depleted HOMO occupation (" + std::to_string(homo_occ) + ")";
        return true;
    }
    
    reason = "Single-reference character";
    return false;
}

// ============================================================================
// Print natural orbital report
// ============================================================================
void MPDensityMatrix::print_report(
    const MPNaturalOrbitalResult& result,
    bool verbose
) {
    using namespace std;
    
    cout << "\n";
    cout << "================================================================================\n";
    cout << "  NATURAL ORBITAL ANALYSIS - " << result.level << "\n";
    cout << "================================================================================\n";
    cout << "System: " << result.n_electrons << " electrons, " 
         << result.n_orbitals << " orbitals\n";
    cout << "\n";
    
    // Natural orbital occupations
    cout << "NATURAL ORBITAL OCCUPATIONS:\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << fixed << setprecision(6);
    
    int n_occ = result.n_electrons / 2;
    int n_print = verbose ? result.n_orbitals : std::min(10, result.n_orbitals);
    
    for (int i = 0; i < n_print; ++i) {
        cout << "  NO " << setw(3) << i << ":  n = " << setw(10) << result.occupations(i);
        if (i < n_occ) {
            cout << "  (occupied)";
        } else if (i == n_occ) {
            cout << "  (LUMO)";
        } else {
            cout << "  (virtual)";
        }
        cout << "\n";
    }
    
    if (!verbose && n_print < result.n_orbitals) {
        cout << "  ... (" << result.n_orbitals - n_print << " more orbitals)\n";
    }
    cout << "\n";
    
    // Correlation measures
    cout << "CORRELATION ANALYSIS:\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << "Total correlation measure: " << scientific << setprecision(4) 
         << result.total_correlation << "\n";
    cout << "\n";
    
    // Diagnostics
    cout << "DIAGNOSTIC NUMBERS:\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << fixed << setprecision(4);
    cout << "T1 diagnostic:  " << result.t1_diagnostic;
    if (result.t1_diagnostic < 0.02) {
        cout << "  (< 0.02: single-reference)\n";
    } else if (result.t1_diagnostic < 0.05) {
        cout << "  (0.02-0.05: weakly multi-reference)\n";
    } else {
        cout << "  (> 0.05: strongly multi-reference!)\n";
    }
    
    cout << "T2 diagnostic:  " << result.t2_diagnostic << "\n";
    cout << "Largest |t_i^a|: " << result.largest_t1_amplitude << "\n";
    cout << "Largest |t_ij^ab|: " << result.largest_t2_amplitude << "\n";
    cout << "\n";
    
    // Multi-reference assessment
    cout << "MULTI-REFERENCE CHARACTER:\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << "Score: " << result.multi_ref_score << "\n";
    cout << "Status: " << (result.is_multi_reference ? "⚠ MULTI-REFERENCE" : "✓ SINGLE-REFERENCE") << "\n";
    cout << "Reason: " << result.multi_ref_reason << "\n";
    
    cout << "================================================================================\n";
    cout << "\n";
}

// ============================================================================
// Compare with CI natural orbitals
// ============================================================================
double MPDensityMatrix::compare_with_ci(
    const MPNaturalOrbitalResult& mp_no,
    const MPNaturalOrbitalResult& ci_no
) {
    // Compute RMS difference in occupations
    if (mp_no.occupations.size() != ci_no.occupations.size()) {
        std::cerr << "Error: Different number of orbitals\n";
        return -1.0;
    }
    
    int n = mp_no.occupations.size();
    double rms = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = mp_no.occupations(i) - ci_no.occupations(i);
        rms += diff * diff;
    }
    rms = std::sqrt(rms / n);
    
    return rms;
}

} // namespace mp
} // namespace mshqc
