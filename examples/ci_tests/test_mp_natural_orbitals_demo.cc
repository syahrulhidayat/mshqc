/**
 * @file test_mp_natural_orbitals_demo.cc
 * @brief Demonstration of natural orbital analysis for MP wavefunctions
 * 
 * Theory References:
 * - Natural Orbitals: P.-O. Löwdin, Phys. Rev. **97**, 1474 (1955)
 * - 1-RDM from PT: R. J. Bartlett, Ann. Rev. Phys. Chem. **32**, 359 (1981)
 * - Diagnostics: T. J. Lee & P. R. Taylor, Int. J. Quantum Chem. Symp. **23**, 199 (1989)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 * 
 * @note This is a demonstration test using mock UMP3 data structure.
 *       For production use, integrate with actual UMP3 calculations.
 * 
 * @copyright MIT License
 */

#include "mshqc/mp/mp_density.h"
#include "mshqc/ump3.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::mp;

/**
 * @brief Create mock UMP3 result for testing
 * 
 * This simulates a small system (e.g., Li / STO-3G) for demonstration.
 * In production, this would come from actual UMP3 calculation.
 */
UMP3Result create_mock_ump3_result() {
    UMP3Result result;
    
    // System dimensions (Li atom: 3 electrons, 5 orbitals)
    result.n_occ_alpha = 2;   // 2 alpha electrons
    result.n_occ_beta = 1;    // 1 beta electron
    result.n_virt_alpha = 3;  // 3 virtual alpha
    result.n_virt_beta = 4;   // 4 virtual beta
    
    // Energies (from Agent 1 Li/cc-pVDZ results)
    result.e_uhf = -7.4324709030;
    result.e_mp2 = -0.0001922455;
    result.e_mp3 = -0.0064239943;
    result.e_total = result.e_uhf + result.e_mp2 + result.e_mp3;
    
    // Allocate T1^(2) amplitudes (second-order singles)
    result.t1_a_2 = Eigen::Tensor<double, 2>(result.n_occ_alpha, result.n_virt_alpha);
    result.t1_b_2 = Eigen::Tensor<double, 2>(result.n_occ_beta, result.n_virt_beta);
    
    // Allocate T2^(1) amplitudes (first-order doubles)
    result.t2_aa_1 = Eigen::Tensor<double, 4>(
        result.n_occ_alpha, result.n_occ_alpha,
        result.n_virt_alpha, result.n_virt_alpha
    );
    result.t2_bb_1 = Eigen::Tensor<double, 4>(
        result.n_occ_beta, result.n_occ_beta,
        result.n_virt_beta, result.n_virt_beta
    );
    result.t2_ab_1 = Eigen::Tensor<double, 4>(
        result.n_occ_alpha, result.n_occ_beta,
        result.n_virt_alpha, result.n_virt_beta
    );
    
    // Allocate T2^(2) amplitudes (second-order doubles)
    result.t2_aa_2 = Eigen::Tensor<double, 4>(
        result.n_occ_alpha, result.n_occ_alpha,
        result.n_virt_alpha, result.n_virt_alpha
    );
    result.t2_bb_2 = Eigen::Tensor<double, 4>(
        result.n_occ_beta, result.n_occ_beta,
        result.n_virt_beta, result.n_virt_beta
    );
    result.t2_ab_2 = Eigen::Tensor<double, 4>(
        result.n_occ_alpha, result.n_occ_beta,
        result.n_virt_alpha, result.n_virt_beta
    );
    
    // Fill with mock data (small realistic values)
    // T1 amplitudes: typically small (< 0.05 for single-reference)
    result.t1_a_2.setRandom();
    result.t1_a_2 = result.t1_a_2 * 0.02;  // Scale to ~0.02
    
    result.t1_b_2.setRandom();
    result.t1_b_2 = result.t1_b_2 * 0.03;  // Slightly larger for open-shell
    
    // T2 amplitudes: typically larger (0.01 - 0.1)
    result.t2_aa_1.setRandom();
    result.t2_aa_1 = result.t2_aa_1 * 0.05;
    
    result.t2_bb_1.setRandom();
    result.t2_bb_1 = result.t2_bb_1 * 0.04;
    
    result.t2_ab_1.setRandom();
    result.t2_ab_1 = result.t2_ab_1 * 0.06;
    
    result.t2_aa_2.setRandom();
    result.t2_aa_2 = result.t2_aa_2 * 0.02;
    
    result.t2_bb_2.setRandom();
    result.t2_bb_2 = result.t2_bb_2 * 0.02;
    
    result.t2_ab_2.setRandom();
    result.t2_ab_2 = result.t2_ab_2 * 0.03;
    
    return result;
}

/**
 * @brief Create mock SCF result for testing
 */
SCFResult create_mock_scf_result() {
    SCFResult result;
    
    int n_basis = 5;  // Li / STO-3G
    
    // Energies
    result.energy_electronic = -7.4324709030;
    result.energy_nuclear = 0.0;
    result.energy_total = result.energy_electronic;
    
    // MO coefficients (identity for simplicity)
    result.C_alpha = Eigen::MatrixXd::Identity(n_basis, n_basis);
    result.C_beta = Eigen::MatrixXd::Identity(n_basis, n_basis);
    
    // Orbital energies (mock)
    result.orbital_energies_alpha = Eigen::VectorXd::LinSpaced(n_basis, -2.5, 0.5);
    result.orbital_energies_beta = Eigen::VectorXd::LinSpaced(n_basis, -2.4, 0.6);
    
    // Density matrices (mock)
    result.P_alpha = Eigen::MatrixXd::Zero(n_basis, n_basis);
    result.P_beta = Eigen::MatrixXd::Zero(n_basis, n_basis);
    
    // Fock matrices (mock)
    result.F_alpha = Eigen::MatrixXd::Zero(n_basis, n_basis);
    result.F_beta = Eigen::MatrixXd::Zero(n_basis, n_basis);
    
    return result;
}

/**
 * @brief Test 1: Basic 1-RDM computation
 */
void test_opdm_computation() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "TEST 1: 1-RDM Computation from UMP3 Amplitudes\n";
    std::cout << "================================================================================\n";
    
    // Create mock data
    auto ump3 = create_mock_ump3_result();
    auto scf = create_mock_scf_result();
    
    std::cout << "System: Mock Li atom (3 electrons, 5 orbitals)\n";
    std::cout << "Method: UMP3\n";
    std::cout << "\n";
    
    // Compute 1-RDM
    std::cout << "Computing 1-RDM from T1^(2), T2^(1), T2^(2) amplitudes...\n";
    auto opdm = MPDensityMatrix::compute_opdm_mp3(ump3, scf);
    
    std::cout << "✓ 1-RDM computed successfully\n";
    std::cout << "  Dimension: " << opdm.rows() << " × " << opdm.cols() << "\n";
    std::cout << "  Trace (should ≈ N_electrons): " << std::fixed << std::setprecision(6) 
              << opdm.trace() << "\n";
    
    // Check sum rule
    double expected_trace = 3.0;  // 3 electrons
    double error = std::abs(opdm.trace() - expected_trace);
    
    if (error < 0.1) {
        std::cout << "  ✓ Sum rule satisfied (error: " << error << ")\n";
    } else {
        std::cout << "  ⚠ Sum rule violation (error: " << error << ")\n";
    }
    
    std::cout << "================================================================================\n";
}

/**
 * @brief Test 2: Natural orbital analysis
 */
void test_natural_orbital_analysis() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "TEST 2: Natural Orbital Analysis\n";
    std::cout << "================================================================================\n";
    
    // Create mock data
    auto ump3 = create_mock_ump3_result();
    auto scf = create_mock_scf_result();
    
    // Compute 1-RDM
    std::cout << "Step 1: Computing 1-RDM...\n";
    auto opdm = MPDensityMatrix::compute_opdm_mp3(ump3, scf);
    std::cout << "  ✓ Done\n\n";
    
    // Compute natural orbitals
    std::cout << "Step 2: Diagonalizing 1-RDM to obtain natural orbitals...\n";
    auto no_result = MPDensityMatrix::compute_natural_orbitals(
        opdm, "UMP3", 3, 5
    );
    std::cout << "  ✓ Done\n\n";
    
    // Print natural orbital occupations
    std::cout << "Natural Orbital Occupations:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < no_result.occupations.size(); ++i) {
        std::cout << "  NO " << i << ":  n = " << std::setw(10) << no_result.occupations(i);
        if (i < 2) {
            std::cout << "  (occupied)";
        } else if (i == 2) {
            std::cout << "  (LUMO)";
        } else {
            std::cout << "  (virtual)";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nCorrelation Analysis:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Total correlation measure: " << std::scientific << std::setprecision(4)
              << no_result.total_correlation << "\n";
    
    std::cout << "\nMulti-Reference Assessment:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Score: " << std::fixed << std::setprecision(4) << no_result.multi_ref_score << "\n";
    std::cout << "Status: " << (no_result.is_multi_reference ? "⚠ MULTI-REFERENCE" : "✓ SINGLE-REFERENCE") << "\n";
    std::cout << "Reason: " << no_result.multi_ref_reason << "\n";
    
    std::cout << "================================================================================\n";
}

/**
 * @brief Test 3: Wavefunction diagnostics (T1, T2)
 */
void test_wavefunction_diagnostics() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "TEST 3: Wavefunction Diagnostics (T1, T2)\n";
    std::cout << "================================================================================\n";
    
    // Create mock data
    auto ump3 = create_mock_ump3_result();
    auto scf = create_mock_scf_result();
    
    // Compute natural orbitals
    auto opdm = MPDensityMatrix::compute_opdm_mp3(ump3, scf);
    auto no_result = MPDensityMatrix::compute_natural_orbitals(opdm, "UMP3", 3, 5);
    
    // Analyze wavefunction
    std::cout << "Computing T1 and T2 diagnostics...\n\n";
    no_result = MPDensityMatrix::analyze_wavefunction(ump3, no_result);
    
    // Print diagnostics
    std::cout << "DIAGNOSTIC NUMBERS:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "T1 diagnostic:     " << no_result.t1_diagnostic;
    if (no_result.t1_diagnostic < 0.02) {
        std::cout << "  ✓ Single-reference\n";
    } else if (no_result.t1_diagnostic < 0.05) {
        std::cout << "  ⚠ Weakly multi-reference\n";
    } else {
        std::cout << "  ⚠ Strongly multi-reference!\n";
    }
    
    std::cout << "T2 diagnostic:     " << no_result.t2_diagnostic << "\n";
    std::cout << "Largest |t_i^a|:   " << no_result.largest_t1_amplitude << "\n";
    std::cout << "Largest |t_ij^ab|: " << no_result.largest_t2_amplitude << "\n";
    
    std::cout << "\nINTERPRETATION:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    if (no_result.t1_diagnostic < 0.02) {
        std::cout << "✓ System is well-described by single-reference methods (MP, CC)\n";
        std::cout << "✓ T1 diagnostic indicates no significant static correlation\n";
    } else {
        std::cout << "⚠ System shows multi-reference character\n";
        std::cout << "⚠ Consider CASSCF/MRCI methods for better accuracy\n";
    }
    
    std::cout << "================================================================================\n";
}

/**
 * @brief Test 4: Complete natural orbital report
 */
void test_complete_report() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "TEST 4: Complete Natural Orbital Report\n";
    std::cout << "================================================================================\n";
    
    // Create mock data
    auto ump3 = create_mock_ump3_result();
    auto scf = create_mock_scf_result();
    
    // Full analysis pipeline
    auto opdm = MPDensityMatrix::compute_opdm_mp3(ump3, scf);
    auto no_result = MPDensityMatrix::compute_natural_orbitals(opdm, "UMP3", 3, 5);
    no_result = MPDensityMatrix::analyze_wavefunction(ump3, no_result);
    
    // Print complete report
    MPDensityMatrix::print_report(no_result, false);  // verbose=false
    
    std::cout << "✓ Complete analysis finished\n";
    std::cout << "================================================================================\n";
}

int main() {
    std::cout << "\n";
    std::cout << "################################################################################\n";
    std::cout << "#                                                                              #\n";
    std::cout << "#        MP NATURAL ORBITAL ANALYSIS - DEMONSTRATION                          #\n";
    std::cout << "#                                                                              #\n";
    std::cout << "#  Theory: Natural orbitals from 1-RDM diagonalization                        #\n";
    std::cout << "#          γ|φ_i⟩ = n_i|φ_i⟩ where 0 ≤ n_i ≤ 2                               #\n";
    std::cout << "#                                                                              #\n";
    std::cout << "#  Reference: P.-O. Löwdin, Phys. Rev. 97, 1474 (1955)                        #\n";
    std::cout << "#             R. J. Bartlett, Ann. Rev. Phys. Chem. 32, 359 (1981)            #\n";
    std::cout << "#                                                                              #\n";
    std::cout << "################################################################################\n";
    
    std::cout << "\nNOTE: This is a DEMONSTRATION using mock UMP3 data.\n";
    std::cout << "      For production use, integrate with actual UMP3 calculations.\n";
    
    // Run all tests
    try {
        test_opdm_computation();
        test_natural_orbital_analysis();
        test_wavefunction_diagnostics();
        test_complete_report();
        
        std::cout << "\n";
        std::cout << "################################################################################\n";
        std::cout << "#                      ALL TESTS COMPLETED SUCCESSFULLY                       #\n";
        std::cout << "################################################################################\n";
        std::cout << "\n";
        
        std::cout << "SUMMARY:\n";
        std::cout << "--------\n";
        std::cout << "✓ 1-RDM computation from T amplitudes\n";
        std::cout << "✓ Natural orbital diagonalization\n";
        std::cout << "✓ T1/T2 diagnostic computation\n";
        std::cout << "✓ Multi-reference character detection\n";
        std::cout << "✓ Complete analysis report generation\n";
        std::cout << "\n";
        std::cout << "NEXT STEPS:\n";
        std::cout << "-----------\n";
        std::cout << "1. Integrate with actual UMP3 calculations (replace mock data)\n";
        std::cout << "2. Validate against CI natural orbitals for comparison\n";
        std::cout << "3. Test on real molecular systems (H2, Li, Be, etc.)\n";
        std::cout << "4. Add MP2 natural orbital support (requires UMP2Result extension)\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << "\n";
        return 1;
    }
}
