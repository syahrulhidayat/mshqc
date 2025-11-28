/**
 * @file test_optimize_h2.cc
 * @brief Test geometry optimization on H2 molecule
 * 
 * Compares three optimization algorithms:
 * 1. Steepest Descent (SD) - baseline, slowest
 * 2. Conjugate Gradient (CG) - faster than SD
 * 3. BFGS Quasi-Newton - fastest, builds Hessian approximation
 * 
 * Starting geometry: R = 2.0 Bohr (far from equilibrium at ~1.4 Bohr)
 * Method: RHF/STO-3G
 * 
 * Expected equilibrium: R_eq ≈ 1.39 Bohr, E ≈ -1.117 Ha
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Original test case - validates optimizer implementation correctness.
 */

#include "mshqc/gradient/optimizer.h"
#include "mshqc/molecule.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::gradient;

int main() {
    std::cout << "=====================================\n";
    std::cout << "  H2 Geometry Optimization Test\n";
    std::cout << "  RHF/STO-3G\n";
    std::cout << "=====================================\n\n";
    
    // ========================================================================
    // Setup initial H2 geometry (stretched, R = 2.0 Bohr)
    // ========================================================================
    
    Molecule h2_initial;
    h2_initial.set_charge(0);
    h2_initial.set_multiplicity(1);
    h2_initial.add_atom(1, 0.0, 0.0, 0.0);    // H at origin
    h2_initial.add_atom(1, 0.0, 0.0, 2.0);     // H at z = 2.0 Bohr
    
    std::cout << "Initial geometry:\n";
    h2_initial.print();
    std::cout << "\n";
    
    double R_initial = h2_initial.atom(1).z - h2_initial.atom(0).z;
    std::cout << "Initial bond length: " << std::fixed << std::setprecision(4) 
              << R_initial << " Bohr (" << R_initial * 0.529177 << " Å)\n";
    std::cout << "Expected equilibrium: ~1.39 Bohr (~0.74 Å)\n\n";
    
    // ========================================================================
    // Test 1: BFGS Quasi-Newton (fastest, default)
    // ========================================================================
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Test 1: BFGS Quasi-Newton\n";
    std::cout << "========================================\n";
    
    OptConfig config_bfgs;
    config_bfgs.algorithm = OptAlgorithm::BFGS;
    config_bfgs.max_iterations = 20;
    config_bfgs.print_level = 1;
    config_bfgs.print_geometry = false;
    
    auto result_bfgs = optimize_rhf(h2_initial, "sto-3g", config_bfgs);
    
    double R_final_bfgs = result_bfgs.final_geometry.atom(1).z - 
                          result_bfgs.final_geometry.atom(0).z;
    
    std::cout << "\nBFGS Results:\n";
    std::cout << "  Converged: " << (result_bfgs.converged ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  Iterations: " << result_bfgs.n_iterations << "\n";
    std::cout << "  Final R: " << std::fixed << std::setprecision(4) 
              << R_final_bfgs << " Bohr\n";
    std::cout << "  Final E: " << std::setprecision(8) 
              << result_bfgs.final_energy << " Ha\n";
    
    // ========================================================================
    // Test 2: Conjugate Gradient
    // ========================================================================
    
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Test 2: Conjugate Gradient\n";
    std::cout << "========================================\n";
    
    OptConfig config_cg;
    config_cg.algorithm = OptAlgorithm::CONJUGATE_GRADIENT;
    config_cg.max_iterations = 20;
    config_cg.print_level = 1;
    config_cg.print_geometry = false;
    
    auto result_cg = optimize_rhf(h2_initial, "sto-3g", config_cg);
    
    double R_final_cg = result_cg.final_geometry.atom(1).z - 
                        result_cg.final_geometry.atom(0).z;
    
    std::cout << "\nConjugate Gradient Results:\n";
    std::cout << "  Converged: " << (result_cg.converged ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  Iterations: " << result_cg.n_iterations << "\n";
    std::cout << "  Final R: " << std::fixed << std::setprecision(4) 
              << R_final_cg << " Bohr\n";
    std::cout << "  Final E: " << std::setprecision(8) 
              << result_cg.final_energy << " Ha\n";
    
    // ========================================================================
    // Test 3: Steepest Descent (slowest, for comparison)
    // ========================================================================
    
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Test 3: Steepest Descent\n";
    std::cout << "========================================\n";
    
    OptConfig config_sd;
    config_sd.algorithm = OptAlgorithm::STEEPEST_DESCENT;
    config_sd.max_iterations = 30;  // May need more iterations
    config_sd.print_level = 1;
    config_sd.print_geometry = false;
    
    auto result_sd = optimize_rhf(h2_initial, "sto-3g", config_sd);
    
    double R_final_sd = result_sd.final_geometry.atom(1).z - 
                        result_sd.final_geometry.atom(0).z;
    
    std::cout << "\nSteepest Descent Results:\n";
    std::cout << "  Converged: " << (result_sd.converged ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  Iterations: " << result_sd.n_iterations << "\n";
    std::cout << "  Final R: " << std::fixed << std::setprecision(4) 
              << R_final_sd << " Bohr\n";
    std::cout << "  Final E: " << std::setprecision(8) 
              << result_sd.final_energy << " Ha\n";
    
    // ========================================================================
    // Comparison Summary
    // ========================================================================
    
    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Algorithm Comparison\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Algorithm              Iters  Converged  Final R (Bohr)  Final E (Ha)\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << std::left;
    std::cout << std::setw(22) << "BFGS" 
              << std::right << std::setw(6) << result_bfgs.n_iterations << "  "
              << std::setw(9) << (result_bfgs.converged ? "YES ✓" : "NO ✗") << "  "
              << std::fixed << std::setprecision(4) << std::setw(14) << R_final_bfgs << "  "
              << std::setprecision(8) << result_bfgs.final_energy << "\n";
    
    std::cout << std::setw(22) << "Conjugate Gradient" 
              << std::right << std::setw(6) << result_cg.n_iterations << "  "
              << std::setw(9) << (result_cg.converged ? "YES ✓" : "NO ✗") << "  "
              << std::setw(14) << R_final_cg << "  "
              << std::setprecision(8) << result_cg.final_energy << "\n";
    
    std::cout << std::setw(22) << "Steepest Descent" 
              << std::right << std::setw(6) << result_sd.n_iterations << "  "
              << std::setw(9) << (result_sd.converged ? "YES ✓" : "NO ✗") << "  "
              << std::setw(14) << R_final_sd << "  "
              << std::setprecision(8) << result_sd.final_energy << "\n";
    
    std::cout << "\n";
    std::cout << "Key observations:\n";
    std::cout << "  - BFGS typically converges fastest (fewest iterations)\n";
    std::cout << "  - All algorithms should reach same final geometry\n";
    std::cout << "  - Steepest descent slowest (linear convergence)\n";
    std::cout << "  - CG intermediate (superlinear convergence)\n";
    std::cout << "  - BFGS fastest (quadratic convergence)\n";
    
    // ========================================================================
    // Validation
    // ========================================================================
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Validation\n";
    std::cout << "========================================\n\n";
    
    // Check if all algorithms converged
    bool all_converged = result_bfgs.converged && result_cg.converged && result_sd.converged;
    
    // Check if final energies are consistent (within 1e-6)
    double E_max = std::max({result_bfgs.final_energy, result_cg.final_energy, result_sd.final_energy});
    double E_min = std::min({result_bfgs.final_energy, result_cg.final_energy, result_sd.final_energy});
    double E_diff = std::abs(E_max - E_min);
    bool energies_consistent = E_diff < 1e-6;
    
    // Check if bond lengths are consistent
    double R_max = std::max({R_final_bfgs, R_final_cg, R_final_sd});
    double R_min = std::min({R_final_bfgs, R_final_cg, R_final_sd});
    double R_diff = std::abs(R_max - R_min);
    bool bonds_consistent = R_diff < 0.01;  // Within 0.01 Bohr
    
    // Check if final geometry is reasonable (R_eq ~ 1.39 Bohr)
    double R_expected = 1.39;
    bool geometry_reasonable = std::abs(R_final_bfgs - R_expected) < 0.05;
    
    std::cout << "All converged:        " << (all_converged ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "Energies consistent:  " << (energies_consistent ? "YES ✓" : "NO ✗") 
              << " (Δ = " << std::scientific << std::setprecision(2) << E_diff << " Ha)\n";
    std::cout << "Bonds consistent:     " << (bonds_consistent ? "YES ✓" : "NO ✗") 
              << " (Δ = " << std::fixed << std::setprecision(4) << R_diff << " Bohr)\n";
    std::cout << "Geometry reasonable:  " << (geometry_reasonable ? "YES ✓" : "NO ✗") 
              << " (R = " << R_final_bfgs << ", expected ~" << R_expected << " Bohr)\n";
    
    std::cout << "\n========================================\n";
    
    if (all_converged && energies_consistent && bonds_consistent && geometry_reasonable) {
        std::cout << "Overall: ALL TESTS PASSED ✓\n";
        std::cout << "========================================\n\n";
        return 0;
    } else {
        std::cout << "Overall: SOME TESTS FAILED ✗\n";
        std::cout << "========================================\n\n";
        return 1;
    }
}
