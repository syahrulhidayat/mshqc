/**
 * @file test_gradient_h2.cc
 * @brief Test numerical gradient calculation for H2 molecule
 * 
 * Validates numerical gradient implementation by testing:
 * - Symmetry: x,y gradients should be ~0 for linear molecule
 * - Translational invariance: Σ(∂E/∂R_A) = 0
 * - Opposite forces: Atoms should have equal/opposite z-gradients
 * 
 * REFERENCE:
 * For H2 at R = 1.4 bohr (non-equilibrium), RHF/STO-3G:
 * - Expected gradient magnitude: ~0.02-0.05 Ha/bohr
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Test validates implementation correctness, not copied from other software.
 */

#include "mshqc/gradient/gradient.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;
using namespace mshqc::gradient;

int main() {
    std::cout << "=====================================\n";
    std::cout << "  H2 Numerical Gradient Test\n";
    std::cout << "  RHF/STO-3G\n";
    std::cout << "=====================================\n\n";
    
    // ========================================================================
    // Setup H2 molecule at non-equilibrium geometry
    // ========================================================================
    
    Molecule h2;
    h2.set_charge(0);
    h2.set_multiplicity(1);
    
    // H2 along z-axis at R = 1.4 Bohr (slightly stretched from equilibrium)
    // Equilibrium for STO-3G is ~1.38 Bohr
    h2.add_atom(1, 0.0, 0.0, 0.0);    // H at origin
    h2.add_atom(1, 0.0, 0.0, 1.4);     // H at z = 1.4 Bohr
    
    std::cout << "Molecular geometry:\n";
    h2.print();
    std::cout << "\n";
    
    // ========================================================================
    // Setup basis set and integrals
    // ========================================================================
    
    std::string basis_name = "sto-3g";
    BasisSet basis(basis_name, h2);
    auto integrals = std::make_shared<IntegralEngine>(h2, basis);
    
    std::cout << "Basis set: " << basis_name << "\n";
    std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n\n";
    
    // ========================================================================
    // Method 1: Using convenience function
    // ========================================================================
    
    std::cout << "Method 1: Using convenience function\n";
    std::cout << "-------------------------------------\n";
    
    SCFConfig config;
    config.print_level = 0;  // Suppress SCF output for clarity
    
    auto result1 = compute_rhf_gradient_numerical(
        h2, basis, integrals, 0, config, 1e-5
    );
    
    print_gradient(result1, h2);
    
    // ========================================================================
    // Method 2: Using lambda function directly
    // ========================================================================
    
    std::cout << "\n\nMethod 2: Using lambda function\n";
    std::cout << "-------------------------------------\n";
    
    auto energy_func = [&](const Molecule& mol) -> double {
        BasisSet basis_disp(basis_name, mol);
        auto integrals_disp = std::make_shared<IntegralEngine>(mol, basis_disp);
        
        RHF rhf(mol, basis_disp, integrals_disp, config);
        auto scf_result = rhf.compute();
        
        return scf_result.energy_total;
    };
    
    NumericalGradient num_grad(energy_func, 1e-5, true);
    auto result2 = num_grad.compute(h2);
    
    print_gradient(result2, h2);
    
    // ========================================================================
    // Verify symmetry
    // ========================================================================
    
    std::cout << "\n\nSymmetry checks:\n";
    std::cout << "=====================================\n";
    
    // For H2 along z-axis:
    // - x and y gradients should be ~0
    // - z gradients should be equal and opposite
    
    double grad_H1_x = result2.gradient_by_atom(0, 0);
    double grad_H1_y = result2.gradient_by_atom(0, 1);
    double grad_H1_z = result2.gradient_by_atom(0, 2);
    
    double grad_H2_x = result2.gradient_by_atom(1, 0);
    double grad_H2_y = result2.gradient_by_atom(1, 1);
    double grad_H2_z = result2.gradient_by_atom(1, 2);
    
    std::cout << "X gradients (should be ~0):\n";
    std::cout << "  H1: " << std::scientific << std::setprecision(4) << grad_H1_x << "\n";
    std::cout << "  H2: " << grad_H2_x << "\n";
    
    std::cout << "\nY gradients (should be ~0):\n";
    std::cout << "  H1: " << grad_H1_y << "\n";
    std::cout << "  H2: " << grad_H2_y << "\n";
    
    std::cout << "\nZ gradients (should be opposite):\n";
    std::cout << "  H1: " << grad_H1_z << "\n";
    std::cout << "  H2: " << grad_H2_z << "\n";
    std::cout << "  Sum: " << (grad_H1_z + grad_H2_z) << " (should be ~0)\n";
    
    // Check translational invariance
    double sum_x = grad_H1_x + grad_H2_x;
    double sum_y = grad_H1_y + grad_H2_y;
    double sum_z = grad_H1_z + grad_H2_z;
    
    std::cout << "\nTranslational invariance (sum should be ~0):\n";
    std::cout << "  Σ∂E/∂x: " << sum_x << "\n";
    std::cout << "  Σ∂E/∂y: " << sum_y << "\n";
    std::cout << "  Σ∂E/∂z: " << sum_z << "\n";
    
    // Tolerances
    double tol_xy = 1e-6;  // x,y should be exactly 0 by symmetry
    double tol_sum = 1e-6; // Sum should be 0 by translational invariance
    
    bool pass_x = std::abs(grad_H1_x) < tol_xy && std::abs(grad_H2_x) < tol_xy;
    bool pass_y = std::abs(grad_H1_y) < tol_xy && std::abs(grad_H2_y) < tol_xy;
    bool pass_trans = std::abs(sum_x) < tol_sum && 
                      std::abs(sum_y) < tol_sum && 
                      std::abs(sum_z) < tol_sum;
    
    std::cout << "\n=====================================\n";
    std::cout << "Test results:\n";
    std::cout << "  X symmetry:    " << (pass_x ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << "  Y symmetry:    " << (pass_y ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << "  Translation:   " << (pass_trans ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << "=====================================\n\n";
    
    if (pass_x && pass_y && pass_trans) {
        std::cout << "All tests passed! ✓\n";
        return 0;
    } else {
        std::cout << "Some tests failed! ✗\n";
        return 1;
    }
}
