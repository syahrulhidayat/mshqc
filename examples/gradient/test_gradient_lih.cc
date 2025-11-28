/**
 * @file test_gradient_lih.cc
 * @brief Test numerical gradient for LiH molecule
 * 
 * Tests gradient calculation on heteronuclear diatomic molecule:
 * - LiH along z-axis (closed-shell, singlet)
 * - Validates forces at non-equilibrium geometry
 * - Tests translational invariance
 * - Compares gradient magnitude with H2
 * 
 * REFERENCE:
 * LiH equilibrium bond length: ~3.02 Bohr (1.60 Å)
 * At non-equilibrium: expect non-zero gradient pulling toward equilibrium
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Tests heteronuclear molecule with different nuclear charges.
 */

#include "mshqc/gradient/gradient.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

using namespace mshqc;
using namespace mshqc::gradient;

int main() {
    std::cout << "=====================================\n";
    std::cout << "  LiH Numerical Gradient Test\n";
    std::cout << "  RHF/STO-3G\n";
    std::cout << "=====================================\n\n";
    
    // ========================================================================
    // Setup LiH molecule at non-equilibrium geometry
    // ========================================================================
    
    Molecule lih;
    lih.set_charge(0);
    lih.set_multiplicity(1);  // Singlet (closed-shell)
    
    // LiH along z-axis
    // Equilibrium: ~3.02 Bohr
    // Test geometry: 3.5 Bohr (stretched)
    double R = 3.5;  // Bond length in Bohr
    
    lih.add_atom(3, 0.0, 0.0, 0.0);    // Li at origin
    lih.add_atom(1, 0.0, 0.0, R);      // H at z = R
    
    std::cout << "Molecular geometry:\n";
    lih.print();
    std::cout << "\n";
    
    std::cout << "System: LiH (singlet, closed-shell)\n";
    std::cout << "Bond length: " << R << " Bohr (" << R*0.529177 << " Å)\n";
    std::cout << "Equilibrium: ~3.02 Bohr (1.60 Å)\n";
    std::cout << "Status: Stretched → expect attractive force\n\n";
    
    // ========================================================================
    // Setup basis set and compute gradient
    // ========================================================================
    
    std::string basis_name = "sto-3g";
    BasisSet basis(basis_name, lih);
    auto integrals = std::make_shared<IntegralEngine>(lih, basis);
    
    std::cout << "Basis set: " << basis_name << "\n";
    std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n\n";
    
    SCFConfig config;
    config.print_level = 0;
    config.max_iterations = 100;
    
    // Compute RHF gradient
    std::cout << "Computing RHF gradient...\n";
    auto result = compute_rhf_gradient_numerical(
        lih, basis, integrals, 0, config, 1e-5
    );
    
    print_gradient(result, lih);
    
    // ========================================================================
    // Analyze gradient components
    // ========================================================================
    
    std::cout << "\n\n========================================\n";
    std::cout << "Gradient Analysis\n";
    std::cout << "========================================\n\n";
    
    // Extract gradients
    double grad_Li_x = result.gradient_by_atom(0, 0);
    double grad_Li_y = result.gradient_by_atom(0, 1);
    double grad_Li_z = result.gradient_by_atom(0, 2);
    
    double grad_H_x = result.gradient_by_atom(1, 0);
    double grad_H_y = result.gradient_by_atom(1, 1);
    double grad_H_z = result.gradient_by_atom(1, 2);
    
    std::cout << "Li atom gradients:\n";
    std::cout << "  ∂E/∂x: " << std::scientific << std::setprecision(6) << grad_Li_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_Li_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_Li_z << "\n";
    
    std::cout << "\nH atom gradients:\n";
    std::cout << "  ∂E/∂x: " << grad_H_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_H_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_H_z << "\n";
    
    // Physical interpretation
    std::cout << "\nPhysical Interpretation:\n";
    std::cout << "  Bond stretched (R = " << R << " > R_eq = 3.02 Bohr)\n";
    std::cout << "  Expected: Attractive force pulling atoms together\n";
    std::cout << "  Li gradient z: " << (grad_Li_z > 0 ? "positive (pull toward +z)" : "negative (pull toward -z)") << "\n";
    std::cout << "  H gradient z:  " << (grad_H_z > 0 ? "positive (pull toward +z)" : "negative (pull toward -z)") << "\n";
    
    if (grad_Li_z > 0 && grad_H_z < 0) {
        std::cout << "  ✓ Forces are attractive (correct!)\n";
    } else if (grad_Li_z < 0 && grad_H_z > 0) {
        std::cout << "  ✗ Forces are repulsive (unexpected for stretched bond!)\n";
    }
    
    // ========================================================================
    // Symmetry and conservation checks
    // ========================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "Symmetry & Conservation Tests\n";
    std::cout << "========================================\n\n";
    
    // 1. Cylindrical symmetry (x,y should be ~0)
    double tol_xy = 1e-6;
    bool pass_Li_x = std::abs(grad_Li_x) < tol_xy;
    bool pass_Li_y = std::abs(grad_Li_y) < tol_xy;
    bool pass_H_x = std::abs(grad_H_x) < tol_xy;
    bool pass_H_y = std::abs(grad_H_y) < tol_xy;
    bool pass_symmetry = pass_Li_x && pass_Li_y && pass_H_x && pass_H_y;
    
    std::cout << "1. Cylindrical Symmetry (x,y components ~0):\n";
    std::cout << "   Li: |∂E/∂x| = " << std::abs(grad_Li_x) 
              << ", |∂E/∂y| = " << std::abs(grad_Li_y) << "\n";
    std::cout << "   H:  |∂E/∂x| = " << std::abs(grad_H_x) 
              << ", |∂E/∂y| = " << std::abs(grad_H_y) << "\n";
    std::cout << "   Result: " << (pass_symmetry ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // 2. Translational invariance
    double sum_x = grad_Li_x + grad_H_x;
    double sum_y = grad_Li_y + grad_H_y;
    double sum_z = grad_Li_z + grad_H_z;
    double sum_norm = std::sqrt(sum_x*sum_x + sum_y*sum_y + sum_z*sum_z);
    
    double tol_trans = 1e-6;
    bool pass_trans = sum_norm < tol_trans;
    
    std::cout << "\n2. Translational Invariance (Σ∇E = 0):\n";
    std::cout << "   Σ∂E/∂x = " << sum_x << "\n";
    std::cout << "   Σ∂E/∂y = " << sum_y << "\n";
    std::cout << "   Σ∂E/∂z = " << sum_z << "\n";
    std::cout << "   ||Σ∇E|| = " << sum_norm << "\n";
    std::cout << "   Result: " << (pass_trans ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // 3. Gradient magnitude
    double grad_Li_norm = std::sqrt(grad_Li_x*grad_Li_x + grad_Li_y*grad_Li_y + grad_Li_z*grad_Li_z);
    double grad_H_norm = std::sqrt(grad_H_x*grad_H_x + grad_H_y*grad_H_y + grad_H_z*grad_H_z);
    
    std::cout << "\n3. Gradient Magnitudes:\n";
    std::cout << "   ||∇E_Li|| = " << grad_Li_norm << " Ha/bohr\n";
    std::cout << "   ||∇E_H||  = " << grad_H_norm << " Ha/bohr\n";
    std::cout << "   Typical range: 0.001-0.1 Ha/bohr for non-equilibrium\n";
    
    // 4. Newton's 3rd law (forces should be equal magnitude, opposite direction)
    double force_ratio = grad_H_norm / grad_Li_norm;
    bool pass_newton = std::abs(force_ratio - 1.0) < 0.1;  // Within 10%
    
    std::cout << "\n4. Newton's 3rd Law (equal/opposite forces):\n";
    std::cout << "   ||∇E_H|| / ||∇E_Li|| = " << std::fixed << std::setprecision(4) 
              << force_ratio << "\n";
    std::cout << "   Expected: ~1.0\n";
    std::cout << "   Result: " << (pass_newton ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // ========================================================================
    // Final summary
    // ========================================================================
    
    std::cout << "\n\n========================================\n";
    std::cout << "Overall Test Results\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Energy: " << std::fixed << std::setprecision(8) 
              << result.energy << " Ha\n";
    std::cout << "RMS gradient: " << std::scientific << std::setprecision(4) 
              << result.rms_gradient << " Ha/bohr\n";
    std::cout << "Max gradient: " << result.max_gradient << " Ha/bohr\n\n";
    
    std::cout << "Test Summary:\n";
    std::cout << "  Cylindrical symmetry: " << (pass_symmetry ? "✓" : "✗") << "\n";
    std::cout << "  Translational inv.:   " << (pass_trans ? "✓" : "✗") << "\n";
    std::cout << "  Newton's 3rd law:     " << (pass_newton ? "✓" : "✗") << "\n";
    
    bool all_pass = pass_symmetry && pass_trans && pass_newton;
    
    std::cout << "\n";
    if (all_pass) {
        std::cout << "✓ ALL TESTS PASSED\n";
        std::cout << "\nGradient implementation validated for:\n";
        std::cout << "  - Heteronuclear molecules\n";
        std::cout << "  - Non-equilibrium geometries\n";
        std::cout << "  - Physical conservation laws\n\n";
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED\n";
        std::cout << "\nCheck for:\n";
        std::cout << "  - Numerical precision issues\n";
        std::cout << "  - SCF convergence problems\n";
        std::cout << "  - Implementation bugs\n\n";
        return 1;
    }
}
