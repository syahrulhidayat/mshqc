/**
 * @file test_gradient_h2o_pvdz.cc
 * @brief Test numerical gradient for H2O molecule with cc-pVDZ basis
 * 
 * Tests gradient calculation on water molecule:
 * - Bent geometry (C2v symmetry)
 * - cc-pVDZ basis (larger, more accurate)
 * - Three atoms (polyatomic system)
 * - Validates 3N=9 gradient components
 * 
 * REFERENCE:
 * H2O equilibrium geometry:
 * - R(O-H) ≈ 1.81 Bohr (0.96 Å)
 * - θ(H-O-H) ≈ 104.5°
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Tests polyatomic molecule with larger basis set.
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
    std::cout << "  H2O Numerical Gradient Test\n";
    std::cout << "  RHF/cc-pVDZ\n";
    std::cout << "=====================================\n\n";
    
    // ========================================================================
    // Setup H2O molecule at non-equilibrium geometry
    // ========================================================================
    
    Molecule h2o;
    h2o.set_charge(0);
    h2o.set_multiplicity(1);  // Singlet (closed-shell)
    
    // H2O geometry (slightly stretched from equilibrium)
    // Equilibrium: R(O-H) ≈ 1.81 Bohr, θ ≈ 104.5°
    // Test: R = 1.9 Bohr, θ = 104.5°
    
    double R_OH = 1.9;  // O-H bond length (Bohr)
    double theta = 104.5 * M_PI / 180.0;  // H-O-H angle (radians)
    
    // O at origin
    h2o.add_atom(8, 0.0, 0.0, 0.0);
    
    // H1 and H2 symmetric about z-axis
    double x_H = R_OH * sin(theta / 2.0);
    double z_H = R_OH * cos(theta / 2.0);
    
    h2o.add_atom(1, -x_H, 0.0, z_H);   // H1 (left)
    h2o.add_atom(1,  x_H, 0.0, z_H);   // H2 (right)
    
    std::cout << "Molecular geometry:\n";
    h2o.print();
    std::cout << "\n";
    
    std::cout << "System: H2O (bent, C2v symmetry)\n";
    std::cout << "R(O-H): " << R_OH << " Bohr (" << R_OH*0.529177 << " Å)\n";
    std::cout << "θ(H-O-H): " << theta*180.0/M_PI << "°\n";
    std::cout << "Basis: cc-pVDZ (25 basis functions expected)\n";
    std::cout << "Status: Slightly stretched → expect forces\n\n";
    
    // ========================================================================
    // Setup basis set and compute gradient
    // ========================================================================
    
    std::string basis_name = "cc-pvdz";
    BasisSet basis(basis_name, h2o);
    auto integrals = std::make_shared<IntegralEngine>(h2o, basis);
    
    std::cout << "Basis set: " << basis_name << "\n";
    std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n\n";
    std::cout << "Number of electrons: " << h2o.n_electrons() << "\n\n";
    
    SCFConfig config;
    config.print_level = 0;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    
    // Compute RHF gradient
    std::cout << "Computing RHF/cc-pVDZ gradient...\n";
    std::cout << "(This may take a few minutes due to larger basis)\n\n";
    
    auto result = compute_rhf_gradient_numerical(
        h2o, basis, integrals, 0, config, 1e-5
    );
    
    print_gradient(result, h2o);
    
    // ========================================================================
    // Analyze gradient components
    // ========================================================================
    
    std::cout << "\n\n========================================\n";
    std::cout << "Gradient Analysis\n";
    std::cout << "========================================\n\n";
    
    // Extract gradients
    double grad_O_x = result.gradient_by_atom(0, 0);
    double grad_O_y = result.gradient_by_atom(0, 1);
    double grad_O_z = result.gradient_by_atom(0, 2);
    
    double grad_H1_x = result.gradient_by_atom(1, 0);
    double grad_H1_y = result.gradient_by_atom(1, 1);
    double grad_H1_z = result.gradient_by_atom(1, 2);
    
    double grad_H2_x = result.gradient_by_atom(2, 0);
    double grad_H2_y = result.gradient_by_atom(2, 1);
    double grad_H2_z = result.gradient_by_atom(2, 2);
    
    std::cout << "Oxygen gradients:\n";
    std::cout << "  ∂E/∂x: " << std::scientific << std::setprecision(6) << grad_O_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_O_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_O_z << "\n";
    
    std::cout << "\nHydrogen 1 gradients:\n";
    std::cout << "  ∂E/∂x: " << grad_H1_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_H1_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_H1_z << "\n";
    
    std::cout << "\nHydrogen 2 gradients:\n";
    std::cout << "  ∂E/∂x: " << grad_H2_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_H2_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_H2_z << "\n";
    
    // ========================================================================
    // Symmetry and conservation checks
    // ========================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "Symmetry & Conservation Tests\n";
    std::cout << "========================================\n\n";
    
    // 1. C2v symmetry (reflection through xz-plane)
    // H1 and H2 should have symmetric x-components
    double tol_sym = 1e-5;
    bool pass_x_sym = std::abs(grad_H1_x + grad_H2_x) < tol_sym;  // Should be opposite
    bool pass_z_sym = std::abs(grad_H1_z - grad_H2_z) < tol_sym;  // Should be equal
    
    std::cout << "1. C2v Symmetry (reflection through xz-plane):\n";
    std::cout << "   H1 ∂E/∂x: " << grad_H1_x << "\n";
    std::cout << "   H2 ∂E/∂x: " << grad_H2_x << " (should be opposite)\n";
    std::cout << "   Difference: " << (grad_H1_x + grad_H2_x) << "\n";
    std::cout << "   Result: " << (pass_x_sym ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // 2. Y-components should be ~0 (molecule in xz-plane)
    double tol_y = 1e-6;
    bool pass_O_y = std::abs(grad_O_y) < tol_y;
    bool pass_H1_y = std::abs(grad_H1_y) < tol_y;
    bool pass_H2_y = std::abs(grad_H2_y) < tol_y;
    bool pass_plane = pass_O_y && pass_H1_y && pass_H2_y;
    
    std::cout << "\n2. Planar Symmetry (molecule in xz-plane):\n";
    std::cout << "   O  ∂E/∂y: " << grad_O_y << "\n";
    std::cout << "   H1 ∂E/∂y: " << grad_H1_y << "\n";
    std::cout << "   H2 ∂E/∂y: " << grad_H2_y << "\n";
    std::cout << "   Expected: all ~0\n";
    std::cout << "   Result: " << (pass_plane ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // 3. Translational invariance
    double sum_x = grad_O_x + grad_H1_x + grad_H2_x;
    double sum_y = grad_O_y + grad_H1_y + grad_H2_y;
    double sum_z = grad_O_z + grad_H1_z + grad_H2_z;
    double sum_norm = std::sqrt(sum_x*sum_x + sum_y*sum_y + sum_z*sum_z);
    
    double tol_trans = 1e-5;
    bool pass_trans = sum_norm < tol_trans;
    
    std::cout << "\n3. Translational Invariance (Σ∇E = 0):\n";
    std::cout << "   Σ∂E/∂x = " << sum_x << "\n";
    std::cout << "   Σ∂E/∂y = " << sum_y << "\n";
    std::cout << "   Σ∂E/∂z = " << sum_z << "\n";
    std::cout << "   ||Σ∇E|| = " << sum_norm << "\n";
    std::cout << "   Result: " << (pass_trans ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // 4. Gradient magnitudes
    double grad_O_norm = std::sqrt(grad_O_x*grad_O_x + grad_O_y*grad_O_y + grad_O_z*grad_O_z);
    double grad_H1_norm = std::sqrt(grad_H1_x*grad_H1_x + grad_H1_y*grad_H1_y + grad_H1_z*grad_H1_z);
    double grad_H2_norm = std::sqrt(grad_H2_x*grad_H2_x + grad_H2_y*grad_H2_y + grad_H2_z*grad_H2_z);
    
    std::cout << "\n4. Gradient Magnitudes:\n";
    std::cout << "   ||∇E_O||  = " << grad_O_norm << " Ha/bohr\n";
    std::cout << "   ||∇E_H1|| = " << grad_H1_norm << " Ha/bohr\n";
    std::cout << "   ||∇E_H2|| = " << grad_H2_norm << " Ha/bohr\n";
    std::cout << "   H1/H2 symmetry: " << std::abs(grad_H1_norm - grad_H2_norm) 
              << " (should be ~0)\n";
    
    bool pass_H_sym = std::abs(grad_H1_norm - grad_H2_norm) < 1e-5;
    
    // ========================================================================
    // Energy and convergence info
    // ========================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "Energy & Convergence\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Energy (RHF/cc-pVDZ): " << std::fixed << std::setprecision(8) 
              << result.energy << " Ha\n";
    std::cout << "RMS gradient:         " << std::scientific << std::setprecision(4) 
              << result.rms_gradient << " Ha/bohr\n";
    std::cout << "Max gradient:         " << result.max_gradient << " Ha/bohr\n";
    
    bool converged = is_gradient_converged(result.gradient);
    std::cout << "\nGeometry converged:   " << (converged ? "YES" : "NO") << "\n";
    std::cout << "(Thresholds: RMS < 3e-4, Max < 4.5e-4 Ha/bohr)\n";
    
    // ========================================================================
    // Final summary
    // ========================================================================
    
    std::cout << "\n\n========================================\n";
    std::cout << "Overall Test Results\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Test Summary:\n";
    std::cout << "  C2v symmetry:         " << (pass_x_sym ? "✓" : "✗") << "\n";
    std::cout << "  Planar symmetry:      " << (pass_plane ? "✓" : "✗") << "\n";
    std::cout << "  Translational inv.:   " << (pass_trans ? "✓" : "✗") << "\n";
    std::cout << "  H1/H2 equivalence:    " << (pass_H_sym ? "✓" : "✗") << "\n";
    
    bool all_pass = pass_x_sym && pass_plane && pass_trans && pass_H_sym;
    
    std::cout << "\n";
    if (all_pass) {
        std::cout << "✓ ALL TESTS PASSED\n";
        std::cout << "\nGradient implementation validated for:\n";
        std::cout << "  - Polyatomic molecules (3+ atoms)\n";
        std::cout << "  - Larger basis sets (cc-pVDZ)\n";
        std::cout << "  - Point group symmetry (C2v)\n";
        std::cout << "  - Non-linear geometries\n\n";
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED\n";
        std::cout << "\nPossible causes:\n";
        std::cout << "  - Numerical precision with larger basis\n";
        std::cout << "  - SCF convergence issues\n";
        std::cout << "  - Symmetry breaking\n\n";
        return 1;
    }
}
