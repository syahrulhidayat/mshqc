/**
 * @file test_gradient_li.cc
 * @brief Test numerical gradient calculation for Li atom
 * 
 * Validates gradient implementation on Lithium atom (open-shell):
 * - Tests UHF gradient (doublet state, S=1/2)
 * - Validates spherical symmetry (all gradients ~0 for atom)
 * - Tests different basis sets (STO-3G, cc-pVDZ)
 * 
 * REFERENCE:
 * Li atom at origin should have zero gradient (no forces on single atom)
 * Any non-zero gradient indicates numerical error or symmetry breaking
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 * 
 * @note Test validates spherical symmetry for atomic systems.
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
    std::cout << "  Li Atom Numerical Gradient Test\n";
    std::cout << "  UHF (Open-shell, doublet)\n";
    std::cout << "=====================================\n\n";
    
    // ========================================================================
    // Setup Li atom at origin
    // ========================================================================
    
    Molecule li;
    li.set_charge(0);
    li.set_multiplicity(2);  // Doublet (S=1/2, unpaired electron)
    
    // Li atom at origin (Z=3)
    li.add_atom(3, 0.0, 0.0, 0.0);
    
    std::cout << "Molecular geometry:\n";
    li.print();
    std::cout << "\n";
    
    std::cout << "System: Li atom (doublet, 2S+1=2)\n";
    std::cout << "Electrons: 3 (2 alpha, 1 beta)\n";
    std::cout << "Expected gradient: ~0 (spherical symmetry)\n\n";
    
    // ========================================================================
    // Test 1: STO-3G basis
    // ========================================================================
    
    std::cout << "========================================\n";
    std::cout << "Test 1: STO-3G basis\n";
    std::cout << "========================================\n\n";
    
    std::string basis_name = "sto-3g";
    BasisSet basis(basis_name, li);
    auto integrals = std::make_shared<IntegralEngine>(li, basis);
    
    std::cout << "Basis set: " << basis_name << "\n";
    std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n\n";
    
    SCFConfig config;
    config.print_level = 0;  // Suppress SCF output
    config.max_iterations = 100;
    
    // Compute UHF gradient
    auto result_sto3g = compute_uhf_gradient_numerical(
        li, basis, integrals, 0, 2, config, 1e-5
    );
    
    print_gradient(result_sto3g, li);
    
    // Check if gradient is close to zero
    double grad_x = result_sto3g.gradient_by_atom(0, 0);
    double grad_y = result_sto3g.gradient_by_atom(0, 1);
    double grad_z = result_sto3g.gradient_by_atom(0, 2);
    double grad_norm = std::sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
    
    std::cout << "\n";
    std::cout << "Gradient components (should be ~0):\n";
    std::cout << "  ∂E/∂x: " << std::scientific << std::setprecision(4) << grad_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_z << "\n";
    std::cout << "  ||∇E||: " << grad_norm << "\n";
    
    // ========================================================================
    // Test 2: cc-pVDZ basis (larger, more accurate)
    // ========================================================================
    
    std::cout << "\n\n========================================\n";
    std::cout << "Test 2: cc-pVDZ basis\n";
    std::cout << "========================================\n\n";
    
    basis_name = "cc-pvdz";
    BasisSet basis_dz(basis_name, li);
    auto integrals_dz = std::make_shared<IntegralEngine>(li, basis_dz);
    
    std::cout << "Basis set: " << basis_name << "\n";
    std::cout << "Number of basis functions: " << basis_dz.n_basis_functions() << "\n\n";
    
    // Compute UHF gradient with larger basis
    auto result_pvdz = compute_uhf_gradient_numerical(
        li, basis_dz, integrals_dz, 0, 2, config, 1e-5
    );
    
    print_gradient(result_pvdz, li);
    
    // Check gradient
    grad_x = result_pvdz.gradient_by_atom(0, 0);
    grad_y = result_pvdz.gradient_by_atom(0, 1);
    grad_z = result_pvdz.gradient_by_atom(0, 2);
    grad_norm = std::sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
    
    std::cout << "\n";
    std::cout << "Gradient components (should be ~0):\n";
    std::cout << "  ∂E/∂x: " << std::scientific << std::setprecision(4) << grad_x << "\n";
    std::cout << "  ∂E/∂y: " << grad_y << "\n";
    std::cout << "  ∂E/∂z: " << grad_z << "\n";
    std::cout << "  ||∇E||: " << grad_norm << "\n";
    
    // ========================================================================
    // Validation
    // ========================================================================
    
    std::cout << "\n\n========================================\n";
    std::cout << "Validation Results\n";
    std::cout << "========================================\n\n";
    
    // Tolerance for atomic gradient (should be machine precision)
    double tol_atom = 1e-6;  // 1 microHartree/bohr
    
    // STO-3G test
    double norm_sto3g = result_sto3g.gradient.norm();
    bool pass_sto3g = norm_sto3g < tol_atom;
    
    // cc-pVDZ test
    double norm_pvdz = result_pvdz.gradient.norm();
    bool pass_pvdz = norm_pvdz < tol_atom;
    
    std::cout << "STO-3G basis:\n";
    std::cout << "  Gradient norm: " << std::scientific << std::setprecision(4) 
              << norm_sto3g << " Ha/bohr\n";
    std::cout << "  Tolerance:     " << tol_atom << " Ha/bohr\n";
    std::cout << "  Result:        " << (pass_sto3g ? "PASS ✓" : "FAIL ✗") << "\n";
    
    std::cout << "\ncc-pVDZ basis:\n";
    std::cout << "  Gradient norm: " << norm_pvdz << " Ha/bohr\n";
    std::cout << "  Tolerance:     " << tol_atom << " Ha/bohr\n";
    std::cout << "  Result:        " << (pass_pvdz ? "PASS ✓" : "FAIL ✗") << "\n";
    
    // Energy comparison
    std::cout << "\nEnergy comparison:\n";
    std::cout << "  E(UHF/STO-3G):   " << std::fixed << std::setprecision(8) 
              << result_sto3g.energy << " Ha\n";
    std::cout << "  E(UHF/cc-pVDZ):  " << result_pvdz.energy << " Ha\n";
    std::cout << "  Difference:      " << (result_pvdz.energy - result_sto3g.energy) 
              << " Ha\n";
    
    // Physical interpretation
    std::cout << "\nPhysical Interpretation:\n";
    std::cout << "  Single atom at origin → spherical symmetry\n";
    std::cout << "  Expectation: ∇E = 0 (no preferred direction)\n";
    std::cout << "  Non-zero gradient → numerical error only\n";
    
    // Overall result
    std::cout << "\n========================================\n";
    std::cout << "Overall Test Result\n";
    std::cout << "========================================\n";
    
    if (pass_sto3g && pass_pvdz) {
        std::cout << "\n✓ ALL TESTS PASSED\n";
        std::cout << "Gradient implementation correctly handles:\n";
        std::cout << "  - Open-shell systems (UHF)\n";
        std::cout << "  - Atomic spherical symmetry\n";
        std::cout << "  - Different basis sets\n\n";
        return 0;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED\n";
        if (!pass_sto3g) std::cout << "  - STO-3G gradient too large\n";
        if (!pass_pvdz) std::cout << "  - cc-pVDZ gradient too large\n";
        std::cout << "\nPossible causes:\n";
        std::cout << "  - Numerical precision issues\n";
        std::cout << "  - SCF convergence problems\n";
        std::cout << "  - Step size too large/small\n\n";
        return 1;
    }
}
