/**
 * rohf_test.cc
 * 
 * Test ROHF SCF for Li atom (doublet, 2S+1 = 2)
 * 
 * Li atom has 3 electrons: 2 alpha, 1 beta
 * Electronic configuration: 1s² 2s¹
 * 
 * Expected energy (STO-3G): ~-7.31 to -7.43 Ha
 * (exact value depends on implementation details)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "  ROHF Test: Lithium Atom (cc-pVTZ)\n";
        std::cout << "========================================\n";
        
        // Create Li atom at origin
        Molecule mol;
        mol.add_atom(3, 0.0, 0.0, 0.0);  // Z=3 (Li), position (0,0,0)
        
        std::cout << "\nMolecule:\n";
        std::cout << "  Lithium atom at origin\n";
        std::cout << "  Nuclear charge: 3\n";
        std::cout << "  Electrons: 3 (2 alpha, 1 beta)\n";
        std::cout << "  Multiplicity: 2 (doublet)\n";
        std::cout << "  Electronic config: 1s² 2s¹\n";
        
        // Load cc-pVTZ basis (larger, triple-zeta quality)
        BasisSet basis("cc-pvtz", mol);
        
        std::cout << "\nBasis Set:\n";
        std::cout << "  Name: cc-pVTZ\n";
        std::cout << "  Shells: " << basis.n_shells() << "\n";
        std::cout << "  Basis functions: " << basis.n_basis_functions() << "\n";
        
        // Setup SCF configuration
        SCFConfig config;
        config.max_iterations = 100;
        config.energy_threshold = 1e-8;
        config.density_threshold = 1e-6;
        config.print_level = 1;
        
        // Create ROHF calculator
        // Li: 3 electrons = 2 alpha + 1 beta
        ROHF scf(mol, basis, 2, 1, config);
        
        // Run SCF
        auto result = scf.run();
        
        // Print summary
        std::cout << "\n========================================\n";
        std::cout << "           FINAL SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "\nConvergence: " << (result.converged ? "SUCCESS" : "FAILED") << "\n";
        std::cout << "SCF iterations: " << result.iterations << "\n";
        
        std::cout << "\nFinal Energies (Hartree):\n";
        std::cout << "  Nuclear repulsion:     " << std::setw(16) << std::fixed 
                  << std::setprecision(10) << result.energy_nuclear << "\n";
        std::cout << "  Electronic energy:     " << std::setw(16) << result.energy_electronic << "\n";
        std::cout << "  Total SCF energy:      " << std::setw(16) << result.energy_total << "\n";
        
        std::cout << "\nOrbital Energies (Hartree):\n";
        std::cout << "  Occupied orbitals:\n";
        std::cout << "    α1 (1s):  " << std::setw(12) << std::setprecision(6) 
                  << result.orbital_energies_alpha(0) << "\n";
        std::cout << "    α2 (2s):  " << std::setw(12) << result.orbital_energies_alpha(1) << "\n";
        std::cout << "    β1 (1s):  " << std::setw(12) << result.orbital_energies_beta(0) << "\n";
        
        std::cout << "\n  Virtual orbitals:\n";
        std::cout << "    α3 (2p):  " << std::setw(12) << result.orbital_energies_alpha(2) << "\n";
        std::cout << "    α4 (2p):  " << std::setw(12) << result.orbital_energies_alpha(3) << "\n";
        std::cout << "    α5 (2p):  " << std::setw(12) << result.orbital_energies_alpha(4) << "\n";
        
        // Validation notes
        std::cout << "\n========================================\n";
        std::cout << "VALIDATION:\n";
        std::cout << "========================================\n";
        std::cout << "Li/cc-pVTZ ROHF Energy: " << std::fixed << std::setprecision(6)
                  << result.energy_total << " Ha\n";
        std::cout << "\nExpected range: -7.43 to -7.44 Ha (near-CBS quality)\n";
        
        if (result.energy_total > -7.45 && result.energy_total < -7.42) {
            std::cout << "✓ Energy within expected range!\n";
        } else {
            std::cout << "✗ Energy outside expected range (check convergence).\n";
        }
        
        std::cout << "\nReference values (for comparison):\n";
        std::cout << "  Psi4 Li/cc-pVTZ ROHF: ~-7.432 Ha\n";
        std::cout << "  CBS limit: ~-7.43 Ha\n";
        std::cout << "\n(Exact values depend on implementation details)\n";
        std::cout << "========================================\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
