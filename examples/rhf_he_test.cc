/**
 * @file rhf_he_test.cc
 * @brief Test RHF implementation on He atom (simplest closed-shell)
 * 
 * Test system: He atom
 * - 2 electrons (closed-shell, 1 doubly-occupied orbital)
 * - Basis: cc-pVDZ (5 basis functions)
 * 
 * Expected results (from Psi4):
 * - RHF/cc-pVDZ: -2.855 Ha
 * - RHF/cc-pVTZ: -2.861 Ha
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-11
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    try {
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════╗\n";
        std::cout << "║         RHF Test: He Atom (cc-pVDZ)                 ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════╝\n\n";
        
        // Create He atom at origin
        Molecule mol;
        mol.add_atom(2, 0.0, 0.0, 0.0);  // He (Z=2)
        
        std::cout << "Molecule: He atom\n";
        std::cout << "  Charge: " << mol.charge() << "\n";
        std::cout << "  Multiplicity: " << mol.multiplicity() << "\n";
        std::cout << "  Electrons: " << mol.n_electrons() << "\n";
        std::cout << "  Nuclear repulsion: " << std::fixed << std::setprecision(10) 
                  << mol.nuclear_repulsion_energy() << " Ha\n\n";
        
        // Load basis set
        BasisSet basis("cc-pVDZ", mol);
        
        std::cout << "Basis: cc-pVDZ\n";
        std::cout << "  Basis functions: " << basis.n_basis_functions() << "\n";
        std::cout << "  Shells: " << basis.n_shells() << "\n\n";
        
        // Create integral engine
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        // Run RHF
        SCFConfig config;
        config.max_iterations = 50;
        config.energy_threshold = 1e-8;
        config.density_threshold = 1e-6;
        config.print_level = 1;
        
        RHF rhf(mol, basis, integrals, config);
        auto result = rhf.compute();
        
        // Summary
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════╗\n";
        std::cout << "║                     Summary                          ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════╝\n";
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Total energy:    " << result.energy_total << " Ha\n";
        std::cout << "Converged:       " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations:      " << result.iterations << "\n";
        std::cout << "HOMO energy:     " << result.orbital_energies_alpha(result.n_occ_alpha-1) << " Ha\n";
        std::cout << "LUMO energy:     " << result.orbital_energies_alpha(result.n_occ_alpha) << " Ha\n";
        std::cout << "HOMO-LUMO gap:   " << (result.orbital_energies_alpha(result.n_occ_alpha) - 
                                              result.orbital_energies_alpha(result.n_occ_alpha-1)) << " Ha\n";
        
        // Validation
        double expected_energy = -2.855160;  // Psi4 cc-pVDZ result
        double error = std::abs(result.energy_total - expected_energy);
        
        std::cout << "\n╔═══════════════════════════════════════════════════════╗\n";
        std::cout << "║                    Validation                        ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════╝\n";
        std::cout << "Expected energy (Psi4): " << expected_energy << " Ha\n";
        std::cout << "MSH-QC energy:          " << result.energy_total << " Ha\n";
        std::cout << "Error:                  " << std::scientific << std::setprecision(4) 
                  << error << " Ha\n";
        
        if (error < 1e-5) {
            std::cout << "\n✓ Test PASSED! (error < 10 μHa)\n";
            return 0;
        } else if (error < 1e-3) {
            std::cout << "\n⚠ Test PASSED with warning (error < 1 mHa)\n";
            return 0;
        } else {
            std::cout << "\n✗ Test FAILED! (error > 1 mHa)\n";
            std::cout << "\nNote: Large error suggests issue in Fock matrix construction.\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << "\n";
        return 1;
    }
}
