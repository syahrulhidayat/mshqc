/**
 * @file rhf_test.cc
 * @brief Test RHF implementation on H2O molecule
 * 
 * Test system: H2O (water molecule)
 * - 10 electrons (closed-shell)
 * - 5 doubly-occupied orbitals
 * - Basis: cc-pVDZ
 * 
 * Expected results (from Psi4):
 * - RHF energy: -76.026760 Ha
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
        std::cout << "║         RHF Test: H2O Molecule (cc-pVDZ)            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════╝\n\n";
        
        // Create H2O molecule (geometry in Angstrom)
        // Equilibrium geometry: O-H = 0.9584 Å, H-O-H = 104.45°
        Molecule mol;
        mol.add_atom(8,  0.000000,  0.000000,  0.117176);   // Oxygen (Z=8)
        mol.add_atom(1,  0.000000,  0.755453, -0.468706);   // Hydrogen 1 (Z=1)
        mol.add_atom(1,  0.000000, -0.755453, -0.468706);   // Hydrogen 2 (Z=1)
        
        std::cout << "Molecule: H2O\n";
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
        double expected_energy = -76.026760;  // Psi4 cc-pVDZ result
        double error = std::abs(result.energy_total - expected_energy);
        
        std::cout << "\n╔═══════════════════════════════════════════════════════╗\n";
        std::cout << "║                    Validation                        ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════╝\n";
        std::cout << "Expected energy (Psi4): " << expected_energy << " Ha\n";
        std::cout << "MSH-QC energy:          " << result.energy_total << " Ha\n";
        std::cout << "Error:                  " << std::scientific << std::setprecision(4) 
                  << error << " Ha\n";
        
        if (error < 1e-6) {
            std::cout << "\n✓ Test PASSED! (error < 1 μHa)\n";
            return 0;
        } else if (error < 1e-5) {
            std::cout << "\n⚠ Test PASSED with warning (error < 10 μHa)\n";
            return 0;
        } else {
            std::cout << "\n✗ Test FAILED! (error > 10 μHa)\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << "\n";
        return 1;
    }
}
