// Test Li atom ROHF + ROMP2 with cc-pVTZ basis
// Expected bug: ROMP2 may have issues

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mp2.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    try {
        // Li atom at origin (doublet, 2S+1=2)
        Molecule mol;
        mol.add_atom(3, 0.0, 0.0, 0.0);  // Z=3 for Li
        
        std::cout << "Lithium atom test (cc-pVTZ basis)\n";
        std::cout << "==================================\n\n";
        
        // Load cc-pVTZ basis
        BasisSet basis("cc-pVTZ", mol, "../data/basis");
        
        std::cout << "Basis set loaded: " << basis.name() << "\n";
        std::cout << "Number of shells: " << basis.n_shells() << "\n";
        std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n\n";
        
        // Electron config: 3 electrons (1s² 2s¹)
        // For ROHF: 2 doubly occupied (1s²) + 1 singly occupied (2s¹)
        // So n_alpha = 2, n_beta = 1
        int n_alpha = 2;
        int n_beta = 1;
        
        std::cout << "Electrons: " << n_alpha << " alpha, " << n_beta << " beta\n";
        std::cout << "Multiplicity: " << (n_alpha - n_beta + 1) << " (doublet)\n\n";
        
        // ROHF calculation
        SCFConfig cfg;
        cfg.max_iterations = 100;
        cfg.energy_threshold = 1e-8;
        cfg.density_threshold = 1e-6;
        cfg.print_level = 1;
        
        ROHF rohf(mol, basis, n_alpha, n_beta, cfg);
        
        std::cout << "Starting ROHF calculation...\n";
        std::cout << "----------------------------\n";
        SCFResult scf_result = rohf.run();
        
        if (!scf_result.converged) {
            std::cerr << "\nWARNING: SCF did not converge!\n";
            return 1;
        }
        
        std::cout << "\n*** ROHF CONVERGED ***\n";
        std::cout << "Final energy: " << std::fixed << std::setprecision(10) 
                  << scf_result.energy_total << " Ha\n\n";
        
        // ROMP2 calculation
        std::cout << "Starting ROMP2 calculation...\n";
        std::cout << "----------------------------\n";
        
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        ROMP2 romp2(scf_result, integrals);
        
        MP2Result mp2_result = romp2.compute();
        
        std::cout << "\n*** ROMP2 RESULTS ***\n";
        std::cout << "ROHF energy:        " << std::fixed << std::setprecision(10) 
                  << mp2_result.energy_scf << " Ha\n";
        std::cout << "MP2 correlation:    " << mp2_result.energy_mp2_corr << " Ha\n";
        std::cout << "  Same-spin:        " << mp2_result.energy_mp2_ss << " Ha\n";
        std::cout << "  Opposite-spin:    " << mp2_result.energy_mp2_os << " Ha\n";
        std::cout << "Total ROMP2 energy: " << mp2_result.energy_total << " Ha\n\n";
        
        // Check for potential bugs
        std::cout << "Bug check:\n";
        if (std::isnan(mp2_result.energy_mp2_corr) || 
            std::isinf(mp2_result.energy_mp2_corr)) {
            std::cout << "  ❌ MP2 correlation energy is NaN/Inf!\n";
        } else if (std::abs(mp2_result.energy_mp2_corr) > 1.0) {
            std::cout << "  ⚠️  MP2 correlation suspiciously large (> 1 Ha)\n";
        } else if (mp2_result.energy_mp2_corr > 0.0) {
            std::cout << "  ⚠️  MP2 correlation positive (should be negative)\n";
        } else {
            std::cout << "  ✓ MP2 correlation looks reasonable\n";
        }
        
        if (std::abs(mp2_result.energy_mp2_os) < 1e-12) {
            std::cout << "  ⚠️  Opposite-spin contribution near zero (unexpected)\n";
        } else {
            std::cout << "  ✓ Opposite-spin contribution non-zero\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
