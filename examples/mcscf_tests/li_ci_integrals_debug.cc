// ============================================================================
// Li CI Integrals Debug - Print h_eff and compute HF determinant energy
// ============================================================================

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "\n=== Li CI Integrals Debug ===\n\n";

    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);
    
    const std::string basis_dir = "/home/shared/project-mshqc/data/basis";
    BasisSet basis("sto-3g", mol, basis_dir);
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);

    // ROHF reference
    SCFConfig cfg;
    cfg.max_iterations = 100;
    ROHF rohf(mol, basis, 2, 1, cfg);
    auto rohf_result = rohf.run();

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "ROHF Energy: " << rohf_result.energy_total << " Ha\n\n";

    // Get CI integrals for full active space
    mcscf::ActiveSpace cas = mcscf::ActiveSpace::CAS(3, 5, 5, 3);
    mcscf::CASSCF casscf(mol, basis, integrals, cas);
    
    // Access transform_integrals_to_active directly
    // (We need to make it public or add a test accessor)
    // For now, let's just compute manually
    
    // Get bare h in AO basis
    Eigen::MatrixXd h_ao = integrals->compute_kinetic() + 
                           integrals->compute_nuclear();
    
    // Transform to MO basis
    Eigen::MatrixXd C_mo = rohf_result.C_alpha;
    Eigen::MatrixXd h_mo = C_mo.transpose() * h_ao * C_mo;
    
    std::cout << "h_mo diagonal:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  " << i << ": " << h_mo(i, i) << "\n";
    }
    std::cout << "\n";
    
    std::cout << "ROHF orbital energies (for comparison):\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  " << i << ": " << rohf_result.orbital_energies_alpha(i) << "\n";
    }
    std::cout << "\n";
    
    // Compute HF determinant energy manually
    // Config: |1α 2α 1β⟩ (orbitals 0 and 1 doubly occupied, orbital 0 singly for beta actually wait)
    // Li ROHF: 2 alpha (orbitals 0,1), 1 beta (orbital 0)
    
    double e_hf_manual = 0.0;
    
    // One-electron: 2 * h_00 + 1 * h_11 + 1 * h_00 = 3*h_00 + h_11
    e_hf_manual += h_mo(0, 0);  // α electron in orbital 0
    e_hf_manual += h_mo(1, 1);  // α electron in orbital 1  
    e_hf_manual += h_mo(0, 0);  // β electron in orbital 0
    
    std::cout << "Manual HF determinant one-electron energy: " << e_hf_manual << "\n";
    
    // Need ERIs too
    // For now, just check if h_mo values make sense
    
    // Expected: sum of one-electron should be around
    // ε_0 + ε_1 + ε_0 for independent particles
    double approx = rohf_result.orbital_energies_alpha(0) + 
                    rohf_result.orbital_energies_alpha(1) +
                    rohf_result.orbital_energies_alpha(0);
    
    std::cout << "Approximate from orbital energies: " << approx << "\n";
    std::cout << "(Note: this is just rough estimate)\n";
    
    return 0;
}
