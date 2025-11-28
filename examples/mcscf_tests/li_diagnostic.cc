// ============================================================================
// Li Atom Diagnostic Test - Print Energy Components
// ============================================================================

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "\n=== Li Atom Diagnostic Test ===\n\n";

    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);
    
    std::cout << "Nuclear repulsion: " << mol.nuclear_repulsion_energy() << " Ha\n";
    std::cout << "(Should be 0 for single atom)\n\n";

    // Smaller basis for testing: STO-3G
    const std::string basis_dir = "/home/shared/project-mshqc/data/basis";
    BasisSet basis("sto-3g", mol, basis_dir);
    
    std::cout << "Basis: STO-3G\n";
    std::cout << "Functions: " << basis.n_basis_functions() << "\n\n";

    auto integrals = std::make_shared<IntegralEngine>(mol, basis);

    // ROHF reference (Restricted Open-shell HF)
    SCFConfig cfg;
    cfg.max_iterations = 100;
    ROHF rohf(mol, basis, 2, 1, cfg);  // ROHF: no integrals parameter
    auto rohf_result = rohf.run();  // Method is run(), not compute()

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "ROHF Energy: " << rohf_result.energy_total << " Ha\n\n";

    // CASSCF: all electrons in active space (full CI)
    int nbf = basis.n_basis_functions();
    mcscf::ActiveSpace cas = mcscf::ActiveSpace::CAS(3, nbf, nbf, 3);

    std::cout << "Active space: " << cas.to_string() << "\n";
    std::cout << "This is FULL CI (all electrons, all orbitals)\n\n";

    mcscf::CASSCF casscf(mol, basis, integrals, cas);
    casscf.set_max_iterations(5);

    auto cas_result = casscf.compute(rohf_result);

    std::cout << "\n=== Results ===\n";
    std::cout << "ROHF:     " << rohf_result.energy_total << " Ha\n";
    std::cout << "CASSCF:   " << cas_result.e_casscf << " Ha\n";
    std::cout << "Δ(CASSCF-ROHF): " << (cas_result.e_casscf - rohf_result.energy_total) << " Ha\n\n";

    // Expected for Li/STO-3G:
    // ROHF ≈ -7.31 Ha
    // FCI ≈ -7.32 Ha (small correlation)
    
    std::cout << "Expected Li/STO-3G:\n";
    std::cout << "  ROHF ≈ -7.31 Ha\n";
    std::cout << "  FCI ≈ -7.32 Ha\n";
    std::cout << "  Correlation ≈ 10 mHa\n\n";

    if (cas_result.e_casscf < rohf_result.energy_total) {
        std::cout << "✓ Variational principle satisfied\n";
    } else {
        std::cout << "✗ WARNING: E(CASSCF) > E(ROHF)\n";
    }

    if (cas_result.e_casscf > -7.5 && cas_result.e_casscf < -7.2) {
        std::cout << "✓ Energy in reasonable range\n";
    } else {
        std::cout << "✗ WARNING: Energy out of expected range\n";
    }

    return 0;
}
