/**
 * @file rmp2_test.cc
 * @brief Test RMP2 implementation on H2O molecule
 * 
 * Target: H2O/cc-pVDZ from Psi4 reference
 * Expected RHF energy: ~ -76.026 Ha
 * Expected MP2 correlation: ~ -0.204 Ha
 * Expected total: ~ -76.230 Ha
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include "mshqc/foundation/rmp2.h"
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mshqc;
using namespace mshqc::foundation;

int main() {
    std::cout << "=== RMP2 Test: H2O/cc-pVDZ ===\n\n";
    
    try {
        // Define H2O molecule (equilibrium geometry, Angstrom)
        Molecule mol;
        mol.add_atom(8, 0.0, 0.0, 0.0);           // O
        mol.add_atom(1, 0.0, 0.757, 0.587);       // H
        mol.add_atom(1, 0.0, -0.757, 0.587);      // H
        
        std::cout << "Molecule: H2O\n";
        std::cout << "Geometry (Angstrom):\n";
        std::cout << "  O   0.000   0.000   0.000\n";
        std::cout << "  H   0.000   0.757   0.587\n";
        std::cout << "  H   0.000  -0.757   0.587\n\n";
        
        // Load basis set
        std::string basis_name = "cc-pvdz";
        BasisSet basis(basis_name, mol);
        
        std::cout << "Basis set: " << basis_name << "\n";
        std::cout << "Basis functions: " << basis.n_basis_functions() << "\n\n";
        
        // Create integral engine
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        // Step 1: Run RHF first
        std::cout << "=== Step 1: RHF Calculation ===\n";
        // NOTE: This assumes RHF class exists and works
        // If not, we'll need to run it separately
        
        // For now, let's assume we have RHF result
        // In real test, you'd call: RHF rhf(mol, basis, integrals);
        //                            auto rhf_result = rhf.compute();
        
        std::cout << "TODO: Run RHF calculation\n";
        std::cout << "(For now, this is a compilation test)\n\n";
        
        // Step 2: Run RMP2
        std::cout << "=== Step 2: RMP2 Calculation ===\n";
        // RMP2 rmp2(rhf_result, basis, integrals);
        // auto result = rmp2.compute();
        
        // Expected results for validation:
        double expected_rhf = -76.026;
        double expected_mp2_corr = -0.204;
        double expected_total = expected_rhf + expected_mp2_corr;
        
        std::cout << "\n=== Expected Results (Psi4 reference) ===\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RHF energy:        " << expected_rhf << " Ha\n";
        std::cout << "MP2 correlation:   " << expected_mp2_corr << " Ha\n";
        std::cout << "Total RMP2 energy: " << expected_total << " Ha\n";
        
        std::cout << "\n✓ RMP2 test compiled successfully\n";
        std::cout << "TODO: Complete integration with RHF for full test\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
