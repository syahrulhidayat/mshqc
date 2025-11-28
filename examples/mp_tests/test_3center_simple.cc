/**
 * @file test_3center_simple.cc
 * @brief Simple test to verify 3-center ERIs are computed correctly
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "\n======================================================================\n";
    std::cout << "3-Center ERI Test: H2 molecule\n";
    std::cout << "======================================================================\n\n";
    
    // H2 molecule
    Molecule mol;
    mol.add_atom(1, 0.0, 0.0, 0.0);
    mol.add_atom(1, 0.0, 0.0, 1.4);  // 1.4 bohr ~= 0.74 Angstrom
    
    // STO-3G (minimal)
    BasisSet basis("sto-3g", mol);
    BasisSet aux_basis("cc-pvdz-ri", mol);
    
    std::cout << "Molecule: H2 (1.4 bohr)\n";
    std::cout << "Primary basis: " << basis.name() << " (" << basis.n_basis_functions() << " functions)\n";
    std::cout << "Auxiliary basis: " << aux_basis.name() << " (" << aux_basis.n_basis_functions() << " functions)\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // Compute 3-center ERI (μν|P)
    std::cout << "Computing 3-center ERIs (μν|P)...\n";
    auto B = integrals->compute_3center_eri(aux_basis);
    std::cout << "  Tensor shape: [" << basis.n_basis_functions() << ", " 
              << basis.n_basis_functions() << ", " << aux_basis.n_basis_functions() << "]\n\n";
    
    // Print some sample values
    std::cout << "Sample 3-center ERI values:\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "  (0,0|0) = " << B(0, 0, 0) << "\n";
    std::cout << "  (0,1|0) = " << B(0, 1, 0) << "\n";
    std::cout << "  (1,1|0) = " << B(1, 1, 0) << "\n";
    std::cout << "  (0,0|1) = " << B(0, 0, 1) << "\n\n";
    
    // Verify symmetry: (μν|P) = (νμ|P)
    std::cout << "Symmetry check: (μν|P) = (νμ|P)\n";
    int n = basis.n_basis_functions();
    int naux = aux_basis.n_basis_functions();
    double max_asym = 0.0;
    for (int mu = 0; mu < n; mu++) {
        for (int nu = mu+1; nu < n; nu++) {
            for (int P = 0; P < naux; P++) {
                double diff = std::abs(B(mu, nu, P) - B(nu, mu, P));
                max_asym = std::max(max_asym, diff);
            }
        }
    }
    std::cout << "  Max asymmetry: " << std::scientific << max_asym << std::fixed << "\n";
    if (max_asym < 1e-10) {
        std::cout << "  ✓ PASS: 3-center ERIs symmetric\n";
    } else {
        std::cout << "  ❌ FAIL: Symmetry violation\n";
    }
    
    std::cout << "\n======================================================================\n";
    
    return 0;
}
