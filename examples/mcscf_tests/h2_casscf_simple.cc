// ============================================================================
// H2 Molecule CASSCF Test - Simplest Ab Initio Test
// CAS(2,2): 2 electrons in 2 orbitals = Full CI in minimal space
// ============================================================================
// Expected: E(CASSCF) should equal E(FCI) for this case
// Reference: E(FCI/STO-3G) ≈ -1.117 Ha at R=0.74 Å
// ============================================================================

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace mshqc;

int main() {
    std::cout << "\n";
    std::cout << "======================================================================\n";
    std::cout << "H2 Molecule: CASSCF/FCI Test (STO-3G)\n";
    std::cout << "======================================================================\n\n";

    // H2 molecule at equilibrium (R = 0.74 Å = 1.4 bohr)
    Molecule mol;
    double R = 1.4;  // bohr
    mol.add_atom(1, 0.0, 0.0, 0.0);      // H at origin
    mol.add_atom(1, 0.0, 0.0, R);        // H along z-axis
    
    std::cout << "Geometry: H--H bond length = " << R << " bohr\n";
    std::cout << "Nuclear repulsion: " << mol.nuclear_repulsion_energy() << " Ha\n\n";

    // Basis: STO-3G (minimal basis, 1 function per H → 2 total)
    const std::string basis_dir = "/home/shared/project-mshqc/data/basis";
    BasisSet basis("sto-3g", mol, basis_dir);
    
    std::cout << "Basis: STO-3G\n";
    std::cout << "Functions: " << basis.n_basis_functions() << "\n\n";

    // Integral engine
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);

    // RHF reference (H2: 2 electrons, spin-restricted)
    SCFConfig cfg;
    cfg.max_iterations = 100;
    cfg.energy_threshold = 1e-10;
    cfg.density_threshold = 1e-8;

    RHF rhf(mol, basis, integrals, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto rhf_result = rhf.compute();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto rhf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    if (!rhf_result.converged) {
        std::cerr << "ERROR: RHF did not converge\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "RHF:\n";
    std::cout << "  E(RHF)   = " << rhf_result.energy_total << " Ha\n";
    std::cout << "  iters    = " << rhf_result.iterations << ", time = " << rhf_ms.count() << " ms\n\n";

    // Active space: CAS(2,2) - ALL electrons, ALL orbitals
    // This is FULL CI for H2 in STO-3G basis
    int nbf = static_cast<int>(basis.n_basis_functions());
    mcscf::ActiveSpace cas = mcscf::ActiveSpace::CAS(
        /*n_elec=*/2,       // Both H electrons in active space
        /*n_orb=*/2,        // Both orbitals in active space
        /*n_total_orb=*/nbf,
        /*n_total_elec=*/2
    );

    std::cout << "Active space: " << cas.to_string() << "\n";
    std::cout << "Expected determinants: 6 (singlet FCI for 2e in 2 orbitals)\n\n";

    mcscf::CASSCF casscf(mol, basis, integrals, cas);
    casscf.set_max_iterations(10);
    casscf.set_energy_threshold(1e-10);
    casscf.set_gradient_threshold(1e-6);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto cas_result = casscf.compute(rhf_result);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto casscf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

    std::cout << "\n";
    std::cout << "======================================================================\n";
    std::cout << "Results\n";
    std::cout << "======================================================================\n";
    std::cout << "RHF energy:     " << rhf_result.energy_total << " Ha\n";
    std::cout << "CASSCF energy:  " << cas_result.e_casscf << " Ha\n";
    std::cout << "Iterations:     " << cas_result.n_iterations << "\n";
    std::cout << "Time:           " << casscf_ms.count() << " ms\n";
    if (cas_result.n_determinants > 0) {
        std::cout << "Determinants:   " << cas_result.n_determinants << "\n";
    }
    std::cout << "Converged:      " << (cas_result.converged ? "Yes" : "No") << "\n";
    std::cout << "======================================================================\n\n";

    // Energy comparison
    double dE = cas_result.e_casscf - rhf_result.energy_total;
    double corr = dE;
    
    std::cout << "Analysis:\n";
    std::cout << "  Correlation energy: " << std::scientific << std::setprecision(6) 
              << corr << " Ha\n";
    std::cout << "  ΔE = E(CASSCF) - E(RHF) = " << dE << " Ha\n";
    
    // Validation
    bool valid = true;
    
    // Check 1: CASSCF ≤ RHF (variational principle)
    if (dE > 1e-6) {
        std::cout << "  ✗ FAIL: E(CASSCF) > E(RHF) - violates variational principle!\n";
        valid = false;
    } else {
        std::cout << "  ✓ PASS: E(CASSCF) ≤ E(RHF)\n";
    }
    
    // Check 2: Energy in reasonable range
    if (cas_result.e_casscf < -1.2 || cas_result.e_casscf > -1.0) {
        std::cout << "  ✗ FAIL: Energy out of expected range [-1.2, -1.0] Ha\n";
        valid = false;
    } else {
        std::cout << "  ✓ PASS: Energy in reasonable range\n";
    }
    
    // Check 3: Correlation should be negative (stabilizing)
    if (corr > 0) {
        std::cout << "  ✗ FAIL: Positive correlation energy (unphysical)\n";
        valid = false;
    } else {
        std::cout << "  ✓ PASS: Negative correlation energy\n";
    }
    
    // Check 4: For CAS(2,2) = FCI, correlation should be ~10-50 mHa
    double corr_mha = std::abs(corr * 1000);
    if (corr_mha < 5 || corr_mha > 100) {
        std::cout << "  ⚠ WARNING: Correlation " << std::fixed << std::setprecision(2) 
                  << corr_mha << " mHa seems unusual\n";
    } else {
        std::cout << "  ✓ PASS: Correlation magnitude reasonable\n";
    }
    
    std::cout << "\n";
    if (valid) {
        std::cout << "======================================================================\n";
        std::cout << "✓ ALL VALIDATION CHECKS PASSED\n";
        std::cout << "======================================================================\n";
        return 0;
    } else {
        std::cout << "======================================================================\n";
        std::cout << "✗ SOME VALIDATION CHECKS FAILED\n";
        std::cout << "======================================================================\n";
        return 1;
    }
}
