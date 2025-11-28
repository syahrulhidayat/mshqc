/**
 * @file test_cipsi_h2_sto3g.cc
 * @brief Test CIPSI (Selected CI) on H2/STO-3G system
 * 
 * This test demonstrates the CIPSI algorithm:
 * - Start with HF + singles (~7 determinants)
 * - Iteratively add most important determinants
 * - Converge to FCI limit with selected space
 * - Expected: 10-100× fewer determinants than full CI
 * 
 * Theory: Huron et al., J. Chem. Phys. 58, 5745 (1973)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/uhf.h"
#include "mshqc/integrals.h"
#include "mshqc/ci/cipsi.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::ci;

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "  CIPSI Test: H2 / STO-3G\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
    
    // Step 1: Setup H2 molecule
    std::cout << "Setting up H2 molecule...\n";
    Molecule mol;
    mol.add_atom("H", 0.0, 0.0, 0.0);
    mol.add_atom("H", 0.0, 0.0, 1.4);  // 1.4 Bohr ≈ 0.74 Å
    
    // Step 2: Load STO-3G basis
    std::cout << "Loading STO-3G basis...\n";
    BasisSet basis("STO-3G", mol);
    int nbf = basis.nbf();
    std::cout << "  Number of basis functions: " << nbf << "\n";
    std::cout << "\n";
    
    // Step 3: Run UHF (reference wavefunction)
    std::cout << "Running UHF...\n";
    UHF uhf(mol, basis);
    auto uhf_result = uhf.compute();
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "  E(UHF) = " << uhf_result.energy << " Ha\n";
    std::cout << "\n";
    
    // Step 4: Compute integrals in MO basis
    std::cout << "Computing integrals in MO basis...\n";
    
    // Core Hamiltonian
    Eigen::MatrixXd h_core_ao = compute_core_hamiltonian(mol, basis);
    Eigen::MatrixXd h_core_mo = uhf_result.C.transpose() * h_core_ao * uhf_result.C;
    
    // ERIs (use spin-averaged for simplicity)
    auto eri_ao = compute_eri_tensor(basis);
    Eigen::Tensor<double, 4> eri_mo = transform_eri_tensor_full(eri_ao, uhf_result.C);
    
    // Convert to physicist notation ⟨pq|rs⟩ for CI
    Eigen::Tensor<double, 4> eri_phys(nbf, nbf, nbf, nbf);
    for (int p = 0; p < nbf; ++p) {
        for (int q = 0; q < nbf; ++q) {
            for (int r = 0; r < nbf; ++r) {
                for (int s = 0; s < nbf; ++s) {
                    eri_phys(p, q, r, s) = eri_mo(p, r, q, s);  // (pr|qs) → ⟨pq|rs⟩
                }
            }
        }
    }
    
    std::cout << "  Integrals computed.\n";
    std::cout << "\n";
    
    // Step 5: Run FCI for exact reference
    std::cout << "Running FCI (exact reference)...\n";
    int n_alpha = 1;
    int n_beta = 1;
    
    FCI fci(eri_phys, h_core_mo, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    std::cout << "  E(FCI) = " << fci_result.energy << " Ha\n";
    std::cout << "  Number of FCI determinants: " << fci_result.determinants.size() << "\n";
    std::cout << "\n";
    
    // Step 6: Run CIPSI
    std::cout << "Running CIPSI (Selected CI)...\n";
    std::cout << "\n";
    
    CIPSIConfig config;
    config.e_pt2_threshold = 1.0e-5;        // Tight convergence (10 µHa)
    config.max_determinants = 1000;         // Allow up to 1000 dets
    config.max_iterations = 20;             // Max 20 iterations
    config.n_select_per_iter = 10;          // Add 10 dets per iteration
    config.start_from_hf = true;
    config.include_singles = true;
    config.include_doubles = false;         // Start with HF + singles only
    config.max_excitation_level = 2;        // Allow doubles in connected space
    config.use_epstein_nesbet = true;
    config.verbose = true;
    
    CIPSI cipsi(eri_phys, h_core_mo, nbf, n_alpha, n_beta, config);
    auto cipsi_result = cipsi.compute();
    
    // Step 7: Compare results
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "  RESULTS COMPARISON\n";
    std::cout << "================================================================================\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\nENERGIES:\n";
    std::cout << "  E(UHF)          = " << uhf_result.energy << " Ha\n";
    std::cout << "  E(FCI)          = " << fci_result.energy << " Ha (exact)\n";
    std::cout << "  E(CIPSI var)    = " << cipsi_result.e_var << " Ha\n";
    std::cout << "  E(CIPSI total)  = " << cipsi_result.e_total << " Ha (var + PT2)\n";
    std::cout << "\n";
    
    std::cout << "ERRORS vs FCI:\n";
    double error_cipsi_var = (cipsi_result.e_var - fci_result.energy) * 1e6;
    double error_cipsi_total = (cipsi_result.e_total - fci_result.energy) * 1e6;
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  CIPSI variational: " << error_cipsi_var << " µHa\n";
    std::cout << "  CIPSI total (PT2): " << error_cipsi_total << " µHa\n";
    std::cout << "\n";
    
    std::cout << "EFFICIENCY:\n";
    std::cout << "  FCI determinants:   " << fci_result.determinants.size() << "\n";
    std::cout << "  CIPSI determinants: " << cipsi_result.n_selected << "\n";
    double reduction = static_cast<double>(fci_result.determinants.size()) / 
                      static_cast<double>(cipsi_result.n_selected);
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Reduction factor:   " << reduction << "× fewer determinants\n";
    std::cout << "\n";
    
    std::cout << "CONVERGENCE:\n";
    std::cout << "  Iterations:    " << cipsi_result.n_iterations << "\n";
    std::cout << "  Converged:     " << (cipsi_result.converged ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  Reason:        " << cipsi_result.conv_reason << "\n";
    std::cout << "\n";
    
    // Step 8: Assessment
    std::cout << "ASSESSMENT:\n";
    bool success = true;
    
    // Check 1: CIPSI variational should be above FCI
    if (cipsi_result.e_var < fci_result.energy - 1e-8) {
        std::cout << "  ✗ FAILED: CIPSI variational below FCI (violates variational principle)\n";
        success = false;
    } else {
        std::cout << "  ✓ PASS: CIPSI variational ≥ FCI (variational principle satisfied)\n";
    }
    
    // Check 2: CIPSI total should be close to FCI
    if (std::abs(error_cipsi_total) < 100.0) {  // < 100 µHa
        std::cout << "  ✓ PASS: CIPSI total within 100 µHa of FCI\n";
    } else {
        std::cout << "  ⚠ WARNING: CIPSI total error > 100 µHa (may need more iterations)\n";
    }
    
    // Check 3: CIPSI should use fewer determinants
    if (cipsi_result.n_selected < fci_result.determinants.size()) {
        std::cout << "  ✓ PASS: CIPSI uses fewer determinants than FCI\n";
    } else {
        std::cout << "  ⚠ WARNING: CIPSI not more efficient than FCI\n";
    }
    
    // Check 4: PT2 correction should improve energy
    if (std::abs(error_cipsi_total) < std::abs(error_cipsi_var)) {
        std::cout << "  ✓ PASS: PT2 correction improves energy estimate\n";
    } else {
        std::cout << "  ⚠ WARNING: PT2 correction does not improve estimate\n";
    }
    
    std::cout << "\n";
    std::cout << "================================================================================\n";
    
    if (success) {
        std::cout << "  CIPSI TEST: SUCCESS ✓\n";
        std::cout << "================================================================================\n";
        std::cout << "\n";
        return 0;
    } else {
        std::cout << "  CIPSI TEST: FAILED ✗\n";
        std::cout << "================================================================================\n";
        std::cout << "\n";
        return 1;
    }
}
