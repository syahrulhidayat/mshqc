/**
 * @file test_natural_orbitals.cc
 * @brief Test natural orbital analysis for Li FCI wavefunction
 * 
 * SYSTEM:
 *   - Li atom (3 electrons: 2α + 1β)
 *   - STO-3G basis (5 functions)
 *   - UHF reference → FCI wavefunction
 * 
 * TEST:
 *   - Build 1-RDM from CI coefficients
 *   - Diagonalize to get natural orbitals + occupations
 *   - Verify sum rule: Σ n_i = N_electrons
 *   - Print correlation measure
 * 
 * EXPECTED:
 *   - Occupation numbers: 0 ≤ n_i ≤ 2
 *   - Total: n_α + n_β = 3 electrons
 *   - Fractional occupations indicate correlation
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/natural_orbitals.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "============================================\n";
    std::cout << "  Natural Orbital Analysis Test\n";
    std::cout << "  System: Li / STO-3G\n";
    std::cout << "============================================\n\n";
    
    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Z=3
    
    std::cout << "System: Li atom (3 electrons)\n";
    std::cout << "  Configuration: 2α + 1β\n";
    std::cout << "  Ground state: ²S (doublet)\n\n";
    
    // STO-3G basis
    BasisSet basis("STO-3G", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "Basis: STO-3G (" << nbf << " functions)\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // ========================================
    // Step 1: UHF Calculation
    // ========================================
    std::cout << "Step 1: UHF Reference\n";
    std::cout << "========================================\n";
    
    int n_alpha = 2;
    int n_beta = 1;
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    config.print_level = 0;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    if (!uhf_result.converged) {
        std::cerr << "ERROR: UHF did not converge!\n";
        return 1;
    }
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF) = " << uhf_result.energy_total << " Ha\n\n";
    
    // ========================================
    // Step 2: Transform integrals to MO basis
    // ========================================
    std::cout << "Step 2: MO Integral Transformation\n";
    std::cout << "========================================\n";
    
    auto T = integrals->compute_kinetic();
    auto V = integrals->compute_nuclear();
    Eigen::MatrixXd h_ao = T + V;
    
    Eigen::MatrixXd h_mo_alpha = uhf_result.C_alpha.transpose() * h_ao * uhf_result.C_alpha;
    Eigen::MatrixXd h_mo_beta = uhf_result.C_beta.transpose() * h_ao * uhf_result.C_beta;
    
    auto eri_ao = integrals->compute_eri();
    
    // Transform ERI
    Eigen::Tensor<double, 4> eri_mo_chemist_aa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_chemist_bb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_chemist_ab(nbf, nbf, nbf, nbf);
    eri_mo_chemist_aa.setZero();
    eri_mo_chemist_bb.setZero();
    eri_mo_chemist_ab.setZero();
    
    for (int p = 0; p < nbf; p++) {
        for (int q = 0; q < nbf; q++) {
            for (int r = 0; r < nbf; r++) {
                for (int s = 0; s < nbf; s++) {
                    for (int mu = 0; mu < nbf; mu++) {
                        for (int nu = 0; nu < nbf; nu++) {
                            for (int lam = 0; lam < nbf; lam++) {
                                for (int sig = 0; sig < nbf; sig++) {
                                    double val = eri_ao(mu, nu, lam, sig);
                                    
                                    eri_mo_chemist_aa(p,q,r,s) += 
                                        uhf_result.C_alpha(mu,p) * uhf_result.C_alpha(nu,q) *
                                        uhf_result.C_alpha(lam,r) * uhf_result.C_alpha(sig,s) * val;
                                    
                                    eri_mo_chemist_bb(p,q,r,s) += 
                                        uhf_result.C_beta(mu,p) * uhf_result.C_beta(nu,q) *
                                        uhf_result.C_beta(lam,r) * uhf_result.C_beta(sig,s) * val;
                                    
                                    eri_mo_chemist_ab(p,q,r,s) += 
                                        uhf_result.C_alpha(mu,p) * uhf_result.C_alpha(nu,q) *
                                        uhf_result.C_beta(lam,r) * uhf_result.C_beta(sig,s) * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
    
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_aa, eri_aaaa);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_bb, eri_bbbb);
    ci::build_alpha_beta_from_chemist(eri_mo_chemist_ab, eri_aabb);
    
    std::cout << "ERI transformation complete.\n\n";
    
    // Setup CI integrals
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = h_mo_alpha;
    ci_ints.h_beta = h_mo_beta;
    ci_ints.eri_aaaa = eri_aaaa;
    ci_ints.eri_bbbb = eri_bbbb;
    ci_ints.eri_aabb = eri_aabb;
    ci_ints.e_nuc = uhf_result.energy_nuclear;
    ci_ints.use_fock = false;
    
    // ========================================
    // Step 3: FCI Calculation
    // ========================================
    std::cout << "Step 3: FCI Computation\n";
    std::cout << "========================================\n";
    
    ci::FCI fci(ci_ints, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    std::cout << "E(FCI) = " << std::setw(14) << fci_result.e_fci << " Ha\n";
    std::cout << "E_corr = " << std::setw(14) << fci_result.e_corr << " Ha\n";
    std::cout << "N_det  = " << fci_result.n_determinants << "\n\n";
    
    // ========================================
    // Step 4: Natural Orbital Analysis
    // ========================================
    std::cout << "Step 4: Natural Orbital Analysis\n";
    std::cout << "========================================\n";
    
    ci::NaturalOrbitalAnalysis no_analysis(fci_result.determinants, fci_result.coefficients);
    auto no_result = no_analysis.compute(nbf);
    
    // Print results
    no_result.print_summary();
    no_result.print_occupations(nbf);  // Print all orbitals
    
    // ========================================
    // Step 5: Validation
    // ========================================
    std::cout << "Step 5: Validation\n";
    std::cout << "========================================\n";
    
    double total_elec_alpha = no_result.total_occupation_alpha;
    double total_elec_beta = no_result.total_occupation_beta;
    double total_elec = total_elec_alpha + total_elec_beta;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nSum rule check:\n";
    std::cout << "  Expected: 3 electrons (2α + 1β)\n";
    std::cout << "  Computed: " << total_elec << " electrons\n";
    std::cout << "            (" << total_elec_alpha << "α + " << total_elec_beta << "β)\n";
    
    double error = std::abs(total_elec - 3.0);
    std::cout << "  Error:    " << error << "\n";
    
    bool passed = (error < 1e-6);
    
    if (passed) {
        std::cout << "\n✅ PASSED: Sum rule satisfied (error < 1e-6)\n";
    } else {
        std::cerr << "\n❌ FAILED: Sum rule violated!\n";
    }
    
    std::cout << "\n============================================\n";
    std::cout << "  Natural Orbital Test Complete\n";
    std::cout << "============================================\n";
    
    return passed ? 0 : 1;
}
