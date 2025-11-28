/**
 * @file test_cipsi_simple.cc
 * @brief Simple CIPSI test - debug energy correctness
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include "mshqc/integrals.h"
#include "mshqc/ci/cipsi.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::ci;

int main() {
    std::cout << "\n=== CIPSI Simple Test: Lithium / STO-3G ===\n\n";
    
    // Setup Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Li: Z=3
    
    BasisSet basis("STO-3G", mol);
    int nbf = basis.n_basis_functions();
    int n_alpha = 2;
    int n_beta = 1;
    
    std::cout << "System: Li atom, " << nbf << " basis functions\n";
    std::cout << "Electrons: " << n_alpha << "α + " << n_beta << "β = 3 total\n\n";
    
    // Run UHF
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    SCFConfig scf_config;
    scf_config.max_iterations = 100;
    scf_config.energy_threshold = 1e-8;
    scf_config.density_threshold = 1e-6;
    scf_config.print_level = 0;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, scf_config);
    auto uhf_result = uhf.compute();
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF total)      = " << uhf_result.energy_total << " Ha\n";
    std::cout << "E(nuclear)        = " << uhf_result.energy_nuclear << " Ha\n";
    std::cout << "E(electronic UHF) = " << (uhf_result.energy_total - uhf_result.energy_nuclear) << " Ha\n\n";
    
    // Transform integrals for UHF → CI
    // CRITICAL: Use FOCK matrices, not bare Hamiltonian!
    // Theory: Fock includes HF mean-field correction
    std::cout << "Transforming Fock matrices to MO basis...\n";
    
    // Transform Fock matrices (NOT bare H!)
    Eigen::MatrixXd f_mo_alpha = uhf_result.C_alpha.transpose() * uhf_result.F_alpha * uhf_result.C_alpha;
    Eigen::MatrixXd f_mo_beta = uhf_result.C_beta.transpose() * uhf_result.F_beta * uhf_result.C_beta;
    
    auto eri_ao = integrals->compute_eri();
    
    // Transform 3 separate ERI tensors (αααα, ββββ, ααββ)
    Eigen::Tensor<double, 4> eri_mo_chemist_aa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_chemist_bb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_chemist_ab(nbf, nbf, nbf, nbf);
    eri_mo_chemist_aa.setZero();
    eri_mo_chemist_bb.setZero();
    eri_mo_chemist_ab.setZero();
    
    for (int p = 0; p < nbf; ++p)
        for (int q = 0; q < nbf; ++q)
            for (int r = 0; r < nbf; ++r)
                for (int s = 0; s < nbf; ++s)
                    for (int i = 0; i < nbf; ++i)
                        for (int j = 0; j < nbf; ++j)
                            for (int k = 0; k < nbf; ++k)
                                for (int l = 0; l < nbf; ++l) {
                                    double val = eri_ao(i,j,k,l);
                                    eri_mo_chemist_aa(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                                   uhf_result.C_alpha(k,r) * uhf_result.C_alpha(l,s) * val;
                                    eri_mo_chemist_bb(p,q,r,s) += uhf_result.C_beta(i,p) * uhf_result.C_beta(j,q) *
                                                                   uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                    eri_mo_chemist_ab(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                                   uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                }
    
    // Antisymmetrize
    Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_aa, eri_aaaa);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_bb, eri_bbbb);
    ci::build_alpha_beta_from_chemist(eri_mo_chemist_ab, eri_aabb);
    
    std::cout << "Done.\n\n";
    
    // Setup CIIntegrals with Fock matrices
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = f_mo_alpha;  // Use Fock, not bare H!
    ci_ints.h_beta = f_mo_beta;    // Use Fock, not bare H!
    ci_ints.eri_aaaa = eri_aaaa;
    ci_ints.eri_bbbb = eri_bbbb;
    ci_ints.eri_aabb = eri_aabb;
    ci_ints.e_nuc = uhf_result.energy_nuclear;
    ci_ints.use_fock = true;  // CRITICAL: Enable Fock-based CI!
    
    // Run CIPSI with new CIIntegrals interface
    std::cout << "Running CIPSI (Fock-based)...\n";
    CIPSIConfig config;
    config.e_pt2_threshold = 1.0e-5;
    config.max_determinants = 100;
    config.max_iterations = 10;
    config.n_select_per_iter = 10;
    config.start_from_hf = true;
    config.include_singles = true;
    config.include_doubles = false;
    config.max_excitation_level = 2;
    config.use_epstein_nesbet = true;
    config.verbose = true;  // Enable to see iteration details
    
    // CIPSI now uses CIIntegrals
    CIPSI cipsi(ci_ints, nbf, n_alpha, n_beta, config);
    auto result = cipsi.compute();
    
    std::cout << "\n=== CIPSI Results (Fock-based) ===\n";
    std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Iterations: " << result.n_iterations << "\n";
    std::cout << "Determinants: " << result.n_selected << "\n";
    std::cout << "E(variational) = " << std::setprecision(8) << result.e_var << " Ha (includes E_nuc)\n";
    std::cout << "E(PT2)         = " << std::scientific << result.e_pt2 << " Ha\n";
    std::cout << "E(total)       = " << std::fixed << std::setprecision(8) << result.e_total << " Ha\n\n";
    
    std::cout << "=== Comparison ===\n";
    std::cout << "E(CIPSI) = " << result.e_total << " Ha\n";
    std::cout << "E(UHF)   = " << uhf_result.energy_total << " Ha\n";
    std::cout << "Difference = " << (result.e_total - uhf_result.energy_total) * 1000.0 << " mHa\n\n";
    
    std::cout << "Test complete.\n";
    
    return 0;
}
