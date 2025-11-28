/**
 * @file test_fci_vs_cipsi.cc
 * @brief Direct comparison: FCI vs CIPSI on SAME system
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-17
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include "mshqc/integrals.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/cipsi.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;
using namespace mshqc::ci;

int main() {
    std::cout << "\n=== FCI vs CIPSI Direct Comparison ===\n";
    std::cout << "System: Li / STO-3G\n\n";
    
    // Setup
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);
    BasisSet basis("STO-3G", mol);
    int nbf = basis.n_basis_functions();
    int n_alpha = 2, n_beta = 1;
    
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // UHF
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF) = " << uhf_result.energy_total << " Ha\n\n";
    
    // Transform integrals - USE FOCK MATRICES (not bare Hamiltonian!)
    // Fock already includes HF mean-field correction
    Eigen::MatrixXd f_mo_alpha = uhf_result.C_alpha.transpose() * uhf_result.F_alpha * uhf_result.C_alpha;
    Eigen::MatrixXd f_mo_beta = uhf_result.C_beta.transpose() * uhf_result.F_beta * uhf_result.C_beta;
    
    auto eri_ao = integrals->compute_eri();
    
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
    
    Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_aa, eri_aaaa);
    ci::build_same_spin_antisym_from_chemist(eri_mo_chemist_bb, eri_bbbb);
    ci::build_alpha_beta_from_chemist(eri_mo_chemist_ab, eri_aabb);
    
    CIIntegrals ci_ints;
    ci_ints.h_alpha = f_mo_alpha;  // Use Fock, not bare H!
    ci_ints.h_beta = f_mo_beta;    // Use Fock, not bare H!
    ci_ints.eri_aaaa = eri_aaaa;
    ci_ints.eri_bbbb = eri_bbbb;
    ci_ints.eri_aabb = eri_aabb;
    ci_ints.e_nuc = uhf_result.energy_nuclear;
    ci_ints.use_fock = true;  // CRITICAL: Use Fock-based CI!
    
    // Run FCI
    std::cout << "Running FCI...\n";
    FCI fci(ci_ints, nbf, n_alpha, n_beta);
    auto fci_result = fci.compute();
    
    std::cout << "\n";
    std::cout << "E(FCI) = " << fci_result.e_fci << " Ha\n";
    std::cout << "N(FCI dets) = " << fci_result.n_determinants << "\n\n";
    
    // Run CIPSI
    std::cout << "Running CIPSI...\n";
    CIPSIConfig cipsi_config;
    cipsi_config.e_pt2_threshold = 1.0e-5;
    cipsi_config.max_determinants = 100;
    cipsi_config.max_iterations = 10;
    cipsi_config.n_select_per_iter = 10;
    cipsi_config.start_from_hf = true;
    cipsi_config.include_singles = true;
    cipsi_config.include_doubles = false;
    cipsi_config.max_excitation_level = 2;
    cipsi_config.use_epstein_nesbet = true;
    cipsi_config.verbose = true;
    
    CIPSI cipsi(ci_ints, nbf, n_alpha, n_beta, cipsi_config);
    auto cipsi_result = cipsi.compute();
    
    std::cout << "\n";
    std::cout << "E(CIPSI var) = " << cipsi_result.e_var << " Ha\n";
    std::cout << "E(CIPSI PT2) = " << cipsi_result.e_pt2 << " Ha\n";
    std::cout << "E(CIPSI tot) = " << cipsi_result.e_total << " Ha\n";
    std::cout << "E_nuc used   = " << uhf_result.energy_nuclear << " Ha\n";
    std::cout << "N(CIPSI dets) = " << cipsi_result.n_selected << "\n\n";
    
    // Compare
    std::cout << "=== COMPARISON ===\n";
    std::cout << "E(UHF)   = " << uhf_result.energy_total << " Ha\n";
    std::cout << "E(FCI)   = " << fci_result.e_fci << " Ha\n";
    std::cout << "E(CIPSI) = " << cipsi_result.e_total << " Ha\n\n";
    
    double diff_fci_uhf = (fci_result.e_fci - uhf_result.energy_total) * 1000.0;
    double diff_cipsi_uhf = (cipsi_result.e_total - uhf_result.energy_total) * 1000.0;
    double diff_cipsi_fci = (cipsi_result.e_total - fci_result.e_fci) * 1000.0;
    
    std::cout << "FCI - UHF:   " << diff_fci_uhf << " mHa\n";
    std::cout << "CIPSI - UHF: " << diff_cipsi_uhf << " mHa\n";
    std::cout << "CIPSI - FCI: " << diff_cipsi_fci << " mHa\n\n";
    
    if (std::abs(diff_cipsi_fci) < 1.0) {
        std::cout << "✓ CIPSI matches FCI (< 1 mHa)!\n";
    } else {
        std::cout << "✗ CIPSI does NOT match FCI!\n";
    }
    
    return 0;
}
