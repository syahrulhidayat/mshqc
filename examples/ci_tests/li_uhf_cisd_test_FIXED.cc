/**
 * @file li_uhf_cisd_test.cc
 * @brief Lithium: UHF vs CISD comparison
 * 
 * Test Li ground state with both UHF and CISD
 * Compare energies to see correlation effect
 * 
 * @author Muhamad Syahrul Hidayat  
 * @date 2025-11-14
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/determinant.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace mshqc;

int main() {
    std::cout << "============================================\n";
    std::cout << "  Li Ground State: UHF vs CISD\n";
    std::cout << "  Basis: cc-pVDZ\n";
    std::cout << "============================================\n\n";
    
    // Li atom
    Molecule mol;
    mol.add_atom(3, 0.0, 0.0, 0.0);  // Z=3
    
    std::cout << "Li atom: 3 electrons (2α, 1β)\n";
    std::cout << "Ground state: ²S (1s² 2s¹)\n\n";
    
    // cc-pVDZ basis (compromise: better than STO-3G, faster than cc-pVTZ)
    BasisSet basis("cc-pVDZ", mol);
    int nbf = basis.n_basis_functions();
    std::cout << "Basis: cc-pVDZ (" << nbf << " functions)\n\n";
    
    // Integrals
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // ========================================
    // Step 1: UHF Calculation
    // ========================================
    std::cout << "Step 1: UHF Calculation\n";
    std::cout << "----------------------------------------\n";
    
    int n_alpha = 2;
    int n_beta = 1;
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;  // Quiet
    
    UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    if (!uhf_result.converged) {
        std::cerr << "ERROR: UHF did not converge!\n";
        return 1;
    }
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "E(UHF) = " << uhf_result.energy_total << " Ha\n";
    std::cout << "Converged in " << uhf_result.iterations << " iterations\n\n";
    
    // ========================================
    // Step 2: Transform integrals to MO basis
    // ========================================
    std::cout << "Step 2: Transform integrals to MO basis\n";
    std::cout << "----------------------------------------\n";
    
    // One-electron: h = T + V (bare Hamiltonian)
    auto T = integrals->compute_kinetic();
    auto V = integrals->compute_nuclear();
    Eigen::MatrixXd h_ao = T + V;
    
    Eigen::MatrixXd h_mo_alpha = uhf_result.C_alpha.transpose() * h_ao * uhf_result.C_alpha;
    Eigen::MatrixXd h_mo_beta = uhf_result.C_beta.transpose() * h_ao * uhf_result.C_beta;
    
    std::cout << "Transforming one-electron integrals to MO basis... done.\n";
    
    
    // Two-electron: ERI
    auto eri_ao = integrals->compute_eri();
    
    // Transform ERIs (simple but slow O(N^9) algorithm)
    std::cout << "Transforming ERIs for " << nbf << " basis functions...\n";
    
    Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
    
    eri_aaaa.setZero();
    eri_bbbb.setZero();
    eri_aabb.setZero();
    
    // Full 4-index transformation
    // Step 1: Transform to MO basis (pq|rs)
    Eigen::Tensor<double, 4> eri_mo_pqrs_aa(nbf, nbf, nbf, nbf);
    Eigen::Tensor<double, 4> eri_mo_pqrs_bb(nbf, nbf, nbf, nbf);
    eri_mo_pqrs_aa.setZero();
    eri_mo_pqrs_bb.setZero();
    
    for (int p = 0; p < nbf; p++) {
        for (int q = 0; q < nbf; q++) {
            for (int r = 0; r < nbf; r++) {
                for (int s = 0; s < nbf; s++) {
                    for (int i = 0; i < nbf; i++) {
                        for (int j = 0; j < nbf; j++) {
                            for (int k = 0; k < nbf; k++) {
                                for (int l = 0; l < nbf; l++) {
                                    double val = eri_ao(i, j, k, l);
                                    
                                    // (pq|rs) in MO basis
                                    eri_mo_pqrs_aa(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                                uhf_result.C_alpha(k,r) * uhf_result.C_alpha(l,s) * val;
                                    
                                    eri_mo_pqrs_bb(p,q,r,s) += uhf_result.C_beta(i,p) * uhf_result.C_beta(j,q) *
                                                                uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                    
                                    // Alpha-beta
                                    eri_aabb(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                         uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    // Step 2: Antisymmetrize for same-spin
    // CRITICAL FIX 2025-11-16 (AI Agent 2):
    // Conversion: (ij|ab)_chemist = <ia|jb>_physicist
    // Therefore: <ij||ab>_phys = <ij|ab> - <ij|ba> = (ia|jb)_chem - (ib|ja)_chem
    // REFERENCE: Szabo & Ostlund (1996), Appendix A
    for (int p = 0; p < nbf; p++) {
        for (int q = 0; q < nbf; q++) {
            for (int r = 0; r < nbf; r++) {
                for (int s = 0; s < nbf; s++) {
                    // Correct antisymmetrization with proper index mapping:
                    // eri_mo_pqrs(p,q,r,s) stores (pq|rs)_chemist
                    // <pq||rs>_phys = (pr|qs)_chem - (ps|qr)_chem
                    eri_aaaa(p,q,r,s) = eri_mo_pqrs_aa(p,r,q,s) - eri_mo_pqrs_aa(p,s,q,r);
                    eri_bbbb(p,q,r,s) = eri_mo_pqrs_bb(p,r,q,s) - eri_mo_pqrs_bb(p,s,q,r);
                }
            }
        }
    }
    
    std::cout << "ERI transformation complete.\n\n";
    
    // ========================================
    // Step 3: CISD Calculation
    // ========================================
    std::cout << "Step 3: CISD Calculation\n";
    std::cout << "----------------------------------------\n";
    
    // Setup CI integrals (bare H formulation - standard approach)
    ci::CIIntegrals ci_ints;
    ci_ints.h_alpha = h_mo_alpha;  // Bare Hamiltonian (T + V)
    ci_ints.h_beta = h_mo_beta;    // Bare Hamiltonian
    ci_ints.eri_aaaa = eri_aaaa;
    ci_ints.eri_bbbb = eri_bbbb;
    ci_ints.eri_aabb = eri_aabb;
    ci_ints.e_nuc = uhf_result.energy_nuclear;
    ci_ints.use_fock = false;  // Use bare H + ERIs (standard CI)
    
    std::cout << "Using standard CI with bare Hamiltonian (T+V) + ERIs.\n\n";
    
    // HF determinant: orbitals 0,1 (alpha), 0 (beta)
    ci::Determinant hf_det(std::vector<int>{0, 1}, std::vector<int>{0});
    
    std::cout << "HF reference: " << hf_det.to_string() << "\n";
    std::cout << "  α: " << hf_det.n_alpha() << " electrons in orbitals 0,1\n";
    std::cout << "  β: " << hf_det.n_beta() << " electron in orbital 0\n\n";
    
    // CISD space
    int n_occ_a = 2;
    int n_occ_b = 1;
    int n_virt_a = nbf - n_occ_a;
    int n_virt_b = nbf - n_occ_b;
    
    std::cout << "CISD space: " << n_occ_a << "α, " << n_occ_b 
              << "β occupied; " << n_virt_a << "α, " << n_virt_b << "β virtual\n\n";
    
    ci::CISD cisd(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
    
    std::cout << "Running CISD...\n";
    auto cisd_result = cisd.compute();
    
    std::cout << "E(CISD) = " << cisd_result.e_cisd << " Ha\n";
    std::cout << "Number of determinants: " << cisd_result.n_determinants << "\n";
    std::cout << "Converged: " << (cisd_result.converged ? "YES" : "NO") << "\n\n";
    
    // ========================================
    // Step 4: Comparison
    // ========================================
    std::cout << "========================================\n";
    std::cout << "  Results Summary\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Method     Energy (Ha)       \n";
    std::cout << "--------   ------------------\n";
    std::cout << "UHF        " << std::setw(16) << uhf_result.energy_total << "\n";
    std::cout << "CISD       " << std::setw(16) << cisd_result.e_cisd << "\n";
    std::cout << "--------   ------------------\n";
    
    double delta_e = cisd_result.e_cisd - uhf_result.energy_total;
    double corr_e = cisd_result.e_corr;
    
    std::cout << "ΔE         " << std::setw(16) << delta_e << " Ha\n";
    std::cout << "E_corr     " << std::setw(16) << corr_e << " Ha\n\n";
    
    // Validation
    if (cisd_result.e_cisd < uhf_result.energy_total) {
        std::cout << "✓ CISD lower than UHF (correct!)\n";
        std::cout << "✓ CISD captures electron correlation\n";
    } else {
        std::cout << "✗ WARNING: CISD higher than UHF (unexpected!)\n";
    }
    
    std::cout << "\nReference (NIST): Li = -7.478 Ha (experimental)\n\n";
    
    std::cout << "============================================\n";
    std::cout << "  TEST COMPLETE\n";
    std::cout << "============================================\n";
    
    return 0;
}
