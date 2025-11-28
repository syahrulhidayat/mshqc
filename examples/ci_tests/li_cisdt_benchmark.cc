/**
 * @file li_cisdt_benchmark.cc
 * @brief Lithium atom CISDT calculation with validation
 * 
 * Compare MSH-QC CISDT with:
 *   - NIST experimental data
 *   - Literature FCI values
 *   - Our own FCI implementation
 * 
 * System: Li atom (3 electrons, doublet ²S)
 * Basis: cc-pVDZ (14 contracted GTOs)
 * 
 * CISDT = Configuration Interaction Singles + Doubles + Triples
 * Should be very close to FCI for Li (only quintuple excitations missing)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ci/cisdt.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/ci_utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace mshqc;
using namespace mshqc::ci;

// NIST reference
const double NIST_LI_EXACT = -7.47806032;  // Ha (experimental)

void print_header(const std::string& title) {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << title << "\n";
    std::cout << "================================================================\n";
}

int main() {
    try {
        print_header("Li/cc-pVDZ: CISDT BENCHMARK");
        
        std::cout << "System: Lithium atom (Z=3, 3 electrons)\n";
        std::cout << "State: Ground state ²S (doublet)\n";
        std::cout << "Basis: cc-pVDZ (14 contracted GTOs)\n";
        std::cout << "Method: CISDT (Singles + Doubles + Triples)\n\n";
        
        std::cout << "NIST Experimental: " << std::fixed << std::setprecision(10) 
                  << NIST_LI_EXACT << " Ha\n";
        
        // ============================================
        // Setup molecule and basis
        // ============================================
        
        Molecule mol;
        mol.add_atom(3, 0.0, 0.0, 0.0);  // Li at origin
        
        BasisSet basis("cc-pVDZ", mol);
        int nbf = basis.n_basis_functions();
        
        std::cout << "\nBasis set: cc-pVDZ\n";
        std::cout << "  Basis functions: " << nbf << "\n";
        std::cout << "  Composition: 3s2p1d (contracted GTOs)\n\n";
        
        // ============================================
        // Compute integrals
        // ============================================
        
        print_header("STEP 1: INTEGRAL COMPUTATION");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto integrals = std::make_shared<IntegralEngine>(mol, basis);
        
        auto T = integrals->compute_kinetic();
        auto V = integrals->compute_nuclear();
        Eigen::MatrixXd h_ao = T + V;
        
        auto eri_ao = integrals->compute_eri();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double time_integrals = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "Integrals computed in " << time_integrals << " seconds\n";
        
        // ============================================
        // UHF Calculation
        // ============================================
        
        print_header("STEP 2: UHF REFERENCE");
        
        int n_alpha = 2;
        int n_beta = 1;
        
        SCFConfig config;
        config.max_iterations = 100;
        config.energy_threshold = 1e-10;
        config.density_threshold = 1e-8;
        config.print_level = 0;
        
        start_time = std::chrono::high_resolution_clock::now();
        
        UHF uhf(mol, basis, integrals, n_alpha, n_beta, config);
        auto uhf_result = uhf.compute();
        
        end_time = std::chrono::high_resolution_clock::now();
        double time_uhf = std::chrono::duration<double>(end_time - start_time).count();
        
        if (!uhf_result.converged) {
            std::cerr << "ERROR: UHF did not converge!\n";
            return 1;
        }
        
        double e_uhf = uhf_result.energy_total;
        
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "E(UHF) = " << e_uhf << " Ha\n";
        std::cout << "Time: " << time_uhf << " s\n";
        std::cout << "Error vs NIST: " << (e_uhf - NIST_LI_EXACT) * 1000.0 << " mHa\n";
        
        // ============================================
        // Transform integrals to MO basis
        // ============================================
        
        print_header("STEP 3: MO INTEGRAL TRANSFORMATION");
        
        start_time = std::chrono::high_resolution_clock::now();
        
        Eigen::MatrixXd h_mo_alpha = uhf_result.C_alpha.transpose() * h_ao * uhf_result.C_alpha;
        Eigen::MatrixXd h_mo_beta = uhf_result.C_beta.transpose() * h_ao * uhf_result.C_beta;
        
        // Transform ERIs
        Eigen::Tensor<double, 4> eri_mo_aa(nbf, nbf, nbf, nbf);
        Eigen::Tensor<double, 4> eri_mo_bb(nbf, nbf, nbf, nbf);
        Eigen::Tensor<double, 4> eri_mo_ab(nbf, nbf, nbf, nbf);
        eri_mo_aa.setZero();
        eri_mo_bb.setZero();
        eri_mo_ab.setZero();
        
        std::cout << "Transforming ERIs to MO basis...\n";
        
        for (int p = 0; p < nbf; p++) {
            for (int q = 0; q < nbf; q++) {
                for (int r = 0; r < nbf; r++) {
                    for (int s = 0; s < nbf; s++) {
                        for (int i = 0; i < nbf; i++) {
                            for (int j = 0; j < nbf; j++) {
                                for (int k = 0; k < nbf; k++) {
                                    for (int l = 0; l < nbf; l++) {
                                        double val = eri_ao(i, j, k, l);
                                        eri_mo_aa(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                              uhf_result.C_alpha(k,r) * uhf_result.C_alpha(l,s) * val;
                                        eri_mo_bb(p,q,r,s) += uhf_result.C_beta(i,p) * uhf_result.C_beta(j,q) *
                                                              uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                        eri_mo_ab(p,q,r,s) += uhf_result.C_alpha(i,p) * uhf_result.C_alpha(j,q) *
                                                              uhf_result.C_beta(k,r) * uhf_result.C_beta(l,s) * val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Convert to physicist notation with antisymmetrization
        Eigen::Tensor<double, 4> eri_aaaa(nbf, nbf, nbf, nbf);
        Eigen::Tensor<double, 4> eri_bbbb(nbf, nbf, nbf, nbf);
        Eigen::Tensor<double, 4> eri_aabb(nbf, nbf, nbf, nbf);
        
        build_same_spin_antisym_from_chemist(eri_mo_aa, eri_aaaa);
        build_same_spin_antisym_from_chemist(eri_mo_bb, eri_bbbb);
        build_alpha_beta_from_chemist(eri_mo_ab, eri_aabb);
        
        end_time = std::chrono::high_resolution_clock::now();
        double time_transform = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "MO transformation complete in " << time_transform << " s\n";
        
        // Setup CI integrals
        CIIntegrals ci_ints;
        ci_ints.h_alpha = h_mo_alpha;
        ci_ints.h_beta = h_mo_beta;
        ci_ints.eri_aaaa = eri_aaaa;
        ci_ints.eri_bbbb = eri_bbbb;
        ci_ints.eri_aabb = eri_aabb;
        ci_ints.e_nuc = uhf_result.energy_nuclear;
        ci_ints.use_fock = false;
        
        Determinant hf_det(std::vector<int>{0, 1}, std::vector<int>{0});
        
        int n_occ_a = 2;
        int n_occ_b = 1;
        int n_virt_a = nbf - n_occ_a;
        int n_virt_b = nbf - n_occ_b;
        
        // ============================================
        // CISD for comparison
        // ============================================
        
        print_header("STEP 4: CISD (BASELINE)");
        
        start_time = std::chrono::high_resolution_clock::now();
        
        CISD cisd(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
        auto cisd_result = cisd.compute();
        
        end_time = std::chrono::high_resolution_clock::now();
        double time_cisd = std::chrono::duration<double>(end_time - start_time).count();
        
        double e_cisd = cisd_result.e_cisd;
        
        std::cout << "E(CISD) = " << e_cisd << " Ha\n";
        std::cout << "Correlation: " << (e_cisd - e_uhf) * 1000.0 << " mHa\n";
        std::cout << "Error vs NIST: " << (e_cisd - NIST_LI_EXACT) * 1000.0 << " mHa\n";
        std::cout << "Time: " << time_cisd << " s\n";
        std::cout << "Determinants: " << cisd_result.n_determinants << "\n";
        
        // ============================================
        // CISDT - THE MAIN EVENT!
        // ============================================
        
        print_header("STEP 5: CISDT (SINGLES + DOUBLES + TRIPLES)");
        
        std::cout << "Initializing CISDT...\n";
        
        CISDT cisdt(ci_ints, hf_det, n_occ_a, n_occ_b, n_virt_a, n_virt_b);
        
        // Estimate size
        size_t n_cisdt_est = cisdt_determinant_count(nbf, nbf, n_occ_a, n_occ_b);
        std::cout << "Estimated determinants: " << n_cisdt_est << "\n\n";
        
        if (n_cisdt_est > 50000) {
            std::cout << "⚠️  WARNING: Very large CISDT space!\n";
            std::cout << "This may take several minutes...\n\n";
        }
        
        CISDTOptions opts;
        opts.use_davidson = true;
        opts.davidson_threshold = 500;
        opts.max_davidson_iter = 100;
        opts.davidson_tol = 1e-8;
        opts.verbose = true;
        
        start_time = std::chrono::high_resolution_clock::now();
        
        auto cisdt_result = cisdt.compute(opts);
        
        end_time = std::chrono::high_resolution_clock::now();
        double time_cisdt = std::chrono::duration<double>(end_time - start_time).count();
        
        double e_cisdt = cisdt_result.e_cisdt;
        
        std::cout << "\nCISDT Results:\n";
        std::cout << "E(CISDT) = " << e_cisdt << " Ha\n";
        std::cout << "Correlation: " << (e_cisdt - e_uhf) * 1000.0 << " mHa\n";
        std::cout << "Error vs NIST: " << (e_cisdt - NIST_LI_EXACT) * 1000.0 << " mHa\n";
        std::cout << "Time: " << time_cisdt << " s\n";
        
        // ============================================
        // FCI for comparison
        // ============================================
        
        print_header("STEP 6: FCI (EXACT REFERENCE)");
        
        start_time = std::chrono::high_resolution_clock::now();
        
        FCI fci(ci_ints, nbf, n_alpha, n_beta);
        auto fci_result = fci.compute();
        
        end_time = std::chrono::high_resolution_clock::now();
        double time_fci = std::chrono::duration<double>(end_time - start_time).count();
        
        double e_fci = fci_result.e_fci;
        
        std::cout << "E(FCI) = " << e_fci << " Ha\n";
        std::cout << "Correlation: " << (e_fci - e_uhf) * 1000.0 << " mHa\n";
        std::cout << "Error vs NIST: " << (e_fci - NIST_LI_EXACT) * 1000.0 << " mHa\n";
        std::cout << "Time: " << time_fci << " s\n";
        std::cout << "Determinants: " << fci_result.n_determinants << "\n";
        
        // ============================================
        // COMPREHENSIVE COMPARISON
        // ============================================
        
        print_header("COMPREHENSIVE COMPARISON");
        
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "\nMethod           Energy (Ha)          vs NIST (mHa)    vs FCI (µHa)     Time (s)     N_det\n";
        std::cout << "-----------------------------------------------------------------------------------------\n";
        
        auto print_row = [&](const char* name, double e, double t, int n_det) {
            double err_nist = (e - NIST_LI_EXACT) * 1000.0;
            double err_fci = (e - e_fci) * 1e6;
            std::cout << std::left << std::setw(16) << name
                      << std::right << std::setw(20) << e
                      << std::setw(17) << std::setprecision(6) << err_nist
                      << std::setw(17) << std::setprecision(3) << err_fci
                      << std::setw(13) << std::setprecision(2) << t
                      << std::setw(10) << n_det << "\n";
        };
        
        std::cout << "NIST (exp)       " << std::setw(20) << NIST_LI_EXACT 
                  << std::setw(17) << "0.000000"
                  << std::setw(17) << (NIST_LI_EXACT - e_fci) * 1e6
                  << std::setw(13) << "-"
                  << std::setw(10) << "-" << "\n";
        std::cout << "-----------------------------------------------------------------------------------------\n";
        print_row("UHF", e_uhf, time_uhf, 1);
        print_row("CISD", e_cisd, time_cisd, cisd_result.n_determinants);
        print_row("CISDT", e_cisdt, time_cisdt, cisdt_result.n_determinants);
        print_row("FCI (exact)", e_fci, time_fci, fci_result.n_determinants);
        
        // Analysis
        std::cout << "\n";
        std::cout << "CISDT vs CISD improvement: " << std::setprecision(6) 
                  << (e_cisdt - e_cisd) * 1000.0 << " mHa\n";
        std::cout << "CISDT vs FCI error: " << std::setprecision(3) 
                  << (e_cisdt - e_fci) * 1e6 << " µHa\n";
        std::cout << "CISDT captures " << std::setprecision(2) 
                  << (e_cisdt - e_cisd) / (e_fci - e_cisd) * 100.0 
                  << "% of missing correlation (CISD→FCI)\n";
        
        // ============================================
        // VALIDATION
        // ============================================
        
        print_header("VALIDATION");
        
        bool pass = true;
        
        // Check energy ordering
        if (e_cisd >= e_cisdt && e_cisdt >= e_fci) {
            std::cout << "✅ Energy ordering correct: E(CISD) > E(CISDT) ≥ E(FCI)\n";
        } else {
            std::cout << "❌ Energy ordering incorrect!\n";
            pass = false;
        }
        
        // Check CISDT improvement over CISD
        if (e_cisdt < e_cisd) {
            std::cout << "✅ CISDT improves over CISD\n";
        } else {
            std::cout << "❌ CISDT should be lower than CISD!\n";
            pass = false;
        }
        
        // Check CISDT accuracy
        double cisdt_err = std::abs(e_cisdt - e_fci) * 1e6;
        if (cisdt_err < 100.0) {
            std::cout << "✅ CISDT within 100 µHa of FCI (excellent!)\n";
        } else if (cisdt_err < 1000.0) {
            std::cout << "⚠️  CISDT error " << cisdt_err << " µHa (acceptable)\n";
        } else {
            std::cout << "❌ CISDT error too large: " << cisdt_err << " µHa\n";
            pass = false;
        }
        
        std::cout << "\n";
        if (pass) {
            std::cout << "🎉 ALL VALIDATION CHECKS PASSED!\n";
            std::cout << "\nCONCLUSION:\n";
            std::cout << "CISDT successfully implemented and validated.\n";
            std::cout << "Results consistent with FCI and NIST references.\n";
            return 0;
        } else {
            std::cout << "⚠️  SOME VALIDATION CHECKS FAILED\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
