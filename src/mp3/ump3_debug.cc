/**
 * Debug version of compute_t2_ab_2nd with detailed printing
 */

#include "mshqc/ump3.h"
#include <iostream>
#include <iomanip>

namespace mshqc {

void UMP3::compute_t2_ab_2nd() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;
    
    // ========================================
    // STEP 1: PRINT DIMENSIONS
    // ========================================
    std::cout << "\n=== DEBUG: T2_ab^(2) Computation ===\n";
    std::cout << "Dimensions:\n";
    std::cout << "  nocc_a = " << nocc_a_ << ", nvir_a = " << nvir_a_ << "\n";
    std::cout << "  nocc_b = " << nocc_b_ << ", nvir_b = " << nvir_b_ << "\n";
    
    std::cout << "\nTensor dimensions:\n";
    std::cout << "  W_oooo_ab: (" 
              << W_oooo_ab_.dimension(0) << ", "
              << W_oooo_ab_.dimension(1) << ", "
              << W_oooo_ab_.dimension(2) << ", "
              << W_oooo_ab_.dimension(3) << ")\n";
    
    std::cout << "  W_vvvv_ab: ("
              << W_vvvv_ab_.dimension(0) << ", "
              << W_vvvv_ab_.dimension(1) << ", "
              << W_vvvv_ab_.dimension(2) << ", "
              << W_vvvv_ab_.dimension(3) << ")\n";
    
    std::cout << "  W_ovov_ab: ("
              << W_ovov_ab_.dimension(0) << ", "
              << W_ovov_ab_.dimension(1) << ", "
              << W_ovov_ab_.dimension(2) << ", "
              << W_ovov_ab_.dimension(3) << ")\n";
    
    std::cout << "  W_ovov_ba: ("
              << W_ovov_ba_.dimension(0) << ", "
              << W_ovov_ba_.dimension(1) << ", "
              << W_ovov_ba_.dimension(2) << ", "
              << W_ovov_ba_.dimension(3) << ")\n";
    
    // ========================================
    // STEP 2: SAMPLE VALUES
    // ========================================
    std::cout << "\nSample W values:\n";
    std::cout << "  W_oooo_ab(0,0,0,0) = " << W_oooo_ab_(0,0,0,0) << "\n";
    std::cout << "  W_oooo_ab(0,1,0,0) = " << W_oooo_ab_(0,1,0,0) << "\n";
    std::cout << "  W_vvvv_ab(0,0,0,0) = " << W_vvvv_ab_(0,0,0,0) << "\n";
    std::cout << "  W_ovov_ab(0,0,0,0) = " << W_ovov_ab_(0,0,0,0) << "\n";
    
    std::cout << "\nSample T2^(1) values:\n";
    std::cout << "  t2_ab_1(0,0,0,0) = " << t2_ab_1_(0,0,0,0) << "\n";
    std::cout << "  t2_ab_1(0,0,1,1) = " << t2_ab_1_(0,0,1,1) << "\n";
    
    // ========================================
    // STEP 3: COMPUTE WITH TERM BREAKDOWN
    // ========================================
    
    // Just compute first element with full breakdown
    int i = 0, j = 0, a = 0, b = 0;
    
    std::cout << "\n=== Computing t2_ab_2(" << i << "," << j << "," << a << "," << b << ") ===\n";
    
    double val = 0.0;
    double term1 = 0.0, term2 = 0.0, term3 = 0.0;
    double term4 = 0.0, term5 = 0.0, term6 = 0.0;
    
    // Term 1: HH Ladder
    std::cout << "\nTerm 1 (HH Ladder):\n";
    for (int k = 0; k < nocc_a_; ++k) {
        for (int l = 0; l < nocc_b_; ++l) {
            double contrib = W_oooo_ab_(k, i, l, j) * t2_ab_1_(k, l, a, b);
            std::cout << "  k=" << k << ", l=" << l 
                      << ": W(" << k << "," << i << "," << l << "," << j << ") = " 
                      << W_oooo_ab_(k, i, l, j)
                      << " × t2(" << k << "," << l << "," << a << "," << b << ") = "
                      << t2_ab_1_(k, l, a, b)
                      << " = " << contrib << "\n";
            term1 += contrib;
        }
    }
    std::cout << "  Term1 total: " << term1 << "\n";
    val += term1;
    
    // Term 2: PP Ladder
    std::cout << "\nTerm 2 (PP Ladder):\n";
    int count = 0;
    for (int c = 0; c < nvir_a_; ++c) {
        for (int d = 0; d < nvir_b_; ++d) {
            double contrib = W_vvvv_ab_(a, b, c, d) * t2_ab_1_(i, j, c, d);
            if (count < 5) {  // Print first 5
                std::cout << "  c=" << c << ", d=" << d 
                          << ": W×t2 = " << contrib << "\n";
            }
            term2 += contrib;
            count++;
        }
    }
    std::cout << "  Term2 total (from " << count << " terms): " << term2 << "\n";
    val += term2;
    
    // Term 3: PH alpha-alpha
    std::cout << "\nTerm 3 (PH α-α):\n";
    for (int m = 0; m < nocc_a_; ++m) {
        for (int e = 0; e < nvir_a_; ++e) {
            double contrib = -W_ovov_aa_(m, a, e, i) * t2_ab_1_(m, j, e, b);
            if (m == 0 && e < 3) {
                std::cout << "  m=" << m << ", e=" << e << ": contrib = " << contrib << "\n";
            }
            term3 += contrib;
        }
    }
    std::cout << "  Term3 total: " << term3 << "\n";
    val += term3;
    
    // Term 4: PH beta-beta
    std::cout << "\nTerm 4 (PH β-β):\n";
    for (int m = 0; m < nocc_b_; ++m) {
        for (int f = 0; f < nvir_b_; ++f) {
            double contrib = -W_ovov_bb_(m, b, f, j) * t2_ab_1_(i, m, a, f);
            if (f < 3) {
                std::cout << "  m=" << m << ", f=" << f << ": contrib = " << contrib << "\n";
            }
            term4 += contrib;
        }
    }
    std::cout << "  Term4 total: " << term4 << "\n";
    val += term4;
    
    // Term 5: PH cross αβ
    std::cout << "\nTerm 5 (PH cross α×β):\n";
    for (int m = 0; m < nocc_a_; ++m) {
        for (int e = 0; e < nvir_b_; ++e) {
            double contrib = -W_ovov_ab_(m, b, i, e) * t2_ab_1_(m, j, a, e);
            if (m == 0 && e < 3) {
                std::cout << "  m=" << m << ", e=" << e << ": contrib = " << contrib << "\n";
            }
            term5 += contrib;
        }
    }
    std::cout << "  Term5 total: " << term5 << "\n";
    val += term5;
    
    // Term 6: PH cross βα
    std::cout << "\nTerm 6 (PH cross β×α):\n";
    for (int m = 0; m < nocc_b_; ++m) {
        for (int f = 0; f < nvir_a_; ++f) {
            double contrib = -W_ovov_ba_(m, a, j, f) * t2_ab_1_(i, m, f, b);
            if (f < 3) {
                std::cout << "  m=" << m << ", f=" << f << ": contrib = " << contrib << "\n";
            }
            term6 += contrib;
        }
    }
    std::cout << "  Term6 total: " << term6 << "\n";
    val += term6;
    
    // Denominator
    double D = ea(i) + eb(j) - ea(nocc_a_ + a) - eb(nocc_b_ + b);
    std::cout << "\nDenominator:\n";
    std::cout << "  ea(" << i << ") = " << ea(i) << "\n";
    std::cout << "  eb(" << j << ") = " << eb(j) << "\n";
    std::cout << "  ea(vir " << a << ") = " << ea(nocc_a_ + a) << "\n";
    std::cout << "  eb(vir " << b << ") = " << eb(nocc_b_ + b) << "\n";
    std::cout << "  D = " << D << "\n";
    
    std::cout << "\nFinal:\n";
    std::cout << "  Numerator = " << val << "\n";
    std::cout << "  t2_ab_2(0,0,0,0) = " << (val / D) << "\n";
    
    std::cout << "\n=== Term Summary ===\n";
    std::cout << "  Term1 (HH):     " << std::setw(15) << term1 << "\n";
    std::cout << "  Term2 (PP):     " << std::setw(15) << term2 << "\n";
    std::cout << "  Term3 (PH αα):  " << std::setw(15) << term3 << "\n";
    std::cout << "  Term4 (PH ββ):  " << std::setw(15) << term4 << "\n";
    std::cout << "  Term5 (PH αβ):  " << std::setw(15) << term5 << "\n";
    std::cout << "  Term6 (PH βα):  " << std::setw(15) << term6 << "\n";
    std::cout << "  --------------------------------\n";
    std::cout << "  Total:          " << std::setw(15) << val << "\n";
    std::cout << "  / D:            " << std::setw(15) << (val/D) << "\n";
    
    // ========================================
    // STEP 4: NOW DO FULL COMPUTATION
    // ========================================
    std::cout << "\n=== Computing all elements... ===\n";
    
    for (int i = 0; i < nocc_a_; ++i) {
        for (int j = 0; j < nocc_b_; ++j) {
            for (int a = 0; a < nvir_a_; ++a) {
                for (int b = 0; b < nvir_b_; ++b) {
                    double val = 0.0;
                    
                    // Term 1: HH Ladder
                    for (int k = 0; k < nocc_a_; ++k) {
                        for (int l = 0; l < nocc_b_; ++l) {
                            val += W_oooo_ab_(k, i, l, j) * t2_ab_1_(k, l, a, b);
                        }
                    }
                    
                    // Term 2: PP Ladder
                    for (int c = 0; c < nvir_a_; ++c) {
                        for (int d = 0; d < nvir_b_; ++d) {
                            val += W_vvvv_ab_(a, b, c, d) * t2_ab_1_(i, j, c, d);
                        }
                    }
                    
                    // Term 3
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int e = 0; e < nvir_a_; ++e) {
                            val -= W_ovov_aa_(m, a, e, i) * t2_ab_1_(m, j, e, b);
                        }
                    }
                    
                    // Term 4
                    for (int m = 0; m < nocc_b_; ++m) {
                        for (int f = 0; f < nvir_b_; ++f) {
                            val -= W_ovov_bb_(m, b, f, j) * t2_ab_1_(i, m, a, f);
                        }
                    }
                    
                    // Term 5
                    for (int m = 0; m < nocc_a_; ++m) {
                        for (int e = 0; e < nvir_b_; ++e) {
                            val -= W_ovov_ab_(m, b, i, e) * t2_ab_1_(m, j, a, e);
                        }
                    }
                    
                    // Term 6
                    for (int m = 0; m < nocc_b_; ++m) {
                        for (int f = 0; f < nvir_a_; ++f) {
                            val -= W_ovov_ba_(m, a, j, f) * t2_ab_1_(i, m, f, b);
                        }
                    }
                    
                    double D = ea(i) + eb(j) - ea(nocc_a_ + a) - eb(nocc_b_ + b);
                    t2_ab_2_(i, j, a, b) = val / D;
                }
            }
        }
    }
    
    std::cout << "Done!\n";
}

} // namespace mshqc