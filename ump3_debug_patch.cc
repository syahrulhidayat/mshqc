/**
 * Tambahkan di ump3.cc untuk debugging
 * Insert setelah compute_t2_ab_2nd() dimulai
 */

void UMP3::compute_t2_ab_2nd() {
    const auto& ea = uhf_.orbital_energies_alpha;
    const auto& eb = uhf_.orbital_energies_beta;
    
    // ===== DEBUG: Print dimensi =====
    std::cout << "\n=== DEBUG T2_ab^(2) ===\n";
    std::cout << "nocc_a = " << nocc_a_ << ", nocc_b = " << nocc_b_ << "\n";
    std::cout << "nvir_a = " << nvir_a_ << ", nvir_b = " << nvir_b_ << "\n";
    std::cout << "W_oooo_ab dimensions: " 
              << W_oooo_ab_.dimension(0) << " × "
              << W_oooo_ab_.dimension(1) << " × "
              << W_oooo_ab_.dimension(2) << " × "
              << W_oooo_ab_.dimension(3) << "\n";
    
    // ===== DEBUG: Sample W values =====
    std::cout << "\nSample W_oooo_ab values:\n";
    for (int m = 0; m < std::min(2, (int)W_oooo_ab_.dimension(0)); ++m) {
        for (int n = 0; n < std::min(2, (int)W_oooo_ab_.dimension(1)); ++n) {
            for (int i = 0; i < std::min(2, (int)W_oooo_ab_.dimension(2)); ++i) {
                for (int j = 0; j < std::min(2, (int)W_oooo_ab_.dimension(3)); ++j) {
                    std::cout << "W(" << m << "," << n << "," << i << "," << j << ") = "
                              << W_oooo_ab_(m, n, i, j) << "\n";
                }
            }
        }
    }
    
    // ===== DEBUG: Compute first amplitude with detailed trace =====
    int i = 0, j = 0, a = 0, b = 0;
    
    std::cout << "\n=== Computing T2(" << i << "," << j << "," << a << "," << b << ") ===\n";
    
    double val_hh = 0.0, val_pp = 0.0, val_ph = 0.0;
    
    // HH Ladder
    std::cout << "\nHH Ladder term:\n";
    for (int k = 0; k < nocc_a_; ++k) {
        for (int l = 0; l < nocc_b_; ++l) {
            double w = W_oooo_ab_(k, i, l, j);
            double t2 = t2_ab_1_(k, l, a, b);
            double contrib = w * t2;
            val_hh += contrib;
            
            std::cout << "  k=" << k << ", l=" << l 
                      << ": W(" << k << "," << i << "," << l << "," << j << ") = " << w
                      << ", T2(" << k << "," << l << "," << a << "," << b << ") = " << t2
                      << ", contrib = " << contrib << "\n";
        }
    }
    std::cout << "HH total: " << val_hh << "\n";
    
    // PP Ladder
    std::cout << "\nPP Ladder term:\n";
    for (int c = 0; c < nvir_a_; ++c) {
        for (int d = 0; d < nvir_b_; ++d) {
            double w = W_vvvv_ab_(a, b, c, d);
            double t2 = t2_ab_1_(i, j, c, d);
            double contrib = w * t2;
            val_pp += contrib;
            
            if (c < 2 && d < 2) {
                std::cout << "  c=" << c << ", d=" << d
                          << ": W(" << a << "," << b << "," << c << "," << d << ") = " << w
                          << ", T2(" << i << "," << j << "," << c << "," << d << ") = " << t2
                          << ", contrib = " << contrib << "\n";
            }
        }
    }
    std::cout << "PP total: " << val_pp << " (showing first 4 terms)\n";
    
    // PH Terms
    std::cout << "\nPH Exchange terms:\n";
    
    // Term 3: α-α
    double val_ph_aa = 0.0;
    for (int m = 0; m < nocc_a_; ++m) {
        for (int e = 0; e < nvir_a_; ++e) {
            double w = W_ovov_aa_(m, a, e, i);
            double t2 = t2_ab_1_(m, j, e, b);
            double contrib = -w * t2;
            val_ph_aa += contrib;
            
            if (m == 0 && e < 2) {
                std::cout << "  PH_aa: m=" << m << ", e=" << e
                          << ": -W(" << m << "," << a << "," << e << "," << i << ") * T2("
                          << m << "," << j << "," << e << "," << b << ") = " << contrib << "\n";
            }
        }
    }
    std::cout << "PH_aa total: " << val_ph_aa << "\n";
    
    // Term 4: β-β
    double val_ph_bb = 0.0;
    for (int m = 0; m < nocc_b_; ++m) {
        for (int f = 0; f < nvir_b_; ++f) {
            double w = W_ovov_bb_(m, b, f, j);
            double t2 = t2_ab_1_(i, m, a, f);
            double contrib = -w * t2;
            val_ph_bb += contrib;
            
            if (m == 0 && f < 2) {
                std::cout << "  PH_bb: m=" << m << ", f=" << f
                          << ": -W(" << m << "," << b << "," << f << "," << j << ") * T2("
                          << i << "," << m << "," << a << "," << f << ") = " << contrib << "\n";
            }
        }
    }
    std::cout << "PH_bb total: " << val_ph_bb << "\n";
    
    // Term 5: α-β cross
    double val_ph_ab = 0.0;
    for (int m = 0; m < nocc_a_; ++m) {
        for (int e = 0; e < nvir_b_; ++e) {
            double w = W_ovov_ab_(m, b, i, e);
            double t2 = t2_ab_1_(m, j, a, e);
            double contrib = -w * t2;
            val_ph_ab += contrib;
            
            if (m == 0 && e < 2) {
                std::cout << "  PH_ab: m=" << m << ", e=" << e
                          << ": -W(" << m << "," << b << "," << i << "," << e << ") * T2("
                          << m << "," << j << "," << a << "," << e << ") = " << contrib << "\n";
            }
        }
    }
    std::cout << "PH_ab total: " << val_ph_ab << "\n";
    
    // Term 6: β-α cross
    double val_ph_ba = 0.0;
    for (int m = 0; m < nocc_b_; ++m) {
        for (int f = 0; f < nvir_a_; ++f) {
            double w = W_ovov_ba_(m, a, j, f);
            double t2 = t2_ab_1_(i, m, f, b);
            double contrib = -w * t2;
            val_ph_ba += contrib;
            
            if (m == 0 && f < 2) {
                std::cout << "  PH_ba: m=" << m << ", f=" << f
                          << ": -W(" << m << "," << a << "," << j << "," << f << ") * T2("
                          << i << "," << m << "," << f << "," << b << ") = " << contrib << "\n";
            }
        }
    }
    std::cout << "PH_ba total: " << val_ph_ba << "\n";
    
    val_ph = val_ph_aa + val_ph_bb + val_ph_ab + val_ph_ba;
    
    double D = ea(i) + eb(j) - ea(nocc_a_ + a) - eb(nocc_b_ + b);
    double val_total = val_hh + val_pp + val_ph;
    
    std::cout << "\n=== Summary for T2(0,0,0,0) ===\n";
    std::cout << "HH contribution:     " << val_hh << "\n";
    std::cout << "PP contribution:     " << val_pp << "\n";
    std::cout << "PH contribution:     " << val_ph << "\n";
    std::cout << "  (PH_aa: " << val_ph_aa << ")\n";
    std::cout << "  (PH_bb: " << val_ph_bb << ")\n";
    std::cout << "  (PH_ab: " << val_ph_ab << ")\n";
    std::cout << "  (PH_ba: " << val_ph_ba << ")\n";
    std::cout << "Total numerator:     " << val_total << "\n";
    std::cout << "Denominator:         " << D << "\n";
    std::cout << "T2^(2)(0,0,0,0):     " << (val_total / D) << "\n";
    
    // Check if any term is abnormally large
    if (std::abs(val_hh) > 1.0) std::cout << "⚠ WARNING: HH term very large!\n";
    if (std::abs(val_pp) > 1.0) std::cout << "⚠ WARNING: PP term very large!\n";
    if (std::abs(val_ph) > 1.0) std::cout << "⚠ WARNING: PH term very large!\n";
    
    std::cout << "\n=== Continuing full calculation... ===\n";
    
    // [Rest of the original function continues here...]
}
