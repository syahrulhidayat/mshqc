/**
 * @file test_li_ump3_ccpvtz.cc
 * @brief Full Li/cc-pVTZ UMP3 calculation
 * 
 * Large basis - T3 will use approximate mode (nbf >= 30)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>

using namespace mshqc;

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Li/cc-pVTZ Full UMP3 Calculation\n";
    std::cout << "========================================\n\n";

    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);

    std::cout << "System: Li atom (²S ground state, doublet)\n";
    std::cout << "Basis: cc-pVTZ\n";
    std::cout << "Method: UHF -> UMP2 -> UMP3\n\n";

    BasisSet basis("cc-pvtz", li, "data/basis");
    int nbf = basis.n_basis_functions();
    std::cout << "Basis functions: " << nbf << "\n";
    std::cout << "NOTE: Large basis, T3 will use approximate mode\n\n";

    auto integrals = std::make_shared<IntegralEngine>(li, basis);

    // UHF
    std::cout << "Running UHF...\n";
    auto uhf_start = std::chrono::high_resolution_clock::now();
    
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.density_threshold = 1e-6;
    config.print_level = 0;

    int n_alpha = 2;
    int n_beta = 1;
    
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    auto uhf_end = std::chrono::high_resolution_clock::now();
    auto uhf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(uhf_end - uhf_start).count();

    std::cout << "\nUHF Results:\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "  E(UHF) = " << uhf_result.energy_total << " Ha\n";
    std::cout << "  Time: " << uhf_ms << " ms\n\n";

    // UMP2
    std::cout << "Running UMP2...\n";
    auto ump2_start = std::chrono::high_resolution_clock::now();
    
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    auto ump2_end = std::chrono::high_resolution_clock::now();
    auto ump2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ump2_end - ump2_start).count();

    std::cout << "\nUMP2 Results:\n";
    std::cout << "  E(MP2) corr = " << ump2_result.e_corr_total << " Ha\n";
    std::cout << "  E(UMP2)     = " << uhf_result.energy_total + ump2_result.e_corr_total << " Ha\n";
    std::cout << "  Time: " << ump2_ms << " ms\n\n";

    // UMP3
    std::cout << "========================================\n";
    std::cout << "  Running UMP3 (cc-pVTZ)\n";
    std::cout << "========================================\n\n";
    
    std::cout << "WARNING: This may take several minutes due to T3!\n\n";
    
    auto ump3_start = std::chrono::high_resolution_clock::now();
    
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    auto ump3_end = std::chrono::high_resolution_clock::now();
    auto ump3_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ump3_end - ump3_start).count();

    // Final summary
    std::cout << "\n========================================\n";
    std::cout << "  FINAL ENERGY SUMMARY (cc-pVTZ)\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "E(UHF):      " << std::setw(16) << uhf_result.energy_total << " Ha\n";
    std::cout << "E(MP2) corr: " << std::setw(16) << ump2_result.e_corr_total << " Ha\n";
    std::cout << "E(MP3) corr: " << std::setw(16) << ump3_result.e_mp3 << " Ha\n";
    std::cout << "E(UMP2):     " << std::setw(16) << uhf_result.energy_total + ump2_result.e_corr_total << " Ha\n";
    std::cout << "E(UMP3):     " << std::setw(16) << ump3_result.e_total << " Ha\n\n";
    
    std::cout << "Timing:\n";
    std::cout << "  UHF:  " << std::setw(8) << uhf_ms << " ms\n";
    std::cout << "  UMP2: " << std::setw(8) << ump2_ms << " ms\n";
    std::cout << "  UMP3: " << std::setw(8) << ump3_ms << " ms\n";
    std::cout << "  Total: " << std::setw(7) << (uhf_ms + ump2_ms + ump3_ms) << " ms\n\n";

    // Energy comparison
    double mp2_correction = ump2_result.e_corr_total;
    double mp3_correction = ump3_result.e_mp3;
    double mp3_to_mp2_ratio = mp3_correction / mp2_correction;
    
    std::cout << "Analysis:\n";
    std::cout << "  E(MP2)/E(MP3) ratio: " << std::fixed << std::setprecision(4) 
              << mp3_to_mp2_ratio << "\n";
    std::cout << "  |E(MP3)| > |E(MP2)|: " 
              << (std::abs(mp3_correction) > std::abs(mp2_correction) ? "YES" : "NO") << "\n\n";

    std::cout << "✅ Full cc-pVTZ UMP3 calculation complete!\n\n";

    return 0;
}
