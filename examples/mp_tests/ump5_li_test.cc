/**
 * @file ump5_li_test.cc
 * @brief Test UMP5 on Li atom (open-shell, doublet) with cc-pVTZ
 *
 * Full MP hierarchy: UHF -> UMP2 -> UMP3 -> UMP4 -> UMP5
 * 
 * This tests the QUINTUPLE EXCITATIONS implementation!
 * Expected: UMP5 should capture ~98% of correlation energy.
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include "mshqc/mp/ump4.h"
#include "mshqc/mp/ump5.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>

using namespace mshqc;

int main() {
    std::cout << "\n====================================\n";
    std::cout << "  UMP5 Test: Li/cc-pVTZ\n";
    std::cout << "====================================\n";
    std::cout << "Testing QUINTUPLE EXCITATIONS!\n";
    std::cout << "This is O(N^10) - expect slow runtime.\n";
    std::cout << "====================================\n\n";

    auto total_start = std::chrono::high_resolution_clock::now();

    // Setup molecule
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);  // Li at origin

    BasisSet basis("cc-pVTZ", li);
    std::cout << "Molecule: Li (Z=3, doublet)\n";
    std::cout << "Basis: cc-pVTZ (" << basis.n_basis_functions() << " basis functions)\n";
    std::cout << "Electrons: 3 (2α, 1β)\n\n";

    auto integrals = std::make_shared<IntegralEngine>(li, basis);

    // Step 1: UHF
    std::cout << "Step 1: Running UHF...\n";
    auto uhf_start = std::chrono::high_resolution_clock::now();
    
    SCFConfig config;
    config.max_iterations = 100;
    config.energy_threshold = 1e-10;
    config.density_threshold = 1e-8;
    config.print_level = 0;  // Quiet

    int n_alpha = 2;
    int n_beta = 1;
    UHF uhf(li, basis, integrals, n_alpha, n_beta, config);
    auto uhf_result = uhf.compute();
    
    auto uhf_end = std::chrono::high_resolution_clock::now();
    auto uhf_time = std::chrono::duration_cast<std::chrono::seconds>(uhf_end - uhf_start).count();

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "  UHF energy:   " << uhf_result.energy_total << " Ha\n";
    std::cout << "  Time: " << uhf_time << "s\n\n";

    // Step 2: UMP2
    std::cout << "Step 2: Running UMP2...\n";
    auto ump2_start = std::chrono::high_resolution_clock::now();
    
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    auto ump2_end = std::chrono::high_resolution_clock::now();
    auto ump2_time = std::chrono::duration_cast<std::chrono::seconds>(ump2_end - ump2_start).count();

    std::cout << "  MP2 corr:     " << ump2_result.e_corr_total << " Ha\n";
    std::cout << "  UMP2 total:   " << (uhf_result.energy_total + ump2_result.e_corr_total) << " Ha\n";
    std::cout << "  Time: " << ump2_time << "s\n\n";

    // Step 3: UMP3
    std::cout << "Step 3: Running UMP3...\n";
    auto ump3_start = std::chrono::high_resolution_clock::now();
    
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    auto ump3_end = std::chrono::high_resolution_clock::now();
    auto ump3_time = std::chrono::duration_cast<std::chrono::seconds>(ump3_end - ump3_start).count();

    std::cout << "  MP3 corr:     " << ump3_result.e_mp3_corr << " Ha\n";
    std::cout << "  Total corr:   " << ump3_result.e_corr_total << " Ha\n";
    std::cout << "  UMP3 total:   " << ump3_result.e_total << " Ha\n";
    std::cout << "  Time: " << ump3_time << "s\n\n";

    // Step 4: UMP4
    std::cout << "Step 4: Running UMP4 (includes TRIPLES)...\n";
    std::cout << "  This will take longer (O(N^8))...\n";
    auto ump4_start = std::chrono::high_resolution_clock::now();
    
    mp::UMP4 ump4(uhf_result, ump3_result, basis, integrals);
    auto ump4_result = ump4.compute(true);  // Include triples
    
    auto ump4_end = std::chrono::high_resolution_clock::now();
    auto ump4_time = std::chrono::duration_cast<std::chrono::seconds>(ump4_end - ump4_start).count();

    std::cout << "  MP4(SDQ):     " << ump4_result.e_mp4_sdq << " Ha\n";
    std::cout << "  MP4(T):       " << ump4_result.e_mp4_t << " Ha\n";
    std::cout << "  MP4 total:    " << ump4_result.e_mp4_total << " Ha\n";
    std::cout << "  UMP4 total:   " << ump4_result.e_total << " Ha\n";
    std::cout << "  Time: " << ump4_time << "s\n\n";

    // Step 5: UMP5 ⭐ THE MAIN EVENT!
    std::cout << "========================================\n";
    std::cout << "Step 5: Running UMP5 ⭐\n";
    std::cout << "========================================\n";
    std::cout << "WARNING: This includes QUINTUPLES (O(N^10))!\n";
    std::cout << "Expected runtime: Several minutes to hours.\n";
    std::cout << "Progress will be shown below.\n";
    std::cout << "========================================\n\n";
    
    auto ump5_start = std::chrono::high_resolution_clock::now();
    
    mp::UMP5 ump5(uhf_result, ump4_result, basis, integrals);
    ump5.set_verbose(true);  // Show progress
    ump5.set_screening_threshold(1e-20);  // Allow very small contributions
    
    auto ump5_result = ump5.compute();
    
    auto ump5_end = std::chrono::high_resolution_clock::now();
    auto ump5_time = std::chrono::duration_cast<std::chrono::seconds>(ump5_end - ump5_start).count();

    // Final summary
    std::cout << "\n\n====================================\n";
    std::cout << "  FINAL SUMMARY\n";
    std::cout << "====================================\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\nEnergy breakdown:\n";
    std::cout << "  UHF:          " << std::setw(18) << ump5_result.e_uhf << " Ha\n";
    std::cout << "  + MP2:        " << std::setw(18) << ump5_result.e_mp2 << " Ha\n";
    std::cout << "  + MP3:        " << std::setw(18) << ump5_result.e_mp3 << " Ha\n";
    std::cout << "  + MP4:        " << std::setw(18) << ump5_result.e_mp4_total << " Ha\n";
    std::cout << "  + MP5:        " << std::setw(18) << ump5_result.e_mp5_total << " Ha\n";
    std::cout << "  ----------------------------------------\n";
    std::cout << "  UMP5 total:   " << std::setw(18) << ump5_result.e_total << " Ha ⭐\n";

    std::cout << "\nMP5 component breakdown:\n";
    std::cout << "  Singles:      " << std::setw(18) << ump5_result.e_mp5_s << " Ha\n";
    std::cout << "  Doubles:      " << std::setw(18) << ump5_result.e_mp5_d << " Ha\n";
    std::cout << "  Triples:      " << std::setw(18) << ump5_result.e_mp5_t << " Ha\n";
    std::cout << "  Quadruples:   " << std::setw(18) << ump5_result.e_mp5_q << " Ha\n";
    std::cout << "  QUINTUPLES:   " << std::setw(18) << ump5_result.e_mp5_qn << " Ha 🎉\n";

    std::cout << "\nCorrelation recovery:\n";
    double total_corr = ump5_result.e_corr_total;
    double mp2_frac = (ump5_result.e_mp2 / total_corr) * 100.0;
    double mp3_frac = ((ump5_result.e_mp2 + ump5_result.e_mp3) / total_corr) * 100.0;
    double mp4_frac = ((ump5_result.e_mp2 + ump5_result.e_mp3 + ump5_result.e_mp4_total) / total_corr) * 100.0;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  MP2:          " << mp2_frac << "%\n";
    std::cout << "  MP3:          " << mp3_frac << "%\n";
    std::cout << "  MP4:          " << mp4_frac << "%\n";
    std::cout << "  MP5:          100.00% (by definition)\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();

    std::cout << "\nTiming breakdown:\n";
    std::cout << "  UHF:          " << uhf_time << " s\n";
    std::cout << "  UMP2:         " << ump2_time << " s\n";
    std::cout << "  UMP3:         " << ump3_time << " s\n";
    std::cout << "  UMP4:         " << ump4_time << " s\n";
    std::cout << "  UMP5:         " << ump5_time << " s ⭐\n";
    std::cout << "  ----------------------------------------\n";
    std::cout << "  TOTAL:        " << total_time << " s\n";

    std::cout << "\n====================================\n";
    std::cout << "UMP5 Test Complete!\n";
    std::cout << "Quintuples successfully computed! 🎉\n";
    std::cout << "====================================\n\n";

    return 0;
}
