/**
 * Quick diagnostic: Validate MP energy hierarchy
 * 
 * Check that |E^(n+1)| < |E^(n)| for all orders
 */

#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include "mshqc/mp/ump4.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "\n====================================\n";
    std::cout << "  Quick MP Hierarchy Diagnostic\n";
    std::cout << "  System: Li (doublet)\n";
    std::cout << "====================================\n\n";
    
    // Setup Li atom
    Molecule mol;
    mol.add_atom("Li", 0.0, 0.0, 0.0);
    
    BasisSet basis("cc-pVDZ", mol, "spherical");
    auto integrals = std::make_shared<IntegralEngine>(mol, basis);
    
    // UHF
    std::cout << "Running UHF...\n";
    SCFConfig config;
    config.max_iter = 100;
    config.conv_threshold = 1e-8;
    
    UHF uhf(mol, basis, integrals, 2, 1, config);  // 2 alpha, 1 beta for Li
    auto uhf_result = uhf.compute();
    
    // UMP2
    std::cout << "Running UMP2...\n";
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // UMP3
    std::cout << "Running UMP3...\n";
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    // UMP4
    std::cout << "Running UMP4(SDQ)...\n";
    mp::UMP4 ump4(uhf_result, ump3_result, basis, integrals);
    auto ump4_result = ump4.compute(false);  // Skip triples for speed
    
    // Print results
    std::cout << "\n====================================\n";
    std::cout << "  MP ENERGY HIERARCHY CHECK\n";
    std::cout << "====================================\n\n";
    
    std::cout << std::fixed << std::setprecision(10);
    
    double e2 = ump2_result.e_corr_total;
    double e3 = ump3_result.e_mp3;
    double e4_sdq = ump4_result.e_mp4_sdq;
    
    std::cout << "E^(2) (MP2): " << std::setw(16) << e2 << " Ha\n";
    std::cout << "E^(3) (MP3): " << std::setw(16) << e3 << " Ha\n";
    std::cout << "E^(4) (MP4): " << std::setw(16) << e4_sdq << " Ha\n\n";
    
    // Check hierarchy
    bool e2_negative = (e2 < 0);
    bool e3_negative = (e3 < 0);
    bool e4_negative = (e4_sdq < 0);
    
    bool e3_smaller = (std::abs(e3) < std::abs(e2));
    bool e4_smaller = (std::abs(e4_sdq) < std::abs(e3));
    
    std::cout << "VALIDATION:\n";
    std::cout << "  E^(2) negative: " << (e2_negative ? "✓" : "✗") << "\n";
    std::cout << "  E^(3) negative: " << (e3_negative ? "✓" : "✗") << "\n";
    std::cout << "  E^(4) negative: " << (e4_negative ? "✓" : "✗") << "\n\n";
    
    std::cout << "  |E^(3)| < |E^(2)|: " << (e3_smaller ? "✓" : "✗") << "\n";
    std::cout << "  |E^(4)| < |E^(3)|: " << (e4_smaller ? "✓" : "✗") << "\n\n";
    
    if (e2_negative && e3_negative && e4_negative && e3_smaller && e4_smaller) {
        std::cout << "✓ MP HIERARCHY CORRECT!\n";
        return 0;
    } else {
        std::cout << "✗ MP HIERARCHY VIOLATED!\n";
        return 1;
    }
}
