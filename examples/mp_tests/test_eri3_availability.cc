/**
 * @file test_eri3_availability.cc
 * @brief Test if PSI4's libint2 supports 3-center ERIs (xs_xx)
 * 
 * @author Muhamad Syahrul Hidayat
 * @date 2025-11-16
 */

#include <libint2.hpp>
#include <iostream>

int main() {
    libint2::initialize();
    
    std::cout << "=== Libint2 3-Center ERI Support Test ===\n\n";
    std::cout << "Libint version: " << LIBINT_VERSION << "\n";
    std::cout << "INCLUDE_ERI3 defined: ";
    
#ifdef INCLUDE_ERI3
    std::cout << "YES (level " << INCLUDE_ERI3 << ")\n";
#else
    std::cout << "NO\n";
#endif
    
    std::cout << "ERI3_MAX_AM_LIST: ";
#ifdef ERI3_MAX_AM_LIST
    std::cout << ERI3_MAX_AM_LIST << "\n";
#else
    std::cout << "undefined\n";
#endif
    
    std::cout << "\nBraKet types supported:\n";
    std::cout << "  x_x    (1-body): " << static_cast<int>(libint2::BraKet::x_x) << "\n";
    std::cout << "  xx_xx  (4-center): " << static_cast<int>(libint2::BraKet::xx_xx) << "\n";
    std::cout << "  xs_xx  (3-center): " << static_cast<int>(libint2::BraKet::xs_xx) << "\n";
    std::cout << "  xx_xs  (3-center): " << static_cast<int>(libint2::BraKet::xx_xs) << "\n";
    std::cout << "  xs_xs  (2-center): " << static_cast<int>(libint2::BraKet::xs_xs) << "\n";
    
    // Try to create 3-center ERI engine
    std::cout << "\nAttempting to create 3-center ERI engine...\n";
    
    try {
        // Try default construction first
        libint2::Engine engine(libint2::Operator::coulomb, 
                              4,  // max nprim
                              4,  // max l
                              0); // deriv order
        
        // Check if supports 3-center
        std::cout << "✓ Default engine created\n";
        std::cout << "  Supports xs_xx: checking...\n";
        
        // Try to set to xs_xx mode
        engine.set(libint2::BraKet::xs_xx);
        std::cout << "✓ SUCCESS: Engine set to xs_xx (3-center) mode!\n";
        std::cout << "  Engine supports (μν|P) integrals\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << "\n";
    }
    
    libint2::finalize();
    
    std::cout << "\n===========================================\n";
    
    return 0;
}
