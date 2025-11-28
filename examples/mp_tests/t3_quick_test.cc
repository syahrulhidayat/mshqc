/**
 * @file t3_quick_test.cc
 * @brief Quick test for T3^(2) with Li/STO-3G (small basis)
 */

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include <iostream>
#include <iomanip>

using namespace mshqc;

int main() {
    std::cout << "========================================\n";
    std::cout << "  T3^(2) Quick Test: Li/STO-3G\n";
    std::cout << "========================================\n\n";
    
    Molecule li;
    li.add_atom(3, 0.0, 0.0, 0.0);  // Li
    
    BasisSet basis("STO-3G", li);
    std::cout << "Basis: STO-3G (" << basis.n_basis_functions() << " functions)\n\n";
    
    auto integrals = std::make_shared<IntegralEngine>(li, basis);
    
    // UHF
    SCFConfig config;
    config.max_iterations = 50;
    config.energy_threshold = 1e-8;
    config.print_level = 0;
    
    UHF uhf(li, basis, integrals, 2, 1, config);  // 2α, 1β
    auto uhf_result = uhf.compute();
    
    std::cout << "UHF energy: " << std::fixed << std::setprecision(10) 
              << uhf_result.energy_total << " Ha\n\n";
    
    // UMP2
    UMP2 ump2(uhf_result, basis, integrals);
    auto ump2_result = ump2.compute();
    
    // UMP3 with T3^(2)
    UMP3 ump3(uhf_result, ump2_result, basis, integrals);
    auto ump3_result = ump3.compute();
    
    std::cout << "\n========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "T3^(2) computed: " << (ump3_result.t3_2_computed ? "YES ✓" : "NO") << "\n";
    std::cout << "T3_ααα size: " << ump3_result.t3_aaa_2.size() << " elements\n";
    std::cout << "T3_βββ size: " << ump3_result.t3_bbb_2.size() << " elements\n";
    
    // Calculate norms
    auto norm6 = [](const Eigen::Tensor<double,6>& T) {
        if(T.size() == 0) return 0.0;
        double s = 0.0;
        for (int i=0;i<T.dimension(0);++i)
          for (int j=0;j<T.dimension(1);++j)
            for (int k=0;k<T.dimension(2);++k)
              for (int a=0;a<T.dimension(3);++a)
                for (int b=0;b<T.dimension(4);++b)
                  for (int c=0;c<T.dimension(5);++c)
                    s += T(i,j,k,a,b,c)*T(i,j,k,a,b,c);
        return std::sqrt(s);
    };
    
    double t3_norm_aaa = norm6(ump3_result.t3_aaa_2);
    double t3_norm_bbb = norm6(ump3_result.t3_bbb_2);
    
    std::cout << "\n||T3^(2)|| ααα = " << t3_norm_aaa << "\n";
    std::cout << "||T3^(2)|| βββ = " << t3_norm_bbb << "\n";
    
    std::cout << "\n========================================\n";
    if(t3_norm_aaa > 1e-10) {
        std::cout << "✓ SUCCESS: T3^(2) is NON-ZERO!\n";
        std::cout << "Full Ψ^(4) wavefunction is now available.\n";
    } else {
        std::cout << "✗ FAIL: T3^(2) is zero\n";
    }
    std::cout << "========================================\n";
    
    return 0;
}
