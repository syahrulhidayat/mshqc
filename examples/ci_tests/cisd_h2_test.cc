/**
 * @file cisd_h2_test.cc
 * @brief Test CISD on H2 molecule
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-12
 */

#include <iostream>
#include <iomanip>
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/slater_condon.h"
#include <Eigen/Dense>

using namespace mshqc::ci;

int main() {
    std::cout << "=== CISD Test: H2 Molecule (Minimal Basis) ===\n\n";
    
    // Minimal H2 setup: 2 electrons in 2 orbitals (STO-3G-like)
    // HF: |↑↓⟩ in orbital 0
    // Singles: |↑⟩₀|↓⟩₁, |↑⟩₁|↓⟩₀  
    // Doubles: |↑↓⟩ in orbital 1
    
    int n_orb = 2;
    
    // Mock integrals for H2 (rough approximation)
    CIIntegrals ints;
    
    // One-electron integrals (kinetic + nuclear attraction)
    ints.h_alpha = Eigen::MatrixXd::Zero(n_orb, n_orb);
    ints.h_beta = Eigen::MatrixXd::Zero(n_orb, n_orb);
    ints.h_alpha(0,0) = -1.12;  // Lower energy (bonding)
    ints.h_alpha(1,1) = -0.48;  // Higher energy (antibonding)
    ints.h_beta(0,0) = -1.12;
    ints.h_beta(1,1) = -0.48;
    
    // Two-electron repulsion integrals <ij||kl> (antisymmetrized)
    ints.eri_aaaa = Eigen::Tensor<double, 4>(n_orb, n_orb, n_orb, n_orb);
    ints.eri_bbbb = Eigen::Tensor<double, 4>(n_orb, n_orb, n_orb, n_orb);
    ints.eri_aabb = Eigen::Tensor<double, 4>(n_orb, n_orb, n_orb, n_orb);
    ints.eri_aaaa.setZero();
    ints.eri_bbbb.setZero();
    ints.eri_aabb.setZero();
    
    // Antisymmetrized integrals <ij||kl> = <ij|kl> - <ij|lk>
    // For same-spin (alpha-alpha)
    ints.eri_aaaa(0,0,0,0) = 0.625;  // <00||00>
    ints.eri_aaaa(1,1,1,1) = 0.625;  // <11||11>
    ints.eri_aaaa(0,1,0,1) = 0.479 - 0.297;  // <01||01>
    ints.eri_aaaa(1,0,1,0) = 0.479 - 0.297;  // <10||10>
    
    // Same for beta-beta
    ints.eri_bbbb(0,0,0,0) = 0.625;
    ints.eri_bbbb(1,1,1,1) = 0.625;
    ints.eri_bbbb(0,1,0,1) = 0.479 - 0.297;
    ints.eri_bbbb(1,0,1,0) = 0.479 - 0.297;
    
    // For opposite-spin (alpha-beta): no antisymmetrization
    ints.eri_aabb(0,0,0,0) = 0.625;  // <00|00>
    ints.eri_aabb(1,1,1,1) = 0.625;  // <11|11>
    ints.eri_aabb(0,1,0,1) = 0.479;  // <01|01>
    ints.eri_aabb(1,0,1,0) = 0.479;  // <10|10>
    
    // Nuclear repulsion
    ints.e_nuc = 0.7;  // Approximate for H2 at R=1.4 bohr
    
    // HF determinant: both electrons in orbital 0
    Determinant hf_det(std::vector<int>{0}, std::vector<int>{0});
    
    std::cout << "HF determinant: " << hf_det.to_string() << "\n";
    std::cout << "n_alpha = " << hf_det.n_alpha() << ", n_beta = " << hf_det.n_beta() << "\n\n";
    
    // Create CISD object
    // Occupied: 1 alpha, 1 beta (orbital 0)
    // Virtual: 1 alpha, 1 beta (orbital 1)
    CISD cisd(ints, hf_det, 1, 1, 1, 1);
    
    // Run CISD computation
    auto result = cisd.compute();
    
    std::cout << "\n=== Final Results ===\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Number of determinants: " << result.n_determinants << "\n";
    std::cout << "HF energy:    " << result.e_hf << " Ha\n";
    std::cout << "CISD energy:  " << result.e_cisd << " Ha\n";
    std::cout << "Correlation:  " << result.e_corr << " Ha\n";
    std::cout << "Convergence:  " << (result.converged ? "YES" : "NO") << "\n";
    
    std::cout << "\n=== Test Complete ===\n";
    
    return 0;
}
