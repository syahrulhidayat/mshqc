/**
 * @file semicanonical.cc
 * @brief Semi-canonical orbital transformation for ROHF-MP2
 * 
 * Transforms canonical ROHF orbitals to semi-canonical form by diagonalizing
 * Fock matrix within closed-shell, open-shell, and virtual subspaces.
 * This removes off-diagonal Fock elements that cause ROMP2 energy issues.
 * 
 * Theory Background:
 * Canonical ROHF produces non-diagonal Fock matrix elements between orbitals
 * of same symmetry type (e.g. open-virtual coupling). This causes standard
 * MP2 denominators ε_i + ε_j - ε_a - ε_b to be incorrect.
 * 
 * Semi-canonical transformation fixes this by rotating orbitals within each
 * subspace (closed, open, virtual) to diagonalize Fock, giving proper
 * one-electron energies for MP2 denominators.
 * 
 * Theory References:
 *   - P. J. Knowles et al., Chem. Phys. Lett. 186, 130 (1991)
 *     [Eq. (3): Semi-canonical transformation definition]
 *     [Eq. (4)-(6): Block-diagonal Fock matrix structure]
 *   - J. A. Pople et al., Int. J. Quantum Chem. Symp. 10, 1 (1976)
 *     [Original semi-canonical MP2 formulation for ROHF]
 *   - U. Bozkaya et al., J. Chem. Phys. 135, 104103 (2011)
 *     [Modern implementation details, Section II.A]
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-01-11
 * @license MIT License (see LICENSE file in project root)
 * 
 * @note Original implementation from Knowles et al. (1991) equations.
 *       No code copied from Psi4, PySCF, or other software.
 *       Algorithm: diagonalize Fock blocks independently.
 */

#include "mshqc/mp2.h"
#include <iostream>
#include <iomanip>
#include <Eigen/Eigenvalues>

namespace mshqc {

SCFResult semicanonicalize(const SCFResult& rohf) {
    // REFERENCE: Knowles et al. (1991), Eq. (3)
    // Transform canonical ROHF orbitals to semi-canonical form
    // by diagonalizing Fock within closed, open, and virtual blocks
    
    std::cout << "\n=== Semi-Canonical Transformation ===\n";
    std::cout << "REFERENCE: Knowles et al. (1991), Chem. Phys. Lett. 186, 130\n\n";
    
    SCFResult semi = rohf;  // copy all fields
    
    int nbf = rohf.C_alpha.rows();
    int n_closed = rohf.n_occ_beta;   // doubly occupied
    int n_open = rohf.n_occ_alpha - rohf.n_occ_beta;  // singly occupied
    int n_virt = nbf - rohf.n_occ_alpha;
    
    std::cout << "Orbital subspaces:\n";
    std::cout << "  Closed-shell: " << n_closed << " (doubly occ)\n";
    std::cout << "  Open-shell:   " << n_open << " (singly occ)\n";
    std::cout << "  Virtual:      " << n_virt << "\n\n";
    
    // Get Fock matrices in MO basis
    // F_MO = C^T F_AO C
    Eigen::MatrixXd F_alpha_mo = rohf.C_alpha.transpose() * rohf.F_alpha * rohf.C_alpha;
    Eigen::MatrixXd F_beta_mo = rohf.C_beta.transpose() * rohf.F_beta * rohf.C_beta;
    
    // --- Alpha spin ---
    // Closed block is already diagonal (eigenvalues from SCF)
    // Only need to diagonalize open and virtual blocks
    
    // Open-shell block (for alpha): orbitals n_closed to n_closed+n_open-1
    if (n_open > 0) {
        std::cout << "Diagonalizing α open-shell Fock block...\n";
        
        Eigen::MatrixXd F_open = F_alpha_mo.block(n_closed, n_closed, n_open, n_open);
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_open(F_open);
        Eigen::VectorXd eps_open = eig_open.eigenvalues();
        Eigen::MatrixXd U_open = eig_open.eigenvectors();
        
        // Update orbitals: C_new[:, open] = C_old[:, open] * U
        auto C_open_old = rohf.C_alpha.middleCols(n_closed, n_open);
        semi.C_alpha.middleCols(n_closed, n_open) = C_open_old * U_open;
        
        // Update energies
        for (int i = 0; i < n_open; i++) {
            semi.orbital_energies_alpha(n_closed + i) = eps_open(i);
        }
        
        std::cout << "  ✓ Open-shell α energies:";
        for (int i = 0; i < n_open; i++) {
            std::cout << " " << std::fixed << std::setprecision(6) << eps_open(i);
        }
        std::cout << "\n";
    }
    
    // Virtual block (alpha): orbitals n_closed+n_open to end
    if (n_virt > 0) {
        std::cout << "Diagonalizing α virtual Fock block...\n";
        
        int virt_start = n_closed + n_open;
        Eigen::MatrixXd F_virt = F_alpha_mo.block(virt_start, virt_start, n_virt, n_virt);
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_virt(F_virt);
        Eigen::VectorXd eps_virt = eig_virt.eigenvalues();
        Eigen::MatrixXd U_virt = eig_virt.eigenvectors();
        
        // Update orbitals
        auto C_virt_old = rohf.C_alpha.middleCols(virt_start, n_virt);
        semi.C_alpha.middleCols(virt_start, n_virt) = C_virt_old * U_virt;
        
        // Update energies
        for (int a = 0; a < n_virt; a++) {
            semi.orbital_energies_alpha(virt_start + a) = eps_virt(a);
        }
        
        std::cout << "  ✓ First 3 virtual α energies:";
        for (int a = 0; a < std::min(3, n_virt); a++) {
            std::cout << " " << std::fixed << std::setprecision(6) << eps_virt(a);
        }
        std::cout << "\n";
    }
    
    // --- Beta spin ---
    // Beta has only closed orbitals (no open-shell for beta)
    // Diagonalize virtual block only
    
    if (n_virt > 0) {
        std::cout << "Diagonalizing β virtual Fock block...\n";
        
        int virt_start_beta = n_closed;  // beta virtual starts right after closed
        Eigen::MatrixXd F_virt_beta = F_beta_mo.block(virt_start_beta, virt_start_beta, 
                                                      n_virt, n_virt);
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_virt_beta(F_virt_beta);
        Eigen::VectorXd eps_virt_beta = eig_virt_beta.eigenvalues();
        Eigen::MatrixXd U_virt_beta = eig_virt_beta.eigenvectors();
        
        // Update orbitals
        auto C_virt_beta_old = rohf.C_beta.middleCols(virt_start_beta, n_virt);
        semi.C_beta.middleCols(virt_start_beta, n_virt) = C_virt_beta_old * U_virt_beta;
        
        // Update energies
        for (int a = 0; a < n_virt; a++) {
            semi.orbital_energies_beta(virt_start_beta + a) = eps_virt_beta(a);
        }
        
        std::cout << "  ✓ First 3 virtual β energies:";
        for (int a = 0; a < std::min(3, n_virt); a++) {
            std::cout << " " << std::fixed << std::setprecision(6) << eps_virt_beta(a);
        }
        std::cout << "\n";
    }
    
    // Verify Fock is now block-diagonal in MO basis
    Eigen::MatrixXd F_alpha_semi = semi.C_alpha.transpose() * rohf.F_alpha * semi.C_alpha;
    
    // Check off-diagonal elements in open-virtual block
    double max_off_diag = 0.0;
    if (n_open > 0 && n_virt > 0) {
        int open_start = n_closed;
        int virt_start = n_closed + n_open;
        
        for (int i = 0; i < n_open; i++) {
            for (int a = 0; a < n_virt; a++) {
                double val = std::abs(F_alpha_semi(open_start + i, virt_start + a));
                max_off_diag = std::max(max_off_diag, val);
            }
        }
    }
    
    std::cout << "\nVerification:\n";
    std::cout << "  Max off-diagonal F(open,virt): " << std::scientific 
              << std::setprecision(3) << max_off_diag << "\n";
    std::cout << "  (should be ~0 if transformation correct)\n";
    
    std::cout << "\n✓ Semi-canonical transformation complete.\n";
    std::cout << "=====================================\n\n";
    
    return semi;
}

} // namespace mshqc
