 // ==============================================================================
 // Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //     http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 // ==============================================================================

#include "mshqc/mp2/mp2.h"
#include <iostream>
#include <iomanip>
#include <Eigen/Eigenvalues>
#ifdef I
#undef I
#endif

namespace mshqc {

SCFResult semicanonicalize(const SCFResult& rohf) {

    std::cout << "\n=== Semi-Canonical Transformation ===\n";
    std::cout << "REFERENCE: Knowles et al. (1991), Chem. Phys. Lett. 186, 130\n\n";

    SCFResult semi = rohf;

    int nbf = rohf.C_alpha.rows();
    int n_closed = rohf.n_occ_beta;
    int n_open = rohf.n_occ_alpha - rohf.n_occ_beta;
    int n_virt = nbf - rohf.n_occ_alpha;

    std::cout << "Orbital subspaces:\n";
    std::cout << "  Closed-shell: " << n_closed << " (doubly occ)\n";
    std::cout << "  Open-shell:   " << n_open << " (singly occ)\n";
    std::cout << "  Virtual:      " << n_virt << "\n\n";

    Eigen::MatrixXd F_alpha_mo = rohf.C_alpha.transpose() * rohf.F_alpha * rohf.C_alpha;
    Eigen::MatrixXd F_beta_mo = rohf.C_beta.transpose() * rohf.F_beta * rohf.C_beta;

    if (n_open > 0) {
        std::cout << "Diagonalizing α open-shell Fock block...\n";

        Eigen::MatrixXd F_open = F_alpha_mo.block(n_closed, n_closed, n_open, n_open);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_open(F_open);
        Eigen::VectorXd eps_open = eig_open.eigenvalues();
        Eigen::MatrixXd U_open = eig_open.eigenvectors();

        auto C_open_old = rohf.C_alpha.middleCols(n_closed, n_open);
        semi.C_alpha.middleCols(n_closed, n_open) = C_open_old * U_open;

        for (int i = 0; i < n_open; i++) {
            semi.orbital_energies_alpha(n_closed + i) = eps_open(i);
        }

        std::cout << "  ✓ Open-shell α energies:";
        for (int i = 0; i < n_open; i++) {
            std::cout << " " << std::fixed << std::setprecision(6) << eps_open(i);
        }
        std::cout << "\n";
    }

    if (n_virt > 0) {
        std::cout << "Diagonalizing α virtual Fock block...\n";

        int virt_start = n_closed + n_open;
        Eigen::MatrixXd F_virt = F_alpha_mo.block(virt_start, virt_start, n_virt, n_virt);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_virt(F_virt);
        Eigen::VectorXd eps_virt = eig_virt.eigenvalues();
        Eigen::MatrixXd U_virt = eig_virt.eigenvectors();

        auto C_virt_old = rohf.C_alpha.middleCols(virt_start, n_virt);
        semi.C_alpha.middleCols(virt_start, n_virt) = C_virt_old * U_virt;

        for (int a = 0; a < n_virt; a++) {
            semi.orbital_energies_alpha(virt_start + a) = eps_virt(a);
        }

        std::cout << "  ✓ First 3 virtual α energies:";
        for (int a = 0; a < std::min(3, n_virt); a++) {
            std::cout << " " << std::fixed << std::setprecision(6) << eps_virt(a);
        }
        std::cout << "\n";
    }

    if (n_virt > 0) {
        std::cout << "Diagonalizing β virtual Fock block...\n";

        int virt_start_beta = n_closed;
        Eigen::MatrixXd F_virt_beta = F_beta_mo.block(virt_start_beta, virt_start_beta,
                                                      n_virt, n_virt);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_virt_beta(F_virt_beta);
        Eigen::VectorXd eps_virt_beta = eig_virt_beta.eigenvalues();
        Eigen::MatrixXd U_virt_beta = eig_virt_beta.eigenvectors();

        auto C_virt_beta_old = rohf.C_beta.middleCols(virt_start_beta, n_virt);
        semi.C_beta.middleCols(virt_start_beta, n_virt) = C_virt_beta_old * U_virt_beta;

        for (int a = 0; a < n_virt; a++) {
            semi.orbital_energies_beta(virt_start_beta + a) = eps_virt_beta(a);
        }

        std::cout << "  ✓ First 3 virtual β energies:";
        for (int a = 0; a < std::min(3, n_virt); a++) {
            std::cout << " " << std::fixed << std::setprecision(6) << eps_virt_beta(a);
        }
        std::cout << "\n";
    }

    Eigen::MatrixXd F_alpha_semi = semi.C_alpha.transpose() * rohf.F_alpha * semi.C_alpha;

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

}
