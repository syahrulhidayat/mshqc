/**
 * @file include/mshqc/mcscf/cholesky_uno.h
 * @brief UHF Natural Orbitals (UNO) generator using Cholesky-UHF densities.
 * * THEORY:
 * UNOs are obtained by diagonalizing the total density matrix P = P_alpha + P_beta
 * from a Broken-Symmetry UHF calculation.
 * * The occupation numbers (n) indicate active space candidates:
 * - n ~ 2.00 : Doubly occupied (Inactive/Core)
 * - n ~ 1.00 : Singly occupied (Strongly Active)
 * - n ~ 0.00 : Virtual (Secondary)
 * * REFERENCE:
 * Pulay, P., & Hamilton, T. P. (1988). J. Chem. Phys. 88, 4926.
 * "UHF natural orbitals for defining active spaces for CASSCF calculations"
 * * @author Muhamad Sahrul Hidayat
 * @date 2025-12-15
 */
#ifndef MSHQC_MCSSC_CHOLESKY_UNO_H
#define MSHQC_MCSSC_CHOLESKY_UNO_H

#include "mshqc/scf/scf.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/core/molecule.h"
#include "mshqc/mcscf/uno_result.h" 


#include <vector>
#include <memory>
#include <string>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {




/**
 * @brief Cholesky-UNO Generator
 */
class CholeskyUNO {
public:
    /**
     * @brief Constructor
     * @param uhf_result Result from CholeskyUHF calculation
     * @param integrals Integral engine (needed for Overlap matrix)
     * @param n_basis Number of basis functions
     */
    CholeskyUNO(const SCFResult& uhf_result,
                std::shared_ptr<IntegralEngine> integrals,
                int n_basis);

    /**
     * @brief Compute Natural Orbitals
     * @return UNOResult containing orbitals and occupations
     */
    UNOResult compute();

    /**
     * @brief Print report including active space suggestions
     * @param threshold Occupation threshold for active space (e.g., 0.02 to 1.98)
     */
    void print_report(double threshold = 0.02) const;

    /**
     * @brief Export orbital file (Molden format usually, simplified here)
     */
    void save_orbitals(const std::string& filename) const;

private:
    SCFResult uhf_res_;
    std::shared_ptr<IntegralEngine> integrals_;
    int nbasis_;
    
    UNOResult result_;
    bool computed_ = false;

    /**
     * @brief Helper to calculate entropy S = -sum (n_i/2 ln(n_i/2))
     */
    double calculate_entropy(const Eigen::VectorXd& n) const;
    
    /**
     * @brief Analyze occupations to suggest CAS
     */
    void analyze_active_space(double threshold);
};

} 

} 


#endif 
