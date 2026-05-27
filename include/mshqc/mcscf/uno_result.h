/**
 * @file include/mshqc/mcscf/uno_result.h
 * @brief Shared definition for UNO Results
 */

#ifndef MSHQC_MCSCF_UNO_RESULT_H
#define MSHQC_MCSCF_UNO_RESULT_H

#include <vector>
#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {

struct UNOResult {
    Eigen::MatrixXd C_uno;          // Koefisien UNO (AO basis)
    Eigen::VectorXd occupations;    // Bilangan okupansi (0.0 - 2.0)
    
    // Active Space Suggestions
    std::vector<int> active_indices;
    double entropy;                 // von Neumann entropy
    int suggested_n_active;
    int suggested_n_electrons;
};

} // namespace mcscf
} // namespace mshqc

#endif // MSHQC_MCSCF_UNO_RESULT_H