#include "mshqc/ints/math_utils.h"
#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace utils {

    // Implementasi Matriks Eksponensial
    Eigen::MatrixXd matrix_exponential(const Eigen::MatrixXd& mat) {
        return mat.exp();
    }

    // Implementasi Helper Tensor
    void set_zero(Tensor4D& tensor) {
        tensor.setZero();
    }

} // namespace utils
} // namespace mshqc