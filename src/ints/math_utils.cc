#include "mshqc/ints/math_utils.h"
#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace utils {

    Eigen::MatrixXd matrix_exponential(const Eigen::MatrixXd& mat) {
        return mat.exp();
    }

    void set_zero(Tensor4D& tensor) {
        tensor.setZero();
    }

}
}
