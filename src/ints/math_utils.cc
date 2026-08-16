// Copyright 2026 Muhamad Syahrul Hidayat
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