 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
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

#ifndef MSHQC_INTS_MATH_UTILS_H
#define MSHQC_INTS_MATH_UTILS_H

#include <Eigen/Dense>

#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace utils {

    using Tensor4D = Eigen::Tensor<double, 4>;
    using Tensor2D = Eigen::Tensor<double, 2>;

    Eigen::MatrixXd matrix_exponential(const Eigen::MatrixXd& mat);

    void set_zero(Tensor4D& tensor);

}

}

#endif
