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

#ifndef MSHQC_MP3_H
#define MSHQC_MP3_H

#include "mshqc/scf/scf.h"
#include "mshqc/mp2/mp2.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace mshqc {

struct MP3Result {
    double e_hf = 0.0;
    double e_mp2 = 0.0;
    double e_mp3 = 0.0;
    double e3_aa = 0.0, e3_bb = 0.0, e3_ab = 0.0;
    double e_corr_total = 0.0;
    double e_total = 0.0;
    bool converged = true;
    int iterations = 1;
};

class BaseMP3 {
protected:
    SCFResult scf_;
    MP2Result mp2_;
    MP2Config config_;
    std::shared_ptr<IntegralEngine> ints_;

    int nbf_, no_a_, no_b_, nv_a_, nv_b_;
    int n_aux_;

    Eigen::Tensor<double, 4> t2_aa_, t2_bb_, t2_ab_;

public:
    BaseMP3(const SCFResult& scf, const MP2Result& mp2, const MP2Config& config, std::shared_ptr<IntegralEngine> ints);
    virtual ~BaseMP3() = default;

    virtual MP3Result compute() = 0;

    double tensor_dot(const Eigen::Tensor<double, 4>& A, const Eigen::Tensor<double, 4>& B) const;
};

class RMP3 : public BaseMP3 {
public:
    using BaseMP3::BaseMP3;
    MP3Result compute() override;
};

class UMP3 : public BaseMP3 {
public:
    using BaseMP3::BaseMP3;
    MP3Result compute() override;
};
}

#endif
