/**
 * @file include/mshqc/mp3/mp3.h
 * @brief Unified MP3 Engine (RMP3, UMP3, OMP3) - Pure TBLIS
 */

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
