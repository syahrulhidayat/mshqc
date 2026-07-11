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

class OMP3 : public BaseMP3 {
public:
    using BaseMP3::BaseMP3;
    MP3Result compute() override;
    Eigen::VectorXd compute_fd_gradient(double delta = 1e-4);
private:
    

    Eigen::Tensor<double, 4> t2_3rd_aa_, t2_3rd_bb_, t2_3rd_ab_;
    Eigen::Tensor<double, 4> L2_aa_, L2_bb_, L2_ab_;
    Eigen::MatrixXd G_oo_alpha_, G_vv_alpha_, G_oo_beta_, G_vv_beta_;
    
    

    void pseudocanonicalize();
    void build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b, Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b);
    double compute_mp2_energy();
    double compute_mp3_correction();
    void solve_zvector(double current_grad_norm = 0.0);
    void build_opdm_alpha();
    void build_opdm_beta();
    
    

    Eigen::MatrixXd H_core_, S_, schwarz_;
    std::vector<std::pair<int, int>> row_map_;
    std::vector<double> J_val_, K_val_;
    std::vector<int> J_ind_, K_ind_;
    std::vector<size_t> J_ptr_, K_ptr_;
    void init_fast_integrals();
};

} 

#endif
