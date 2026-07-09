#ifndef MSHQC_CORE_FOCK_BUILDER_H
#define MSHQC_CORE_FOCK_BUILDER_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "mshqc/ints/integrals.h" 


#include "mshqc/symmetry/petite_list.h"
#ifdef I
#undef I
#endif

namespace mshqc {

class FockBuilder {
public:
    


    FockBuilder(std::shared_ptr<IntegralEngine> integrals,
                const BasisSet& basis,
                const Eigen::MatrixXd& H_core,
                const Eigen::MatrixXd& schwarz);

    void reset();
    void set_petite_list(const PetiteList* list);

    void compute(const Eigen::MatrixXd& P_alpha, 
                 const Eigen::MatrixXd& P_beta,
                 Eigen::MatrixXd& F_alpha, 
                 Eigen::MatrixXd& F_beta);

private:
    std::shared_ptr<IntegralEngine> integrals_; 


    const BasisSet& basis_;
    int nbasis_;
    Eigen::MatrixXd H_core_;
    Eigen::MatrixXd schwarz_;
    
    


    Eigen::MatrixXd G_alpha_accum_;
    Eigen::MatrixXd G_beta_accum_;

    


    bool is_first_iter_;
    Eigen::MatrixXd P_alpha_old_;
    Eigen::MatrixXd P_beta_old_;

    const PetiteList* petite_list_ = nullptr;
};

} 


#endif