#ifndef MSHQC_UMP2_H
#define MSHQC_UMP2_H

#include "mshqc/scf.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory> 

#ifdef I
#undef I
#endif

/**
 * @file ump2.h
 * @brief Unrestricted MP2 for open-shell systems
 */

namespace mshqc {





class PointGroup;

/**
 * UMP2 result structure
 */
struct UMP2Result {
    double e_corr_ss_aa;  

    double e_corr_ss_bb;  

    double e_corr_os;     

    double e_corr_total;  

    double e_total;       

};

/**
 * T2 amplitude tensors for wavefunction analysis
 */
struct T2Amplitudes {
    Eigen::Tensor<double, 4> t2_aa;  

    Eigen::Tensor<double, 4> t2_bb;  

    Eigen::Tensor<double, 4> t2_ab;  

};

/**
 * Unrestricted Møller-Plesset 2nd order
 */
class UMP2 {
public:
    /**
     * Constructor
     * @param uhf_result UHF SCF result (must contain C_alpha, C_beta, eps_alpha, eps_beta)
     * @param basis Basis set
     * @param integrals Integral engine
     * @param pg Point Group symmetry object (Optional/Shared Pointer)
     */
    UMP2(const SCFResult& uhf_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         std::shared_ptr<PointGroup> pg = nullptr); 

    
    /**
     * Compute UMP2 energy
     */
    UMP2Result compute();
    
    /**
     * Get T2 amplitudes for wavefunction analysis
     * Must be called after compute()
     */
    T2Amplitudes get_t2_amplitudes() const;
    
private:
    const SCFResult& uhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    std::shared_ptr<PointGroup> pg_; 

    
    

    int nbf_;       

    int nocc_a_;    

    int nocc_b_;    

    int nvir_a_;    

    int nvir_b_;    

    
    

    Eigen::Tensor<double, 4> eri_aaaa_;  

    Eigen::Tensor<double, 4> eri_bbbb_;  

    Eigen::Tensor<double, 4> eri_aabb_;  

    
    

    Eigen::Tensor<double, 4> t2_aa_;  

    Eigen::Tensor<double, 4> t2_bb_;  

    Eigen::Tensor<double, 4> t2_ab_;  


    

    std::vector<int> irreps_occ_a_;
    std::vector<int> irreps_vir_a_;
    std::vector<int> irreps_occ_b_;
    std::vector<int> irreps_vir_b_;
    
    void transform_integrals();
    double compute_ss_alpha();
    double compute_ss_beta();
    double compute_os();
};

} 


#endif 
