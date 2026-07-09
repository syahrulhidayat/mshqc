/**
 * @file include/mshqc/foundation/rmp2.h
 * @brief Restricted MP2 Header - Turbo Optimized
 * @details Matching the new SIMD + Symmetry implementation in rmp2.cc
 */

#ifndef MSHQC_FOUNDATION_RMP2_H
#define MSHQC_FOUNDATION_RMP2_H

#include "mshqc/scf/scf.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/symmetry/point_group.h" 

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace foundation {

/**
 * @brief RMP2 calculation result structure
 */
struct RMP2Result {
    double e_rhf;         


    double e_corr;        


    double e_total;       


    int n_occ;            


    int n_virt;           


    
    


    Eigen::Tensor<double, 4> t2;  
};

/**
 * @brief Restricted Møller-Plesset 2nd order (High-Performance)
 */
class RMP2 {
public:
    /**
     * @brief Construct RMP2 solver
     * @param pg Optional PointGroup for symmetry acceleration (can be nullptr)
     */
    RMP2(const SCFResult& rhf_result,
         const BasisSet& basis,
         std::shared_ptr<IntegralEngine> integrals,
         std::shared_ptr<PointGroup> pg = nullptr);
    
    /**
     * @brief Execute the RMP2 calculation
     */
    RMP2Result compute();
    
    /**
     * @brief Retrieve T2 amplitudes after computation
     */
    const Eigen::Tensor<double, 4>& get_t2_amplitudes() const;
    
private:
    const SCFResult& rhf_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    std::shared_ptr<PointGroup> pg_; 


    
    


    int nbf_;      
    int nocc_;     
    int nvirt_;    
    
    


    Eigen::Tensor<double, 4> eri_mo_; 


    Eigen::Tensor<double, 4> t2_;     
    
    


    std::vector<int> irreps_occ_;
    std::vector<int> irreps_vir_;
    double e_corr_ = 0.0;

    


    void transform_integrals_ao_to_mo();
    
    


    void compute_amplitudes_and_energy();
};

} 


} 



#endif 

