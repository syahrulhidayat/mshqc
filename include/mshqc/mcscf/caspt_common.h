/**
 * @file caspt_common.h
 * @brief Shared Data Structures for TBLIS-based CASPT2/3
 * @author Muhamad Syahrul Hidayat
 * @date 2025-12-16
 */

#ifndef MSHQC_MCSCF_CASPT_COMMON_H
#define MSHQC_MCSCF_CASPT_COMMON_H

#include <vector>
#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {

/**
 * @brief Pre-computed MO integrals from Cholesky decomposition
 * Storage: Flat vector with (p,q,r,s) ordering
 * Access: V[p*N³ + q*N² + r*N + s]
 */
struct MOIntegrals {
    std::vector<double> V_pqrs;  

    int nbasis = 0;
    
    

    inline size_t idx(int p, int q, int r, int s) const {
        return static_cast<size_t>(p) * nbasis * nbasis * nbasis +
               static_cast<size_t>(q) * nbasis * nbasis +
               static_cast<size_t>(r) * nbasis +
               static_cast<size_t>(s);
    }
    
    double get(int p, int q, int r, int s) const {
        return V_pqrs[idx(p, q, r, s)];
    }
};

/**
 * @brief T2 amplitudes from PT2 (input for PT3)
 * Separated by space type for efficient contractions
 */
struct PT2Amplitudes {
    

    std::vector<double> t2_core;   

    
    

    std::vector<double> t2_active; 

    
    

    int n_occ = 0;
    int n_act = 0;
    int n_vir = 0;
    
    

    inline size_t idx_core(int i, int j, int a, int b) const {
        return static_cast<size_t>(i) * n_occ * n_vir * n_vir +
               static_cast<size_t>(j) * n_vir * n_vir +
               static_cast<size_t>(a) * n_vir +
               static_cast<size_t>(b);
    }
    
    inline size_t idx_active(int t, int u, int a, int b) const {
        return static_cast<size_t>(t) * n_act * n_vir * n_vir +
               static_cast<size_t>(u) * n_vir * n_vir +
               static_cast<size_t>(a) * n_vir +
               static_cast<size_t>(b);
    }
    
    void resize(int occ, int act, int vir) {
        n_occ = occ;
        n_act = act;
        n_vir = vir;
        t2_core.assign(occ * occ * vir * vir, 0.0);
        t2_active.assign(act * act * vir * vir, 0.0);
    }
};

/**
 * @brief Configuration for integral pre-computation
 */
struct IntegralConfig {
    bool use_symmetry = true;     

    double threshold = 1e-12;      

    int max_memory_mb = 4096;      

};

} 

} 


#endif