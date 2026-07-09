#ifndef MSHQC_MPN_HIERARCHY_H
#define MSHQC_MPN_HIERARCHY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#ifdef I
#undef I
#endif

/**
 * @file mpn_hierarchy.h
 * @brief Complete Møller-Plesset perturbation hierarchy (orders 0-5)
 * 
 * Provides unified structure for:
 * - Energies: E^(0) through E^(5)
 * - Wavefunctions: Ψ^(0) through Ψ^(4)
 * 
 * All formulas are EXACT from Rayleigh-Schrödinger perturbation theory.
 * No approximations except basis set truncation.
 * 
 * THEORY REFERENCES:
 * - Møller & Plesset (1934): Original MP theory
 * - Szabo & Ostlund (1996): Textbook derivation (Chapter 6)
 * - Helgaker et al. (2000): Complete theory (Chapter 14)
 * 
 * @author MSH-QC Project (Original Implementation)
 * @date 2025-01-16
 * @license MIT
 */

namespace mshqc {

/**
 * @brief Complete perturbation hierarchy result
 * 
 * Contains all orders of energy and wavefunction from
 * exact Møller-Plesset perturbation theory.
 */
struct MPnHierarchyResult {
    

    

    

    
    

    double e0_hf;
    
    

    double e1;  

    
    

    double e2_mp2;
    double e2_aa;     

    double e2_bb;     

    double e2_ab;     

    
    

    double e3_mp3;
    double e3_aa;     

    double e3_bb;     

    double e3_ab;     

    
    

    double e4_mp4;
    double e4_s;      

    double e4_d;      

    double e4_t;      

    double e4_q;      

    
    

    double e5_mp5;
    double e5_t;      

    double e5_q;      

    double e5_p;      

    
    

    

    

    
    

    double e_total_mp0;  

    double e_total_mp1;  

    double e_total_mp2;  

    double e_total_mp3;  

    double e_total_mp4;  

    double e_total_mp5;  

    
    

    

    

    
    

    

    
    

    

    Eigen::Tensor<double, 4> t2_aa_1;  

    Eigen::Tensor<double, 4> t2_bb_1;  

    Eigen::Tensor<double, 4> t2_ab_1;  

    
    

    

    
    

    

    Eigen::Tensor<double, 2> t1_a_2;   

    Eigen::Tensor<double, 2> t1_b_2;   

    Eigen::Tensor<double, 4> t2_aa_2;  

    Eigen::Tensor<double, 4> t2_bb_2;  

    Eigen::Tensor<double, 4> t2_ab_2;  

    
    

    

    Eigen::Tensor<double, 2> t1_a_3;   

    Eigen::Tensor<double, 2> t1_b_3;   

    Eigen::Tensor<double, 4> t2_aa_3;  

    Eigen::Tensor<double, 4> t2_bb_3;  

    Eigen::Tensor<double, 4> t2_ab_3;  

    Eigen::Tensor<double, 6> t3_aaa_2; 

    Eigen::Tensor<double, 6> t3_bbb_2; 

    Eigen::Tensor<double, 6> t3_aab_2; 

    Eigen::Tensor<double, 6> t3_abb_2; 

    
    

    

    

    
    int n_occ_alpha;   

    int n_occ_beta;    

    int n_virt_alpha;  

    int n_virt_beta;   

    int n_basis;       

    
    

    

    

    
    double norm_t2_1;   

    double norm_t1_2;   

    double norm_t2_2;   

    double norm_t3_2;   

    double norm_t1_3;   

    double norm_t2_3;   

    
    

    

    

    
    bool mp2_computed;  

    bool mp3_computed;  

    bool mp4_computed;  

    bool mp5_computed;  

    
    bool psi1_computed; 

    bool psi3_computed; 

    bool psi4_computed; 

    
    std::string basis_name;  

    std::string molecule;    

    
    /**
     * @brief Print complete energy hierarchy table
     * 
     * Shows convergence of perturbation series:
     * Order | Energy        | Contribution | Cumulative
     * ------|---------------|--------------|------------
     * E^(0) | ...           | ...          | ...
     * E^(1) | 0.0           | 0.0          | ...
     * E^(2) | ...           | ...          | ...
     * etc.
     */
    void print_energy_table() const;
    
    /**
     * @brief Print wavefunction components summary
     * 
     * Shows which amplitudes are available:
     * Ψ^(n) | Singles | Doubles | Triples | Available?
     * ------|---------|---------|---------|------------
     * etc.
     */
    void print_wavefunction_summary() const;
    
    /**
     * @brief Print complete hierarchy summary
     * 
     * Combined energy table + wavefunction summary + convergence analysis
     */
    void print() const;
};

/**
 * @brief Build complete MPn hierarchy from individual results
 * 
 * Combines UHF, UMP2, UMP3, UMP4, UMP5 results into unified structure.
 * 
 * @param uhf_result HF reference
 * @param ump2_result MP2 result (required)
 * @param ump3_result MP3 result (optional, nullptr if not computed)
 * @param ump4_result MP4 result (optional, nullptr if not computed)
 * @param ump5_result MP5 result (optional, nullptr if not computed)
 * @return Complete hierarchy
 */
MPnHierarchyResult build_mpn_hierarchy(
    const struct SCFResult& uhf_result,
    const struct UMP2Result& ump2_result,
    const struct UMP3Result* ump3_result = nullptr,
    const void* ump4_result = nullptr,  

    const void* ump5_result = nullptr   

);

} 


#endif 

