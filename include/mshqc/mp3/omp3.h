/**
 * @file include/mshqc/mp3/omp3.h
 * @brief Orbital-Optimized MP3 (OMP3) Header - Iterative 2-RDM
 * @author Muhamad Syahrul Hidayat
 * @license MIT License
 */

#ifndef MSHQC_OMP3_H
#define MSHQC_OMP3_H

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf/scf.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/mp2/mp2.h" 


#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#ifdef I
#undef I
#endif

namespace mshqc {

/**
 * @struct OMP3Result
 * @brief Results from OMP3 calculation
 */
struct OMP3Result {
    double energy_total;        

    double energy_mp2_corr;     

    double energy_mp3_corr;     

    double energy_omp2;         

    double energy_omp3;         

    bool converged;             

    int iterations;             

    
    Eigen::VectorXd orbital_energies_alpha;
    Eigen::VectorXd orbital_energies_beta;
    
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
};

/**
 * @class OMP3
 * @brief Orbital-Optimized Third-Order Møller-Plesset Perturbation Theory
 */
class OMP3 {
public:
    OMP3(const Molecule& mol, const BasisSet& basis,
        std::shared_ptr<IntegralEngine> integrals,
        const MP2Result& mp2_result,
        const SCFConfig& config,
        std::shared_ptr<PointGroup> pg = nullptr,      
        std::shared_ptr<PetiteList> pl = nullptr);
    
    OMP3Result compute();
    
    void set_max_iterations(int max_iter) { max_iter_ = max_iter; }
    void set_convergence_threshold(double thresh) { conv_thresh_ = thresh; }
    void set_gradient_threshold(double thresh) { grad_thresh_ = thresh; }

private:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFConfig config_;
    
    int nbf_;
    int na_; 
    int nb_; 
    int va_; 
    int vb_; 
    
    int max_iter_;
    double conv_thresh_;
    double grad_thresh_;
    
    SCFResult scf_; 
    
    

    Eigen::Tensor<double, 4> t2_aa_;
    Eigen::Tensor<double, 4> t2_bb_;
    Eigen::Tensor<double, 4> t2_ab_;

    

    Eigen::Tensor<double, 4> t2_3rd_aa_;
    Eigen::Tensor<double, 4> t2_3rd_bb_;
    Eigen::Tensor<double, 4> t2_3rd_ab_;

    

    

    

    Eigen::Tensor<double, 4> L2_aa_;
    Eigen::Tensor<double, 4> L2_bb_;
    Eigen::Tensor<double, 4> L2_ab_;

    

    Eigen::MatrixXd G_oo_alpha_, G_vv_alpha_;
    Eigen::MatrixXd G_oo_beta_, G_vv_beta_;
    
    Eigen::MatrixXd Gamma_oo_alpha_, Gamma_vv_alpha_;
    Eigen::MatrixXd Gamma_oo_beta_, Gamma_vv_beta_;

    Eigen::MatrixXd F_ia_alpha_;
    Eigen::MatrixXd F_ia_beta_;

    Eigen::MatrixXd H_core_;
    Eigen::MatrixXd S_;


    std::vector<double> J_val_, K_val_;
    std::vector<int> J_ind_, K_ind_;
    std::vector<size_t> J_ptr_, K_ptr_;
    std::vector<std::pair<int, int>> row_map_;
    Eigen::MatrixXd schwarz_;
    
    std::shared_ptr<PointGroup> pg_;
    std::shared_ptr<PetiteList> pl_;
    std::unique_ptr<BasisSymmetrizer> symmetrizer_;



    

    

    


    

    Eigen::Tensor<double, 4> compute_z_residual_aa(const Eigen::Tensor<double, 4>& L2, 
                                                   const Eigen::Tensor<double, 4>& T2, 
                                                   const Eigen::MatrixXd& packed_vvvv, 
                                                   const Eigen::MatrixXd& packed_oooo, 
                                                   const Eigen::MatrixXd& V_ring);

    Eigen::Tensor<double, 4> compute_z_residual_bb(const Eigen::Tensor<double, 4>& L2, 
                                                   const Eigen::Tensor<double, 4>& T2, 
                                                   const Eigen::MatrixXd& packed_vvvv, 
                                                   const Eigen::MatrixXd& packed_oooo, 
                                                   const Eigen::MatrixXd& V_ring);

    Eigen::Tensor<double, 4> compute_z_residual_ab(const Eigen::Tensor<double, 4>& L2, 
                                                   const Eigen::MatrixXd& V_vvvv_ab, 
                                                   const Eigen::MatrixXd& V_oooo_ab, 
                                                   const Eigen::MatrixXd& V_ring_aa, 
                                                   const Eigen::MatrixXd& V_ring_bb);

    

    

    

    
    

    void build_fock_ao(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b, 
                       Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b);

    void solve_zvector();
    void pseudocanonicalize();
    double compute_mp2_energy(); 
    double compute_mp2_singles_energy();
    double compute_mp3_correction(); 
    
    void build_opdm_alpha();
    void build_opdm_beta();

    void init_fast_integrals();
    void build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b,
                         Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b);
    
    

    void compute_t2_amplitudes(); 
    double compute_mp2_energy_from_t2(); 
    void build_mp3_density_contributions_alpha();
    void build_mp3_density_contributions_beta();
    Eigen::MatrixXd build_gfock_alpha(const Eigen::MatrixXd& G_mo_alpha, const Eigen::MatrixXd& Gamma_alpha);
    Eigen::MatrixXd build_gfock_beta(const Eigen::MatrixXd& G_mo_beta, const Eigen::MatrixXd& Gamma_beta);
    Eigen::MatrixXd compute_orbital_gradient_alpha(const Eigen::MatrixXd& F_mo);
    Eigen::MatrixXd compute_orbital_gradient_beta(const Eigen::MatrixXd& F_mo);
    void rotate_orbitals_alpha(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo);
    void rotate_orbitals_beta(const Eigen::MatrixXd& w_ai, const Eigen::MatrixXd& F_mo);
    bool converged(const Eigen::MatrixXd& w_alpha, const Eigen::MatrixXd& w_beta, double e_new, double e_old);
};

} 


#endif 
