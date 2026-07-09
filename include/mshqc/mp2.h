/**
 * @file include/mshqc/mp2.h
 * @brief Unified MP2 Engine - Consolidated RMP2, UMP2, and OMP2
 * @details Supports Incore/Direct execution alongside Exact/DF/Cholesky integral methods.
 */

#ifndef MSHQC_MP2_H
#define MSHQC_MP2_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/scf.h"
#include "mshqc/diis.h"
#include "mshqc/symmetry/blocked_tensor.h"
#include "mshqc/symmetry/point_group.h"
#include "mshqc/symmetry/petite_list.h"
#include "mshqc/symmetry/molecule_sym.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#include <string>

namespace mshqc {





struct MP2Config {
    std::string scf_type = "incore";     
    std::string eri_method = "exact";    
    
    bool use_df = false;
    std::string aux_basis_name = "";
    double df_threshold = 1e-9;
    double cholesky_threshold = 1e-9;
    std::string opt_method = "lbfgs"; 


    int max_iterations = 50;
    double energy_threshold = 1e-9;
    double gradient_threshold = -1.0;    
    int print_level = 1;
    bool exact_2rdm = true;              
    
    double memory_limit_gb = 4.0;        
};

struct MP2Result {
    double energy_scf = 0.0;
    double energy_mp2_corr = 0.0;
    double energy_mp2_ss = 0.0;
    double energy_mp2_os = 0.0;
    double energy_total = 0.0;
    
    

    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
    Eigen::VectorXd orbital_energies_alpha;
    Eigen::VectorXd orbital_energies_beta;
    Eigen::Tensor<double, 4> t2_aa;
    Eigen::Tensor<double, 4> t2_bb;
    Eigen::Tensor<double, 4> t2_ab;
    int n_occ_alpha = 0;
    int n_occ_beta = 0;
    int n_virt_alpha = 0;
    int n_virt_beta = 0;
    bool converged = false;
    int iterations = 0;
};

struct T2Amplitudes {
    Eigen::Tensor<double, 4> t2_aa;
    Eigen::Tensor<double, 4> t2_bb;
    Eigen::Tensor<double, 4> t2_ab;
};







class BaseMP2 {
protected:
    const Molecule& mol_;
    const BasisSet& basis_;
    std::shared_ptr<IntegralEngine> integrals_;
    SCFResult scf_;
    MP2Config config_;
    
    std::shared_ptr<PointGroup> pg_;
    std::shared_ptr<PetiteList> pl_;

    

    int nbf_;
    int nocc_a_, nocc_b_;
    int nvir_a_, nvir_b_;
    int n_frozen_ = 0; 

    

    Eigen::MatrixXd B_ia_P_alpha_;
    Eigen::MatrixXd B_ia_P_beta_;

public:
    BaseMP2(const Molecule& mol, const BasisSet& basis, 
            std::shared_ptr<IntegralEngine> integrals, 
            const SCFResult& scf_guess,
            const MP2Config& config,
            std::shared_ptr<PointGroup> pg = nullptr,
            std::shared_ptr<PetiteList> pl = nullptr);

    virtual ~BaseMP2() = default;

    

    virtual MP2Result compute() = 0;
    virtual void transform_integrals() = 0;

    

    void transform_3center_mo();
    void set_frozen_core(int n_frozen) { n_frozen_ = n_frozen; }
};







namespace foundation {
class RMP2 : public BaseMP2 {
public:
    using BaseMP2::BaseMP2; 

    
    MP2Result compute() override;
    void transform_integrals() override;
    
    const Eigen::Tensor<double, 4>& get_t2_amplitudes() const { return t2_; }
    
private:
    Eigen::Tensor<double, 4> eri_mo_; 

    Eigen::Tensor<double, 4> t2_;    
    double e_corr_ = 0.0; 
    
    std::vector<int> irreps_mo_;
    void compute_amplitudes_and_energy();
};
} 








class UMP2 : public BaseMP2 {
public:
    using BaseMP2::BaseMP2;
    
    MP2Result compute() override;
    void transform_integrals() override;
    
    T2Amplitudes get_t2_amplitudes() const;
    
private:
    Eigen::Tensor<double, 4> eri_aaaa_;  
    Eigen::Tensor<double, 4> eri_bbbb_;  
    Eigen::Tensor<double, 4> eri_aabb_;  
    
    Eigen::Tensor<double, 4> t2_aa_;  
    Eigen::Tensor<double, 4> t2_bb_;  
    Eigen::Tensor<double, 4> t2_ab_;  

    double compute_ss_alpha();
    double compute_ss_beta();
    double compute_os();
};







class OMP2 : public BaseMP2 {
public:
    OMP2(const Molecule& mol, const BasisSet& basis, 
         std::shared_ptr<IntegralEngine> integrals, 
         const SCFResult& scf_guess,
         const MP2Config& config,         
         std::shared_ptr<PointGroup> pg = nullptr,
         std::shared_ptr<PetiteList> pl = nullptr
        );
    
    MP2Result compute() override;
    void transform_integrals() override;
    
    

    void reset_diis();
    Eigen::MatrixXd build_opdm();
    Eigen::MatrixXd extrapolate_diis(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&);

private:
    std::unique_ptr<BasisSymmetrizer> symmetrizer_;
    int na_, nb_, va_, vb_;         
    bool exact_2rdm_;
    

    int max_iter_;
    double conv_thresh_;
    double grad_thresh_;
    
    Eigen::MatrixXd S_;
    Eigen::MatrixXd H_core_;

    

    std::vector<double> J_val_, K_val_;
    std::vector<int> J_ind_, K_ind_;
    std::vector<size_t> J_ptr_, K_ptr_;
    std::vector<std::pair<int, int>> row_map_;
    Eigen::MatrixXd schwarz_;
    
    

    BlockedTensor4D g_aa_, g_bb_, g_ab_;
    BlockedTensor4D t2_aa_, t2_bb_, t2_ab_;
    
    

    Eigen::MatrixXd G_oo_alpha_;
    Eigen::MatrixXd G_vv_alpha_;
    Eigen::MatrixXd G_oo_beta_; 
    Eigen::MatrixXd G_vv_beta_; 

    

    Eigen::MatrixXd C_a_current_; 
    Eigen::MatrixXd C_b_current_;
    Eigen::MatrixXd F_gen_a_;
    Eigen::MatrixXd F_gen_b_;

    

    Eigen::VectorXd orbital_gradient_;
    Eigen::VectorXd hessian_diag_;

    double e_ss_ = 0.0;
    double e_os_ = 0.0;

    

    double execute_micro_iterations();
    void execute_macro_iterations();
    void execute_macro_iterations(DIIS& diis_a, DIIS& diis_b, int macro_iter);
    void build_generalized_fock();
    Eigen::VectorXd compute_soscf_step();
    void apply_orbital_rotation(const Eigen::VectorXd& kappa);
    void init_fast_integrals(); 
    void pseudocanonicalize();
    void compute_t2_amplitudes();
    double compute_mp2_energy(); 
    void build_opdm_alpha();
    void build_opdm_beta();
    void build_fock_fast(const Eigen::MatrixXd& P_a, const Eigen::MatrixXd& P_b,
                         Eigen::MatrixXd& F_a, Eigen::MatrixXd& F_b);
};

} 


#endif 

