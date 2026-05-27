/**
 * @file eri_transformer.h
 * @brief Unified ERI transformation utilities for CI and MP modules
 * * This module provides efficient AO→MO integral transformation routines
 * used by both CI (Configuration Interaction) and MP (Møller-Plesset) modules.
 * Includes support for MP2, MP3, and MP4 (Triples).
 */

#ifndef MSHQC_INTEGRALS_ERI_TRANSFORMER_H
#define MSHQC_INTEGRALS_ERI_TRANSFORMER_H

#include "mshqc/symmetry/blocked_tensor.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

// Safety undef for complex number macros sometimes defined by system headers
namespace mshqc {
namespace integrals {

class ERITransformer {
public:
    // ========================================================================
    // OOVV TRANSFORMATIONS (Occupied-Occupied-Virtual-Virtual)
    // Returns (i, a, j, b) -> Chemist (ia|jb)
    // Used for: MP2 amplitudes, etc.
    // ========================================================================

    
    static Eigen::Tensor<double, 4> transform_oovv_quarter(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nocc, int nvirt
    );
    
    static Eigen::Tensor<double, 4> transform_oovv_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ_A, 
        const Eigen::MatrixXd& C_occ_B, 
        const Eigen::MatrixXd& C_virt_A, 
        const Eigen::MatrixXd& C_virt_B,
        int nbf, int nocc_A, int nocc_B, int nvirt_A, int nvirt_B
    );

    // ========================================================================
    // OO-VV TRANSFORMATIONS (Occupied-Occupied | Virtual-Virtual)
    // Returns (i, j, a, b) -> Chemist (ij|ab)
    // Used for: MP3 Ring Terms, CCD, CCSD
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_oo_vv(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nocc, int nvirt,
        bool use_disk = false,                                  
        const std::string& hdf5_filename = "eri_oo_vv.h5"
    );

    static Eigen::Tensor<double, 4> transform_oo_vv_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ_1, 
        const Eigen::MatrixXd& C_occ_2,
        const Eigen::MatrixXd& C_virt_1,
        const Eigen::MatrixXd& C_virt_2,
        int nbf, int nocc_1, int nocc_2, int nvirt_1, int nvirt_2
    );
    
    // ========================================================================
    // VVOV TRANSFORMATIONS (Virtual-Virtual-Occupied-Virtual)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_vvov(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nocc, int nvirt
    );
    
    // ========================================================================
    // OOOO TRANSFORMATIONS (Hole-Hole)
    // Returns (i, j, k, l)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_oooo(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        int nbf, int nocc
    );
    
    static Eigen::Tensor<double, 4> transform_oooo_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ_A,    
        const Eigen::MatrixXd& C_occ_B,    
        int nbf, int nocc_A, int nocc_B
    );
    
    // ========================================================================
    // VVVV TRANSFORMATIONS (Particle-Particle)
    // Returns (a, b, c, d)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_vvvv(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nvirt,
        bool use_disk = false,                                  
        const std::string& hdf5_filename = "eri_vvvv.h5"
    );
    
    static Eigen::Tensor<double, 4> transform_vvvv_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_virt_A,   
        const Eigen::MatrixXd& C_virt_B,   
        int nbf, int nvirt_A, int nvirt_B,
        bool use_disk = false,
        const std::string& hdf5_filename = "eri_vvvv_mixed.h5"
    );
    
    // ========================================================================
    // OVOV TRANSFORMATIONS (Particle-Hole / Ring)
    // Returns (i, a, j, b)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_ovov(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nocc, int nvirt,
        bool use_disk = false,                                  
        const std::string& hdf5_filename = "eri_ovov.h5"
    );
    
    static Eigen::Tensor<double, 4> transform_ovov_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ_A,    
        const Eigen::MatrixXd& C_virt_B,   
        int nbf, int nocc_A, int nvirt_B
    );

    // ========================================================================
    // VVVO TRANSFORMATIONS (Triples)
    // Returns (a, b, c, k)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_vvvo(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nocc, int nvirt,
        bool use_disk = false,
        const std::string& hdf5_filename = "eri_vvvo.h5"
    );
    
    static Eigen::Tensor<double, 4> transform_vvvo_mixed(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ_K,
        const Eigen::MatrixXd& C_virt_AC,
        const Eigen::MatrixXd& C_virt_B,
        int nbf, int nocc_K, int nvirt_AC, int nvirt_B
    );

    // ========================================================================
    // PARALLEL WRAPPERS (OpenMP optimized)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_oovv_parallel(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nocc, int nvirt,
        int n_threads = 0
    );
    
    static Eigen::Tensor<double, 4> transform_vvvv_parallel(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_virt,
        int nbf, int nvirt,
        int n_threads = 0
    );
    
    static Eigen::Tensor<double, 4> transform_oooo_parallel(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C_occ,
        int nbf, int nocc,
        int n_threads = 0
    );

    // ========================================================================
    // VARIATIONAL MP2 TRANSFORMATIONS (OOOV & OVVV)
    // [FIXED] Removed extra 'ERITransformer::' qualification
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_ooov(
        const Eigen::Tensor<double, 4>& eri_ao, 
        const Eigen::MatrixXd& Co, 
        const Eigen::MatrixXd& Cv, 
        int nbf, int no, int nv
    );
    
    static Eigen::Tensor<double, 4> transform_ovvv(
        const Eigen::Tensor<double, 4>& eri_ao, 
        const Eigen::MatrixXd& Co, 
        const Eigen::MatrixXd& Cv, 
        int nbf, int no, int nv,
        bool use_disk = false,
        const std::string& hdf5_filename = "eri_ovvv.h5"
    );
  
    static Eigen::Tensor<double, 4> transform_oovv(
        const Eigen::Tensor<double, 4>& eri_ao, 
        const Eigen::MatrixXd& Co, 
        const Eigen::MatrixXd& Cv, 
        int nbf, int no, int nv,
        bool use_disk = false,
        const std::string& hdf5_filename = "eri_oovv.h5"
    );
    static Eigen::Tensor<double, 4> transform_oovv_packed(
        const std::vector<double>& packed_eri, 
        const Eigen::MatrixXd& Co1, 
        const Eigen::MatrixXd& Cv1, 
        const Eigen::MatrixXd& Co2, 
        const Eigen::MatrixXd& Cv2, 
        int nbf, int no1, int nv1, int no2, int nv2
        
    );
    static BlockedTensor4D transform_oovv_blocked(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
        const std::vector<IrrepSpace>& occ_spaces,
        const std::vector<IrrepSpace>& virt_spaces,
        int nbf
    );
    static BlockedTensor4D transform_ovvv_blocked(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
        const std::vector<IrrepSpace>& occ_spaces,
        const std::vector<IrrepSpace>& virt_spaces,
        int nbf
    );

    static BlockedTensor4D transform_ooov_blocked(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& Co, const Eigen::MatrixXd& Cv,
        const std::vector<IrrepSpace>& occ_spaces,
        const std::vector<IrrepSpace>& virt_spaces,
        int nbf
    );
    
    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================
    static void antisymmetrize_vvvv(Eigen::Tensor<double, 4>& eri, int nvirt);
    static void antisymmetrize_oooo(Eigen::Tensor<double, 4>& eri, int nocc);
    static void antisymmetrize_oovv(Eigen::Tensor<double, 4>& eri, int nocc, int nvirt);
    static void antisymmetrize_ovov(Eigen::Tensor<double, 4>& eri, int nocc, int nvirt);
    // ========================================================================
    // CUSTOM / MIXED-SPIN TRANSFORMATIONS (For AB Cross-Terms in OMP3)
    // ========================================================================
    static Eigen::Tensor<double, 4> transform_custom(
        const Eigen::Tensor<double, 4>& eri_ao,
        const Eigen::MatrixXd& C1, 
        const Eigen::MatrixXd& C2, 
        const Eigen::MatrixXd& C3, 
        const Eigen::MatrixXd& C4,
        int nbf, int n1, int n2, int n3, int n4
    );
    
    static void print_transform_info(
        const char* name,
        int dim1, int dim2, int dim3, int dim4,
        double time_ms
    );
};

} // namespace integrals
} // namespace mshqc

#endif // MSHQC_INTEGRALS_ERI_TRANSFORMER_H