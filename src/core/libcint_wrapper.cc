/**
 * @file libcint_wrapper.cc
 * @brief Implementation of Libcint wrapper for 3-center integrals
 * 
 * @author Muhamad Sahrul Hidayat
 * @date 2025-11-16
 * @license MIT License
 */

#include "mshqc/libcint_wrapper.h"
#include <cint.h>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace mshqc {

// Forward declaration of C functions
extern "C" {
    // 3-center 2-electron integral (μν|P)
    int cint3c2e_sph(double *out, int *shls,
                     int *atm, int natm, int *bas, int nbas, double *env);
    
    // 2-center 2-electron integral (P|Q)  
    int cint2c2e_sph(double *out, int *shls,
                     int *atm, int natm, int *bas, int nbas, double *env);
    
    // Get spherical dimension
    int CINTcgto_spheric(int bas_id, int *bas);
    
    // Normalization
    double CINTgto_norm(int l, double alpha);
}

struct LibcintWrapper::LibcintData {
    int* atm = nullptr;
    int* bas = nullptr;
    double* env = nullptr;
    int natm = 0;
    int nbas = 0;
    
    ~LibcintData() {
        delete[] atm;
        delete[] bas;
        delete[] env;
    }
};

LibcintWrapper::LibcintWrapper(const Molecule& mol, const BasisSet& basis)
    : mol_(mol), basis_(basis), nbasis_(basis.n_basis_functions()),
      libcint_data_(std::make_unique<LibcintData>()) {
}

LibcintWrapper::~LibcintWrapper() {
    // unique_ptr handles cleanup
}

void LibcintWrapper::convert_basis_to_libcint(
    const BasisSet& basis,
    int& natm, int*& atm,
    int& nbas, int*& bas,
    double*& env
) {
    // Allocate arrays
    natm = mol_.n_atoms();
    nbas = basis.n_shells();
    
    atm = new int[natm * ATM_SLOTS];
    bas = new int[nbas * BAS_SLOTS];
    env = new double[100000];  // Large buffer for exponents/coefficients
    
    std::memset(atm, 0, sizeof(int) * natm * ATM_SLOTS);
    std::memset(bas, 0, sizeof(int) * nbas * BAS_SLOTS);
    std::memset(env, 0, sizeof(double) * 100000);
    
    int env_offset = PTR_ENV_START;
    
    // Fill atom data
    for (size_t i = 0; i < natm; i++) {
        const auto& atom = mol_.atom(i);
        atm[CHARGE_OF + ATM_SLOTS * i] = atom.atomic_number;
        atm[PTR_COORD + ATM_SLOTS * i] = env_offset;
        
        env[env_offset + 0] = atom.x;
        env[env_offset + 1] = atom.y;
        env[env_offset + 2] = atom.z;
        env_offset += 3;
    }
    
    // Fill basis data
    for (size_t i = 0; i < nbas; i++) {
        const auto& shell = basis.shell(i);
        
        // Find atom index for this shell
        auto shell_pos = shell.position();
        int atom_idx = -1;
        for (size_t j = 0; j < natm; j++) {
            const auto& atom = mol_.atom(j);
            double dx = shell_pos[0] - atom.x;
            double dy = shell_pos[1] - atom.y;
            double dz = shell_pos[2] - atom.z;
            if (dx*dx + dy*dy + dz*dz < 1e-10) {
                atom_idx = j;
                break;
            }
        }
        if (atom_idx < 0) {
            throw std::runtime_error("Could not find atom for shell");
        }
        
        bas[ATOM_OF  + BAS_SLOTS * i] = atom_idx;
        bas[ANG_OF   + BAS_SLOTS * i] = shell.l();
        bas[NPRIM_OF + BAS_SLOTS * i] = shell.n_primitives();
        bas[NCTR_OF  + BAS_SLOTS * i] = 1;  // Assume uncontracted for now
        bas[PTR_EXP  + BAS_SLOTS * i] = env_offset;
        
        // Store exponents
        for (size_t j = 0; j < shell.n_primitives(); j++) {
            env[env_offset + j] = shell.primitive(j).exponent;
        }
        env_offset += shell.n_primitives();
        
        bas[PTR_COEFF + BAS_SLOTS * i] = env_offset;
        
        // Store normalized coefficients
        int l = shell.l();
        for (size_t j = 0; j < shell.n_primitives(); j++) {
            double alpha = shell.primitive(j).exponent;
            double coeff = shell.primitive(j).coefficient;
            env[env_offset + j] = coeff * CINTgto_norm(l, alpha);
        }
        env_offset += shell.n_primitives();
    }
}

Eigen::Tensor<double, 3> LibcintWrapper::compute_3center_eri(
    const BasisSet& aux_basis
) {
    std::cout << "Computing 3-center integrals using Libcint (4c->3c trick)...\n";
    std::cout << "  Primary basis: " << nbasis_ << " functions\n";
    std::cout << "  Auxiliary basis: " << aux_basis.n_basis_functions() << " functions\n";
    std::cout << "  Method: FALLBACK - Libcint 3c2e not available\n";
    std::cout << "  Recommendation: Use Cholesky ERI or improved Libint2\n";
    
    // IMPORTANT: Libcint does NOT have basic cint3c2e_sph function!
    // Only specialized versions exist (ip1, ip2, etc for derivatives)
    // 
    // OPTIONS:
    // 1. Use Cholesky decomposition (already implemented in MSH-QC)
    // 2. Use Libint2 4-center with contraction (current fallback)
    // 3. Integrate PySCF for 3-center integrals
    // 4. Use other library (e.g. Psi4's Libint2 patch)
    
    throw std::runtime_error(
        "Libcint does not provide basic 3-center integrals (cint3c2e_sph).\n"
        "Available options:\n"
        "1. Use CholeskyERI (already in MSH-QC): mshqc/integrals/cholesky_eri.h\n"
        "2. Use Libint2 approximation (compute_3center_eri fallback)\n"
        "3. Use density-fitted MP2 with Cholesky vectors\n"
        "For production DF-CASPT2, recommend Cholesky decomposition."
    );
}

Eigen::MatrixXd LibcintWrapper::compute_2center_eri(
    const BasisSet& aux_basis
) {
    std::cout << "Computing 2-center metric using Libcint...\\n";
    
    size_t naux = aux_basis.n_basis_functions();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(naux, naux);
    
    // Convert auxiliary basis to Libcint format
    int natm, nbas_aux;
    int *atm, *bas_aux;
    double *env;
    
    convert_basis_to_libcint(aux_basis, natm, atm, nbas_aux, bas_aux, env);
    
    // Compute 2-center integrals
    int shls[2];
    for (int P = 0; P < nbas_aux; P++) {
        int dP = CINTcgto_spheric(P, bas_aux);
        for (int Q = 0; Q <= P; Q++) {
            int dQ = CINTcgto_spheric(Q, bas_aux);
            
            shls[0] = P;
            shls[1] = Q;
            
            double* buf = new double[dP * dQ];
            
            int non_zero = cint2c2e_sph(buf, shls, atm, natm, 
                                       bas_aux, nbas_aux, env);
            
            if (non_zero) {
                // Map to basis functions
                int P_start = 0;
                for (int PP = 0; PP < P; PP++) {
                    P_start += CINTcgto_spheric(PP, bas_aux);
                }
                int Q_start = 0;
                for (int QQ = 0; QQ < Q; QQ++) {
                    Q_start += CINTcgto_spheric(QQ, bas_aux);
                }
                
                int idx = 0;
                for (int fP = 0; fP < dP; fP++) {
                    for (int fQ = 0; fQ < dQ; fQ++) {
                        int bf_P = P_start + fP;
                        int bf_Q = Q_start + fQ;
                        
                        J(bf_P, bf_Q) = buf[idx++];
                        if (bf_P != bf_Q) {
                            J(bf_Q, bf_P) = buf[idx-1];
                        }
                    }
                }
            }
            
            delete[] buf;
        }
    }
    
    std::cout << "  2-center metric complete\\n";
    
    // Cleanup
    delete[] atm;
    delete[] bas_aux;
    delete[] env;
    
    return J;
}

} // namespace mshqc
