#ifndef MSHQC_SAD_H
#define MSHQC_SAD_H

#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#ifdef I
#undef I
#endif

namespace mshqc {



enum class SadBasisType { 
    MINIMAL, 
    DOUBLE_ZETA, 
    UNKNOWN 
};



class SADGuess {
public:
    /**
     * @brief Membangun tebakan densitas awal menggunakan Superposition of Atomic Densities.
     * @return Matriks Densitas (Nbf x Nbf). Mengembalikan Zero Matrix jika data tidak tersedia.
     */
    static Eigen::MatrixXd build(const Molecule& mol, const BasisSet& basis);

private:
    /**
     * @brief Mengambil densitas atomik pre-computed.
     * @param Z Atomic number.
     * @param btype Tipe basis set (enum).
     * @param n_basis_functions Jumlah fungsi basis pada atom ini (untuk validasi).
     */
    

    static Eigen::MatrixXd get_atomic_density(int Z, SadBasisType btype, int n_bf);
};

} 


#endif