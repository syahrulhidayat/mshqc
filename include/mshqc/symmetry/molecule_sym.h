#ifndef MSHQC_SYMMETRY_MOLECULE_SYM_H
#define MSHQC_SYMMETRY_MOLECULE_SYM_H

#include "mshqc/basis.h"
#include "mshqc/symmetry/point_group.h"
#include "mshqc/symmetry/petite_list.h"
#include <map>
#include <Eigen/Dense>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

class BasisSymmetrizer {
public:
    BasisSymmetrizer(const BasisSet& basis, const PointGroup& pg, const PetiteList& pl);

    // Fungsi Utama: Proyeksi Simetri Penuh untuk Matriks Fock
    void symmetrize(Eigen::MatrixXd& F) const;
    
    // Memproyeksikan matriks koefisien C ke operasi simetri untuk menebak Irrep-nya
    std::vector<int> assign_mo_irreps(const Eigen::MatrixXd& C, double threshold = 1e-5) const;

    // Akses publik ke matriks transformasi
    const std::vector<Eigen::MatrixXd>& get_R_ao() const { return R_ao_; }

private:
    const BasisSet& basis_;
    const PointGroup& pg_;
    const PetiteList& pl_;

    std::vector<std::vector<int>> shell_map_;
    std::vector<Eigen::MatrixXd> R_ao_; // Matriks Transformasi AO (N x N)

    void build_map_and_matrices();
};

} // namespace mshqc

#endif