#pragma once

#include <Eigen/Dense>
#include <vector>
#include "mshqc/symmetry/molecule_sym.h"

namespace mshqc {

class SalcBuilder {
public:
    // Ubah parameter menjadi Raw Pointer
    SalcBuilder(BasisSymmetrizer* sym);
    
    std::pair<Eigen::MatrixXd, std::vector<int>> build_salc(const Eigen::MatrixXd& S);

private:
    // Ubah member menjadi Raw Pointer
    BasisSymmetrizer* sym_;
};

} // namespace mshqc