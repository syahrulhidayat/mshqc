#ifndef MSHQC_SAD_H
#define MSHQC_SAD_H

#include "mshqc/core/molecule.h"
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

    static Eigen::MatrixXd build(const Molecule& mol, const BasisSet& basis);

private:

    static Eigen::MatrixXd get_atomic_density(int Z, SadBasisType btype, int n_bf);
};

}

#endif
