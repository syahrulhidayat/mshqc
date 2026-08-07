#ifndef MSHQC_GRADIENT_H
#define MSHQC_GRADIENT_H

#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/scf/scf.h"
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <string>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

struct GradientResult {

    Eigen::VectorXd gradient;

    double energy;

    double rms_gradient;

    double max_gradient;

    std::string method;

    bool is_analytical;

    Eigen::MatrixXd gradient_by_atom;

    void populate_gradient_by_atom(int natoms);
};

using EnergyFunction = std::function<double(const Molecule&)>;

class NumericalGradient {
public:

    explicit NumericalGradient(
        EnergyFunction energy_func,
        double delta = 1e-5,
        bool use_central = true
    );

    GradientResult compute(const Molecule& mol);

    double compute_component(const Molecule& mol, int atom_index, int coord_index);

    void set_delta(double delta) { delta_ = delta; }

    double get_delta() const { return delta_; }

    void set_use_central(bool use_central) { use_central_ = use_central; }

private:
    EnergyFunction energy_func_;
    double delta_;
    bool use_central_;

    Molecule displace_coordinate(
        const Molecule& mol,
        int atom_idx,
        int coord_idx,
        double displacement
    ) const;
};

class AnalyticalGradient {
public:
    virtual ~AnalyticalGradient() = default;

    virtual GradientResult compute(const Molecule& mol) = 0;

protected:

    static Eigen::VectorXd compute_nuclear_gradient(const Molecule& mol);
};

GradientResult compute_rhf_gradient_numerical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int charge,
    const SCFConfig& config = SCFConfig(),
    double delta = 1e-5
);

GradientResult compute_uhf_gradient_numerical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int charge,
    int multiplicity,
    const SCFConfig& config = SCFConfig(),
    double delta = 1e-5
);

void print_gradient(const GradientResult& result, const Molecule& mol);

bool is_gradient_converged(
    const Eigen::VectorXd& gradient,
    double rms_threshold = 3e-4,
    double max_threshold = 4.5e-4
);

}
}

#endif
