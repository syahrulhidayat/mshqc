// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================


#include "mshqc/gradient/gradient.h"
#include "mshqc/scf/scf.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

void GradientResult::populate_gradient_by_atom(int natoms) {
    if (gradient.size() != 3 * natoms) {
        throw std::runtime_error("Gradient size mismatch: expected " +
                                std::to_string(3*natoms) + " but got " +
                                std::to_string(gradient.size()));
    }

    gradient_by_atom.resize(natoms, 3);
    for (int i = 0; i < natoms; ++i) {
        gradient_by_atom(i, 0) = gradient(3*i + 0);
        gradient_by_atom(i, 1) = gradient(3*i + 1);
        gradient_by_atom(i, 2) = gradient(3*i + 2);
    }
}

NumericalGradient::NumericalGradient(
    EnergyFunction energy_func,
    double delta,
    bool use_central
) : energy_func_(energy_func),
    delta_(delta),
    use_central_(use_central)
{
    if (delta_ <= 0.0) {
        throw std::invalid_argument("Delta must be positive");
    }
}

Molecule NumericalGradient::displace_coordinate(
    const Molecule& mol,
    int atom_idx,
    int coord_idx,
    double displacement
) const {
    const int natoms = mol.n_atoms();

    if (atom_idx < 0 || atom_idx >= natoms) {
        throw std::out_of_range("Atom index out of range");
    }
    if (coord_idx < 0 || coord_idx > 2) {
        throw std::out_of_range("Coordinate index must be 0, 1, or 2");
    }

    Molecule displaced_mol(mol.charge(), mol.multiplicity());

    for (int i = 0; i < natoms; ++i) {
        const auto& atom = mol.atom(i);
        double x = atom.x;
        double y = atom.y;
        double z = atom.z;

        if (i == atom_idx) {
            if (coord_idx == 0) {
                x += displacement;
            } else if (coord_idx == 1) {
                y += displacement;
            } else {
                z += displacement;
            }
        }

        displaced_mol.add_atom(atom.atomic_number, x, y, z);
    }

    return displaced_mol;
}

double NumericalGradient::compute_component(
    const Molecule& mol,
    int atom_index,
    int coord_index
) {
    if (use_central_) {

        Molecule mol_plus = displace_coordinate(mol, atom_index, coord_index, delta_);
        Molecule mol_minus = displace_coordinate(mol, atom_index, coord_index, -delta_);

        double E_plus = energy_func_(mol_plus);
        double E_minus = energy_func_(mol_minus);

        return (E_plus - E_minus) / (2.0 * delta_);
    } else {

        Molecule mol_plus = displace_coordinate(mol, atom_index, coord_index, delta_);

        double E_0 = energy_func_(mol);
        double E_plus = energy_func_(mol_plus);

        return (E_plus - E_0) / delta_;
    }
}

GradientResult NumericalGradient::compute(const Molecule& mol) {
    const int natoms = mol.n_atoms();
    const int ncoords = 3 * natoms;

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Numerical Gradient Calculation\n";
    std::cout << "========================================\n";
    std::cout << "Method:     " << (use_central_ ? "Central" : "Forward") << " difference\n";
    std::cout << "Step size:  " << delta_ << " au (" << delta_*0.529177 << " Å)\n";
    std::cout << "Atoms:      " << natoms << "\n";
    std::cout << "Evaluations:" << (use_central_ ? 2*ncoords : ncoords+1) << "\n";
    std::cout << "========================================\n\n";

    double E_0 = energy_func_(mol);
    std::cout << "Reference energy: " << std::fixed << std::setprecision(10)
              << E_0 << " Ha\n\n";

    Eigen::VectorXd grad(ncoords);

    std::cout << "Computing gradient components...\n";
    std::cout << "Atom  Coord      dE/dR (Ha/bohr)\n";
    std::cout << "------------------------------------\n";

    for (int atom = 0; atom < natoms; ++atom) {
        for (int coord = 0; coord < 3; ++coord) {
            int idx = 3*atom + coord;
            grad(idx) = compute_component(mol, atom, coord);

            char coord_label = (coord == 0) ? 'x' : (coord == 1) ? 'y' : 'z';
            std::cout << std::setw(4) << atom+1 << "  "
                      << coord_label << "    "
                      << std::setw(18) << std::scientific << std::setprecision(10)
                      << grad(idx) << "\n";
        }
    }

    double rms = grad.norm() / std::sqrt(static_cast<double>(ncoords));
    double max_component = grad.cwiseAbs().maxCoeff();

    std::cout << "\n";
    std::cout << "Gradient statistics:\n";
    std::cout << "  RMS gradient: " << std::scientific << std::setprecision(4)
              << rms << " Ha/bohr\n";
    std::cout << "  Max gradient: " << std::scientific << std::setprecision(4)
              << max_component << " Ha/bohr\n";
    std::cout << "  Norm:         " << std::scientific << std::setprecision(4)
              << grad.norm() << " Ha/bohr\n";
    std::cout << "\n";

    GradientResult result;
    result.gradient = grad;
    result.energy = E_0;
    result.rms_gradient = rms;
    result.max_gradient = max_component;
    result.method = "Numerical";
    result.is_analytical = false;
    result.populate_gradient_by_atom(natoms);

    return result;
}

Eigen::VectorXd AnalyticalGradient::compute_nuclear_gradient(const Molecule& mol) {
    const int natoms = mol.n_atoms();
    Eigen::VectorXd grad_nuc = Eigen::VectorXd::Zero(3 * natoms);

    for (int A = 0; A < natoms; ++A) {
        const auto& atomA = mol.atom(A);
        double xA = atomA.x;
        double yA = atomA.y;
        double zA = atomA.z;
        int ZA = atomA.atomic_number;

        for (int B = 0; B < natoms; ++B) {
            if (A == B) continue;

            const auto& atomB = mol.atom(B);
            double xB = atomB.x;
            double yB = atomB.y;
            double zB = atomB.z;
            int ZB = atomB.atomic_number;

            double dx = xA - xB;
            double dy = yA - yB;
            double dz = zA - zB;

            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            double R3 = R * R * R;

            double prefactor = static_cast<double>(ZA * ZB) / R3;

            grad_nuc(3*A + 0) += prefactor * dx;
            grad_nuc(3*A + 1) += prefactor * dy;
            grad_nuc(3*A + 2) += prefactor * dz;
        }
    }

    return grad_nuc;
}

GradientResult compute_rhf_gradient_numerical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int ,
    const SCFConfig& config,
    double delta
) {

    auto energy_func = [&](const Molecule& m) -> double {

        BasisSet basis_displaced(basis.name(), m);
        auto integrals_displaced = std::make_shared<IntegralEngine>(m, basis_displaced);

        RHF rhf(m, basis_displaced, integrals_displaced, nullptr, nullptr, config);
        auto result = rhf.compute();

        if (!result.converged) {
            std::cerr << "Warning: SCF did not converge at displaced geometry\n";
        }

        return result.energy_total;
    };

    NumericalGradient num_grad(energy_func, delta, true);
    auto result = num_grad.compute(mol);
    result.method = "RHF (numerical)";

    return result;
}

GradientResult compute_uhf_gradient_numerical(
    const Molecule& mol,
    const BasisSet& basis,
    std::shared_ptr<IntegralEngine> integrals,
    int charge,
    int multiplicity,
    const SCFConfig& config,
    double delta
) {

    int n_elec = mol.n_electrons() - charge;
    int n_alpha = (n_elec + multiplicity - 1) / 2;
    int n_beta = (n_elec - multiplicity + 1) / 2;

    auto energy_func = [&, n_alpha, n_beta](const Molecule& m) -> double {

        BasisSet basis_displaced(basis.name(), m);
        auto integrals_displaced = std::make_shared<IntegralEngine>(m, basis_displaced);

        UHF uhf(m, basis_displaced, integrals_displaced, nullptr, nullptr, n_alpha, n_beta, config);
        auto result = uhf.compute();

        if (!result.converged) {
            std::cerr << "Warning: SCF did not converge at displaced geometry\n";
        }

        return result.energy_total;
    };

    NumericalGradient num_grad(energy_func, delta, true);
    auto result = num_grad.compute(mol);
    result.method = "UHF (numerical)";

    return result;
}

void print_gradient(const GradientResult& result, const Molecule& mol) {
    const int natoms = mol.n_atoms();

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Gradient Results (" << result.method << ")\n";
    std::cout << "========================================\n";
    std::cout << "Energy: " << std::fixed << std::setprecision(10)
              << result.energy << " Ha\n";
    std::cout << "Type:   " << (result.is_analytical ? "Analytical" : "Numerical") << "\n";
    std::cout << "\n";
    std::cout << "Gradient by atom (Ha/bohr):\n";
    std::cout << "Atom     Element         X               Y               Z\n";
    std::cout << "----------------------------------------------------------------\n";

    for (int i = 0; i < natoms; ++i) {
        std::cout << std::setw(4) << i+1 << "     ";

        int Z = mol.atom(i).atomic_number;
        std::string elem = "?";
        if (Z == 1) elem = "H";
        else if (Z == 2) elem = "He";
        else if (Z == 3) elem = "Li";
        else if (Z == 4) elem = "Be";
        else if (Z == 6) elem = "C";
        else if (Z == 7) elem = "N";
        else if (Z == 8) elem = "O";
        else if (Z == 9) elem = "F";
        std::cout << std::setw(2) << std::left << elem << "       " << std::right;

        std::cout << std::setw(15) << std::scientific << std::setprecision(6)
                  << result.gradient_by_atom(i, 0) << " "
                  << std::setw(15) << result.gradient_by_atom(i, 1) << " "
                  << std::setw(15) << result.gradient_by_atom(i, 2) << "\n";
    }

    std::cout << "\n";
    std::cout << "Statistics:\n";
    std::cout << "  RMS gradient: " << std::scientific << std::setprecision(4)
              << result.rms_gradient << " Ha/bohr\n";
    std::cout << "  Max gradient: " << std::scientific << std::setprecision(4)
              << result.max_gradient << " Ha/bohr\n";
    std::cout << "\n";

    bool converged = is_gradient_converged(result.gradient);
    std::cout << "Convergence:  " << (converged ? "YES" : "NO") << "\n";
    std::cout << "========================================\n";
}

bool is_gradient_converged(
    const Eigen::VectorXd& gradient,
    double rms_threshold,
    double max_threshold
) {

    int ncoords = gradient.size();
    double rms = gradient.norm() / std::sqrt(static_cast<double>(ncoords));
    double max_component = gradient.cwiseAbs().maxCoeff();

    return (rms < rms_threshold) && (max_component < max_threshold);
}

}
}
