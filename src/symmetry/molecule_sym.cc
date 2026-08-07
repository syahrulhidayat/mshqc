#include "mshqc/symmetry/molecule_sym.h"
#include <iostream>
#include <cmath>

namespace mshqc {

BasisSymmetrizer::BasisSymmetrizer(const BasisSet& basis, const PointGroup& pg, const PetiteList& pl)
    : basis_(basis), pg_(pg), pl_(pl) {
    build_map_and_matrices();
}

double get_universal_phase(bool is_spherical, int l, int idx, double sx, double sy, double sz) {
    if (!is_spherical) {
        int a = 0, b = 0, c = 0, count = 0;
        for (int i = l; i >= 0; --i) {
            for (int j = l - i; j >= 0; --j) {
                if (count == idx) { a = i; b = j; c = l - i - j; break; }
                count++;
            }
        }
        double phase = 1.0;
        if (sx < 0 && (a % 2 != 0)) phase *= -1.0;
        if (sy < 0 && (b % 2 != 0)) phase *= -1.0;
        if (sz < 0 && (c % 2 != 0)) phase *= -1.0;
        return phase;
    } else {
        bool ox = false, oy = false, oz = false;

        if (l == 1) {
            if (idx == 0) ox = true;
            else if (idx == 1) oy = true;
            else if (idx == 2) oz = true;
        } else {

            int m = idx - l;
            if (m < 0) {
                oy = true;
                if (std::abs(m) % 2 == 0) ox = true;
            } else if (m > 0) {
                if (m % 2 != 0) ox = true;
            }
            if ((l - std::abs(m)) % 2 != 0) oz = true;
        }

        double phase = 1.0;
        if (sx < 0 && ox) phase *= -1.0;
        if (sy < 0 && oy) phase *= -1.0;
        if (sz < 0 && oz) phase *= -1.0;
        return phase;
    }
}

void BasisSymmetrizer::build_map_and_matrices() {
    int nshells = basis_.n_shells();
    int nbasis = basis_.n_basis_functions();
    const auto& ops = pg_.get_operations();
    int n_ops = ops.size();

    shell_map_.assign(n_ops, std::vector<int>(nshells, -1));
    R_ao_.assign(n_ops, Eigen::MatrixXd::Zero(nbasis, nbasis));

    auto map = basis_.shell_to_basis_function_map();

    for (int k = 0; k < n_ops; ++k) {
        const auto& R = ops[k].matrix;
        double sx = (R(0,0) > 0.1) ? 1.0 : -1.0;
        double sy = (R(1,1) > 0.1) ? 1.0 : -1.0;
        double sz = (R(2,2) > 0.1) ? 1.0 : -1.0;

        for (int i = 0; i < nshells; ++i) {
            Eigen::Vector3d p_i(basis_.shell(i).position()[0], basis_.shell(i).position()[1], basis_.shell(i).position()[2]);
            Eigen::Vector3d new_pos = R * p_i;
            int l_target = basis_.shell(i).l();

            int n_prev = 0;
            for (int x = 0; x < i; ++x) {
                if (basis_.shell(x).l() == l_target) {
                    Eigen::Vector3d px(basis_.shell(x).position()[0], basis_.shell(x).position()[1], basis_.shell(x).position()[2]);
                    if ((px - p_i).norm() < 1e-3) n_prev++;
                }
            }

            int n_found = 0;
            int target_idx = -1;
            for (int j = 0; j < nshells; ++j) {
                if (basis_.shell(j).l() != l_target) continue;
                Eigen::Vector3d pj(basis_.shell(j).position()[0], basis_.shell(j).position()[1], basis_.shell(j).position()[2]);
                if ((pj - new_pos).norm() < 1e-3) {
                    if (n_found == n_prev) { target_idx = j; break; }
                    n_found++;
                }
            }

            shell_map_[k][i] = target_idx;

            if (target_idx != -1) {
                int start_i = map[i];
                int start_j = map[target_idx];
                int dim = basis_.shell(i).is_spherical() ? (2*l_target + 1) : ((l_target+1)*(l_target+2)/2);

                for (int d = 0; d < dim; ++d) {
                    double phase = get_universal_phase(basis_.shell(i).is_spherical(), l_target, d, sx, sy, sz);
                    R_ao_[k](start_j + d, start_i + d) = phase;
                }
            }
        }
    }
}

void BasisSymmetrizer::symmetrize(Eigen::MatrixXd& F) const {
    if (R_ao_.empty()) return;

    Eigen::MatrixXd F_sym = Eigen::MatrixXd::Zero(F.rows(), F.cols());
    Eigen::MatrixXd temp_FR(F.rows(), F.cols());

    for (const auto& R : R_ao_) {
        temp_FR.noalias() = F * R;
        F_sym.noalias() += R.transpose() * temp_FR;
    }
    F = F_sym / static_cast<double>(R_ao_.size());
}
std::vector<int> BasisSymmetrizer::assign_mo_irreps(const Eigen::MatrixXd& C, double threshold) const {
    int n_mo = C.cols();
    std::vector<int> irreps(n_mo, 0);
    int n_ops = R_ao_.size();
    std::map<int, int> irrep_counts;

    for (int mo = 0; mo < n_mo; ++mo) {
        Eigen::VectorXd C_i = C.col(mo);
        int irrep_id = 0;

        for (int k = 0; k < n_ops; ++k) {
            Eigen::VectorXd C_trans = R_ao_[k] * C_i;
            double chi = C_i.dot(C_trans);

            double norm = C_i.norm() * C_trans.norm();
            if (norm > 1e-10 && (chi / norm) < -0.5) {
                irrep_id |= (1 << k);
            }
        }

        int mapped_id = 0;
        if (irrep_id & (1 << 1)) mapped_id ^= 1;
        if (irrep_id & (1 << 2)) mapped_id ^= 2;
        if (irrep_id & (1 << 4)) mapped_id ^= 4;

        irreps[mo] = mapped_id;
        irrep_counts[mapped_id]++;
    }

    return irreps;
}
}
