// Copyright 2026 Muhamad Syahrul Hidayat
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

/**
 * @file src/symmetry/molecule_sym.cc
 * @brief Universal AO Transformation Matrix Builder
 */

#include "mshqc/symmetry/molecule_sym.h"
#include <iostream>
#include <cmath>
#include <vector>

namespace mshqc {

BasisSymmetrizer::BasisSymmetrizer(const BasisSet& basis, const PointGroup& pg, const PetiteList& pl)
    : basis_(basis), pg_(pg), pl_(pl) {
    build_map_and_matrices();
}

// THE UNIVERSAL PHASE SOLVER FOR D2h/C2v (Menangani semua s, p, d, f, g, dst)
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
        
        // PENGECUALIAN MUTLAK: libcint selalu memaksakan l=1 menjadi px, py, pz
        if (l == 1) {
            if (idx == 0) ox = true;
            else if (idx == 1) oy = true;
            else if (idx == 2) oz = true;
        } else {
            // Rumus Universal Parity untuk l=2 (d), 3 (f), 4 (g), dst (m = -l ... +l)
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
    int n_ops = static_cast<int>(ops.size());

    // [INJECTED_FINAL_OPT_NO_DENSE_RAO]
    shell_map_.assign(n_ops, std::vector<int>(nshells, -1));
    R_ao_.clear();
    ao_perm_.assign(n_ops, std::vector<int>(nbasis, -1));
    ao_phase_.assign(n_ops, std::vector<double>(nbasis, 0.0));

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
                    ao_perm_[k][start_i + d] = start_j + d;
                    ao_phase_[k][start_i + d] = phase;
                }
            }
        }
    }
}

void BasisSymmetrizer::symmetrize(Eigen::MatrixXd& F) const {
    const int n_ops = static_cast<int>(ao_perm_.size());
    if (n_ops == 0) return;

    const int n = static_cast<int>(F.rows());
    Eigen::MatrixXd F_sym = Eigen::MatrixXd::Zero(n, n);

    for (int k = 0; k < n_ops; ++k) {
        const auto& perm = ao_perm_[k];
        const auto& ph   = ao_phase_[k];
        for (int q = 0; q < n; ++q) {
            const int rq = perm[q];
            if (rq < 0) continue;
            const double ph_q = ph[q];
            const double* F_col = F.col(rq).data();
            double* dst_col = F_sym.col(q).data();
            for (int p = 0; p < n; ++p) {
                const int rp = perm[p];
                if (rp >= 0) dst_col[p] += ph[p] * ph_q * F_col[rp];
            }
        }
    }
    F.noalias() = F_sym / static_cast<double>(n_ops);
}

std::vector<int> BasisSymmetrizer::assign_mo_irreps(const Eigen::MatrixXd& C, double threshold) const {
    (void)threshold;
    const int nbf = static_cast<int>(C.rows());
    const int n_mo = static_cast<int>(C.cols());
    const int n_ops = static_cast<int>(ao_perm_.size());
    std::vector<int> irreps(n_mo, 0);

    for (int mo = 0; mo < n_mo; ++mo) {
        const double* c_ptr = C.col(mo).data();
        double norm_c = C.col(mo).norm();
        int irrep_id = 0;

        for (int k = 0; k < n_ops; ++k) {
            const auto& perm = ao_perm_[k];
            const auto& ph   = ao_phase_[k];
            double chi = 0.0;
            double norm_t_sq = 0.0;
            for (int p = 0; p < nbf; ++p) {
                const int rp = perm[p];
                if (rp >= 0) {
                    const double val = ph[p] * c_ptr[p];
                    chi += c_ptr[rp] * val;
                    norm_t_sq += val * val;
                }
            }
            double denom = norm_c * std::sqrt(norm_t_sq);
            if (denom > 1e-10 && (chi / denom) < -0.5) {
                irrep_id |= (1 << k);
            }
        }

        // GF(2) ISOMORPHISM: Mampatkan Bitmask 8-bit menjadi ID 0-7
        int mapped_id = 0;
        if (irrep_id & (1 << 1)) mapped_id ^= 1;
        if (irrep_id & (1 << 2)) mapped_id ^= 2;
        if (irrep_id & (1 << 4)) mapped_id ^= 4;
        irreps[mo] = mapped_id;
    }
    return irreps;
}

} // namespace mshqc