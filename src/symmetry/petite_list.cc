 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
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

#include "mshqc/symmetry/petite_list.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#ifdef I
#undef I
#endif

namespace mshqc {

PetiteList::PetiteList(const BasisSet& basis, const PointGroup& pg)
    : basis_(basis), pg_(pg) {}

int PetiteList::find_shell_at(const Eigen::Vector3d& pos, int original_shell_idx) const {
    int target_l = basis_.shell(original_shell_idx).l();
    auto p_orig = basis_.shell(original_shell_idx).position();
    Eigen::Vector3d pos_orig(p_orig[0], p_orig[1], p_orig[2]);

    int n_prev = 0;
    for (int x = 0; x < original_shell_idx; ++x) {
        if (basis_.shell(x).l() == target_l) {
            auto px = basis_.shell(x).position();
            Eigen::Vector3d pos_x(px[0], px[1], px[2]);
            if ((pos_x - pos_orig).norm() < 1e-3) n_prev++;
        }
    }

    int n_found = 0;
    for (int i = 0; i < basis_.n_shells(); ++i) {
        if (basis_.shell(i).l() != target_l) continue;
        auto pi = basis_.shell(i).position();
        Eigen::Vector3d pos_i(pi[0], pi[1], pi[2]);
        if ((pos_i - pos).norm() < 1e-3) {
            if (n_found == n_prev) return i;
            n_found++;
        }
    }
    return -1;
}

void PetiteList::build() {
    unique_pairs_.clear();
    unique_quartets_.clear();
    int nshells = basis_.n_shells();
    const auto& ops = pg_.get_operations();
    double group_order = (double)ops.size();

    std::vector<std::vector<int>> shell_map(ops.size(), std::vector<int>(nshells));
    for (int s = 0; s < nshells; ++s) {
        auto p = basis_.shell(s).position();
        Eigen::Vector3d pos(p[0], p[1], p[2]);
        for (size_t k = 0; k < ops.size(); ++k) {
            Eigen::Vector3d new_pos = ops[k].matrix * pos;
            shell_map[k][s] = find_shell_at(new_pos, s);
            if (shell_map[k][s] == -1) shell_map[k][s] = s;
        }
    }

    for (int M = 0; M < nshells; ++M) {
        for (int N = 0; N <= M; ++N) {
            bool is_canonical = true;
            int equivalence_count = 0;

            for (size_t k = 0; k < ops.size(); ++k) {
                int M_new = shell_map[k][M];
                int N_new = shell_map[k][N];

                if (M_new < N_new) std::swap(M_new, N_new);

                if (M_new > M || (M_new == M && N_new > N)) {
                    is_canonical = false;
                    break;
                }
                if (M_new == M && N_new == N) equivalence_count++;
            }

            if (is_canonical) {
                double weight = group_order / (double)std::max(1, equivalence_count);
                unique_pairs_.push_back({M, N, weight});
            }
        }
    }

    for (int M = 0; M < nshells; ++M) {
        for (int N = 0; N <= M; ++N) {
            for (int P = 0; P <= M; ++P) {
                int Q_max = (M == P) ? N : P;
                for (int Q = 0; Q <= Q_max; ++Q) {

                    bool is_canonical = true;
                    int stabilizer_count = 0;

                    for (size_t k = 0; k < ops.size(); ++k) {
                        int m_new = shell_map[k][M];
                        int n_new = shell_map[k][N];
                        int p_new = shell_map[k][P];
                        int q_new = shell_map[k][Q];

                        if (m_new < n_new) std::swap(m_new, n_new);
                        if (p_new < q_new) std::swap(p_new, q_new);
                        if (m_new < p_new || (m_new == p_new && n_new < q_new)) {
                            std::swap(m_new, p_new);
                            std::swap(n_new, q_new);
                        }

                        if (m_new > M) { is_canonical = false; break; }
                        else if (m_new == M) {
                            if (n_new > N) { is_canonical = false; break; }
                            else if (n_new == N) {
                                if (p_new > P) { is_canonical = false; break; }
                                else if (p_new == P) {
                                    if (q_new > Q) { is_canonical = false; break; }
                                    else if (q_new == Q) {
                                        stabilizer_count++;
                                    }
                                }
                            }
                        }
                    }

                    if (is_canonical) {
                        double weight = group_order / (double)stabilizer_count;
                        unique_quartets_.push_back({M, N, P, Q, weight});
                    }
                }
            }
        }
    }
}

}
