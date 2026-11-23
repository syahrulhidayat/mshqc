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
 * @file src/scf/sad_guess.cc
 * @brief Implementasi logika pembangun SAD Guess berbasis pemetaan shell aktual.
 */

#include "mshqc/scf/sad.h"
#include <algorithm>
#include <string>
#include <vector>
#ifdef I
#undef I
#endif

namespace mshqc {

namespace {

struct SubshellOcc {
    std::vector<double> s_occ;
    std::vector<double> p_occ;
    std::vector<double> d_occ;
    std::vector<double> f_occ;
};

// [INJECTED_HPC_V4_SAD_HEAVY]
SubshellOcc get_subshell_occupations(int Z) {
  SubshellOcc cfg;
  auto push_if = [](std::vector<double>& vec, double val) {
    if (val > 0.0) vec.push_back(val);
  };

  // Tabel anomali konfigurasi ground-state eksperimental (logam transisi d & f-block)
  // mencegah tebakan awal SCF terjebak di konfigurasi excited s^2 d^(n-1)
  auto set_cfg = [&](std::initializer_list<double> s,
                     std::initializer_list<double> p,
                     std::initializer_list<double> d,
                     std::initializer_list<double> f) -> SubshellOcc {
    SubshellOcc out;
    out.s_occ.assign(s.begin(), s.end());
    out.p_occ.assign(p.begin(), p.end());
    out.d_occ.assign(d.begin(), d.end());
    out.f_occ.assign(f.begin(), f.end());
    return out;
  };

  switch (Z) {
    case 24: return set_cfg({2,2,2,1}, {6,6}, {5}, {});               // Cr: [Ar] 4s1 3d5
    case 29: return set_cfg({2,2,2,1}, {6,6}, {10}, {});              // Cu: [Ar] 4s1 3d10
    case 41: return set_cfg({2,2,2,2,1}, {6,6,6}, {10,4}, {});        // Nb: [Kr] 5s1 4d4
    case 42: return set_cfg({2,2,2,2,1}, {6,6,6}, {10,5}, {});        // Mo: [Kr] 5s1 4d5
    case 44: return set_cfg({2,2,2,2,1}, {6,6,6}, {10,7}, {});        // Ru: [Kr] 5s1 4d7
    case 45: return set_cfg({2,2,2,2,1}, {6,6,6}, {10,8}, {});        // Rh: [Kr] 5s1 4d8
    case 46: return set_cfg({2,2,2,2},   {6,6,6}, {10,10}, {});       // Pd: [Kr] 4d10
    case 47: return set_cfg({2,2,2,2,1}, {6,6,6}, {10,10}, {});       // Ag: [Kr] 5s1 4d10
    case 57: return set_cfg({2,2,2,2,2,2}, {6,6,6,6}, {10,10,1}, {}); // La: [Xe] 6s2 5d1
    case 58: return set_cfg({2,2,2,2,2,2}, {6,6,6,6}, {10,10,1}, {1});// Ce: [Xe] 6s2 4f1 5d1
    case 64: return set_cfg({2,2,2,2,2,2}, {6,6,6,6}, {10,10,1}, {7});// Gd: [Xe] 6s2 4f7 5d1
    case 78: return set_cfg({2,2,2,2,2,1}, {6,6,6,6}, {10,10,9}, {14});// Pt: [Xe] 6s1 4f14 5d9
    case 79: return set_cfg({2,2,2,2,2,1}, {6,6,6,6}, {10,10,10}, {14});// Au: [Xe] 6s1 4f14 5d10
    case 89: return set_cfg({2,2,2,2,2,2,2}, {6,6,6,6,6}, {10,10,10,1}, {14});    // Ac
    case 90: return set_cfg({2,2,2,2,2,2,2}, {6,6,6,6,6}, {10,10,10,2}, {14});    // Th
    case 91: return set_cfg({2,2,2,2,2,2,2}, {6,6,6,6,6}, {10,10,10,1}, {14,2});  // Pa
    case 92: return set_cfg({2,2,2,2,2,2,2}, {6,6,6,6,6}, {10,10,10,1}, {14,3});  // U
    case 93: return set_cfg({2,2,2,2,2,2,2}, {6,6,6,6,6}, {10,10,10,1}, {14,4});  // Np
    case 96: return set_cfg({2,2,2,2,2,2,2}, {6,6,6,6,6}, {10,10,10,1}, {14,7});  // Cm
    default: break;
  }

  if (Z >= 1) push_if(cfg.s_occ, std::min(2.0, static_cast<double>(Z)));
  if (Z > 2) {
    double rem = Z - 2;
    double o2s = std::min(2.0, rem); rem -= o2s; push_if(cfg.s_occ, o2s);
    double o2p = std::min(6.0, rem); rem -= o2p; push_if(cfg.p_occ, o2p);
  }
  if (Z > 10) {
    double rem = Z - 10;
    double o3s = std::min(2.0, rem); rem -= o3s; push_if(cfg.s_occ, o3s);
    double o3p = std::min(6.0, rem); rem -= o3p; push_if(cfg.p_occ, o3p);
  }
  if (Z > 18) {
    double rem = Z - 18;
    double o4s = std::min(2.0, rem); rem -= o4s; push_if(cfg.s_occ, o4s);
    double o3d = std::min(10.0, rem); rem -= o3d; push_if(cfg.d_occ, o3d);
    double o4p = std::min(6.0, rem); rem -= o4p; push_if(cfg.p_occ, o4p);
  }
  if (Z > 36) {
    double rem = Z - 36;
    double o5s = std::min(2.0, rem); rem -= o5s; push_if(cfg.s_occ, o5s);
    double o4d = std::min(10.0, rem); rem -= o4d; push_if(cfg.d_occ, o4d);
    double o5p = std::min(6.0, rem); rem -= o5p; push_if(cfg.p_occ, o5p);
  }
  if (Z > 54) {
    double rem = Z - 54;
    double o6s = std::min(2.0, rem); rem -= o6s; push_if(cfg.s_occ, o6s);
    double o4f = std::min(14.0, rem); rem -= o4f; push_if(cfg.f_occ, o4f);
    double o5d = std::min(10.0, rem); rem -= o5d; push_if(cfg.d_occ, o5d);
    double o6p = std::min(6.0, rem); rem -= o6p; push_if(cfg.p_occ, o6p);
  }
  if (Z > 86) {
    double rem = Z - 86;
    double o7s = std::min(2.0, rem); rem -= o7s; push_if(cfg.s_occ, o7s);
    double o5f = std::min(14.0, rem); rem -= o5f; push_if(cfg.f_occ, o5f);
    double o6d = std::min(10.0, rem); rem -= o6d; push_if(cfg.d_occ, o6d);
    double o7p = std::min(6.0, rem); rem -= o7p; push_if(cfg.p_occ, o7p);
  }
  if (Z > 118) {
    double rem = Z - 118;
    double o8s = std::min(2.0, rem); rem -= o8s; push_if(cfg.s_occ, o8s);
  }
  return cfg;
}

} // namespace

Eigen::MatrixXd SADGuess::build(const Molecule& mol, const BasisSet& basis) {
    int nbf = basis.n_basis_functions();
    Eigen::MatrixXd P_sad = Eigen::MatrixXd::Zero(nbf, nbf);

    std::string bname = basis.name();
    std::transform(bname.begin(), bname.end(), bname.begin(), ::tolower);

    bool is_minimal = (bname.find("sto") != std::string::npos ||
                       bname.find("min") != std::string::npos ||
                       bname.find("3-21g") != std::string::npos);
    bool allow_valence_split = !is_minimal;

    int current_shell_idx = 0;
    int current_bf_offset = 0;

    for (int i = 0; i < mol.n_atoms(); ++i) {
        int Z = mol.atom(i).atomic_number;

        int n_atom_bf = 0;
        int shells_for_this_atom = 0;

        // Simpan daftar (local_bf_offset, n_functions) per momentum sudut
        std::vector<std::pair<int, int>> s_shells;
        std::vector<std::pair<int, int>> p_shells;
        std::vector<std::pair<int, int>> d_shells;
        std::vector<std::pair<int, int>> f_shells;

        for (int s = current_shell_idx; s < basis.n_shells(); ++s) {
            const auto& shell = basis.shell(s);
            if (shell.center_index() == i) {
               
                const int l = shell.l();
                const int nf = shell.n_functions();
                if (l == 0) s_shells.push_back({n_atom_bf, nf});
                else if (l == 1) p_shells.push_back({n_atom_bf, nf});
                else if (l == 2) d_shells.push_back({n_atom_bf, nf});
                else if (l == 3) f_shells.push_back({n_atom_bf, nf});
                n_atom_bf += nf;
                shells_for_this_atom++;
            } else {
                break;
            }
        }

        if (n_atom_bf == 0) {
            current_shell_idx += shells_for_this_atom;
            continue;
        }

        Eigen::MatrixXd D_atom = Eigen::MatrixXd::Zero(n_atom_bf, n_atom_bf);
        SubshellOcc occ_cfg = get_subshell_occupations(Z);

        auto assign_angular_group = [&](const std::vector<double>& occ_list,
                                        const std::vector<std::pair<int, int>>& shells) {
            size_t n_sub = occ_list.size();
            for (size_t k = 0; k < n_sub; ++k) {
                if (k >= shells.size()) break;
                int off1 = shells[k].first;
                int deg  = shells[k].second;
                double avg = occ_list[k] / static_cast<double>(deg);

                bool is_valence = (k + 1 == n_sub);
                if (is_valence && allow_valence_split && (k + 1 < shells.size())) {
                    int off2 = shells[k + 1].first;
                    for (int m = 0; m < deg; ++m) {
                        D_atom(off1 + m, off1 + m) = avg * 0.8;
                        D_atom(off2 + m, off2 + m) = avg * 0.2;
                    }
                } else {
                    for (int m = 0; m < deg; ++m) {
                        D_atom(off1 + m, off1 + m) = avg;
                    }
                }
            }
        };

        assign_angular_group(occ_cfg.s_occ, s_shells);
        assign_angular_group(occ_cfg.p_occ, p_shells);
        assign_angular_group(occ_cfg.d_occ, d_shells);
        assign_angular_group(occ_cfg.f_occ, f_shells);

        if (D_atom.norm() > 1e-6) {
            P_sad.block(current_bf_offset, current_bf_offset, n_atom_bf, n_atom_bf) = D_atom;
        }

        current_bf_offset += n_atom_bf;
        current_shell_idx += shells_for_this_atom;
    }

    return P_sad;
}

} // namespace mshqc