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


#include "mshqc/scf/sad.h"
#include <iostream>
#include <algorithm>
#include <string>
#ifdef I
#undef I
#endif

namespace mshqc {

Eigen::MatrixXd SADGuess::build(const Molecule& mol, const BasisSet& basis) {
    int nbf = basis.n_basis_functions();
    Eigen::MatrixXd P_sad = Eigen::MatrixXd::Zero(nbf, nbf);

    std::string bname = basis.name();
    std::transform(bname.begin(), bname.end(), bname.begin(), ::tolower);

    SadBasisType btype = SadBasisType::UNKNOWN;

    if (bname.find("sto") != std::string::npos ||
        bname.find("min") != std::string::npos ||
        bname.find("3-21g") != std::string::npos) {
        btype = SadBasisType::MINIMAL;
    }
    else if (bname.find("dz") != std::string::npos ||
             bname.find("6-31g") != std::string::npos ||
             bname.find("def2-sv") != std::string::npos) {
        btype = SadBasisType::DOUBLE_ZETA;
    }

    int current_shell_idx = 0;
    int current_bf_offset = 0;
    bool missing_data = false;
    int atoms_found = 0;

    for (int i = 0; i < mol.n_atoms(); ++i) {
        int Z = mol.atom(i).atomic_number;

        int n_atom_bf = 0;
        int shells_for_this_atom = 0;

        for (int s = current_shell_idx; s < basis.n_shells(); ++s) {
            const auto& shell = basis.shell(s);

            if (shell.center_index() == i) {
                n_atom_bf += shell.n_functions();
                shells_for_this_atom++;
            } else {

                break;
            }
        }

        if (n_atom_bf == 0) {
            current_shell_idx += shells_for_this_atom;
            continue;
        }

        Eigen::MatrixXd D_atom = get_atomic_density(Z, btype, n_atom_bf);

        if (D_atom.rows() == 0 || D_atom.rows() != n_atom_bf) {
            missing_data = true;

        } else {

            P_sad.block(current_bf_offset, current_bf_offset, n_atom_bf, n_atom_bf) = D_atom;
            atoms_found++;
        }

        current_bf_offset += n_atom_bf;
        current_shell_idx += shells_for_this_atom;
    }

    if (P_sad.norm() < 1e-6) {
        return Eigen::MatrixXd::Zero(0,0);
    }

    return P_sad;
}

}
