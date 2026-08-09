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

#include "mshqc/scf/sad.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef I
#undef I
#endif

namespace mshqc {

struct AtomConfig {

    double occ_1s = 0;

    double occ_2s = 0; double occ_2p = 0;

    double occ_3s = 0; double occ_3p = 0;

    double occ_4s = 0; double occ_3d = 0; double occ_4p = 0;

    double occ_5s = 0; double occ_4d = 0; double occ_5p = 0;

    double occ_6s = 0; double occ_4f = 0; double occ_5d = 0; double occ_6p = 0;

    double occ_7s = 0; double occ_5f = 0; double occ_6d = 0; double occ_7p = 0;

    double occ_8s = 0;
};

AtomConfig get_config(int Z) {
    AtomConfig c;

    if (Z >= 1) c.occ_1s = std::min(2.0, (double)Z);

    if (Z > 2) {
        double rem = Z - 2;
        c.occ_2s = std::min(2.0, rem); rem -= c.occ_2s;
        if (rem > 0) c.occ_2p = std::min(6.0, rem);
    }

    if (Z > 10) {
        double rem = Z - 10;
        c.occ_3s = std::min(2.0, rem); rem -= c.occ_3s;
        if (rem > 0) c.occ_3p = std::min(6.0, rem);
    }

    if (Z > 18) {
        double rem = Z - 18;
        c.occ_4s = std::min(2.0, rem); rem -= c.occ_4s;
        if (rem > 0) { c.occ_3d = std::min(10.0, rem); rem -= c.occ_3d; }
        if (rem > 0) { c.occ_4p = std::min(6.0, rem); rem -= c.occ_4p; }
    }

    if (Z > 36) {
        double rem = Z - 36;
        c.occ_5s = std::min(2.0, rem); rem -= c.occ_5s;
        if (rem > 0) { c.occ_4d = std::min(10.0, rem); rem -= c.occ_4d; }
        if (rem > 0) { c.occ_5p = std::min(6.0, rem); rem -= c.occ_5p; }
    }

    if (Z > 54) {
        double rem = Z - 54;
        c.occ_6s = std::min(2.0, rem); rem -= c.occ_6s;
        if (rem > 0) { c.occ_4f = std::min(14.0, rem); rem -= c.occ_4f; }
        if (rem > 0) { c.occ_5d = std::min(10.0, rem); rem -= c.occ_5d; }
        if (rem > 0) { c.occ_6p = std::min(6.0, rem); rem -= c.occ_6p; }
    }

    if (Z > 86) {
        double rem = Z - 86;
        c.occ_7s = std::min(2.0, rem); rem -= c.occ_7s;
        if (rem > 0) { c.occ_5f = std::min(14.0, rem); rem -= c.occ_5f; }
        if (rem > 0) { c.occ_6d = std::min(10.0, rem); rem -= c.occ_6d; }
        if (rem > 0) { c.occ_7p = std::min(6.0, rem); rem -= c.occ_7p; }
    }

    if (Z > 118) {
        double rem = Z - 118;
        c.occ_8s = std::min(2.0, rem); rem -= c.occ_8s;
    }

    return c;
}

Eigen::MatrixXd SADGuess::get_atomic_density(int Z, SadBasisType btype, int n_bf) {
    AtomConfig cfg = get_config(Z);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_bf, n_bf);
    int idx = 0;

    auto fill = [&](double occ, int deg, bool split) {
        double avg = occ / (double)deg;

        if (split) {

            if (idx + 2 * deg <= n_bf) {

                for(int k=0; k<deg; ++k) D(idx+k, idx+k) = avg * 0.8;

                for(int k=deg; k<2*deg; ++k) D(idx+k, idx+k) = avg * 0.2;
                idx += 2 * deg;
            } else if (idx + deg <= n_bf) {

                for(int k=0; k<deg; ++k) D(idx+k, idx+k) = avg;
                idx += deg;
            }
        } else {

            if (idx + deg <= n_bf) {
                for(int k=0; k<deg; ++k) D(idx+k, idx+k) = avg;
                idx += deg;
            }
        }
    };

    bool split_1s = (btype == SadBasisType::DOUBLE_ZETA && Z <= 2);
    fill(cfg.occ_1s, 1, split_1s);

    if (Z <= 2) return D;

    bool split_p2 = (btype == SadBasisType::DOUBLE_ZETA && Z <= 10);
    fill(cfg.occ_2s, 1, split_p2);
    fill(cfg.occ_2p, 3, split_p2);

    if (Z <= 10) return D;

    bool split_p3 = (btype == SadBasisType::DOUBLE_ZETA && Z <= 18);
    fill(cfg.occ_3s, 1, split_p3);
    fill(cfg.occ_3p, 3, split_p3);

    if (Z <= 18) return D;

    bool is_p4_valence = (Z <= 36);
    bool split_4s = (btype == SadBasisType::DOUBLE_ZETA && is_p4_valence);
    bool split_4p = (btype == SadBasisType::DOUBLE_ZETA && is_p4_valence && Z >= 31);
    bool split_3d = (btype == SadBasisType::DOUBLE_ZETA && is_p4_valence && Z >= 21);

    fill(cfg.occ_4s, 1, split_4s);
    if (idx + 5 <= n_bf) fill(cfg.occ_3d, 5, split_3d);
    fill(cfg.occ_4p, 3, split_4p);

    if (Z <= 36) return D;

    bool is_p5_valence = (Z <= 54);
    bool split_5s = (btype == SadBasisType::DOUBLE_ZETA && is_p5_valence);
    bool split_5p = (btype == SadBasisType::DOUBLE_ZETA && is_p5_valence && Z >= 49);
    bool split_4d = (btype == SadBasisType::DOUBLE_ZETA && Z >= 39);

    fill(cfg.occ_5s, 1, split_5s);
    if (idx + 5 <= n_bf) fill(cfg.occ_4d, 5, split_4d);
    fill(cfg.occ_5p, 3, split_5p);

    if (Z <= 54) return D;

    bool is_p6_valence = (Z <= 86);
    bool split_6s = (btype == SadBasisType::DOUBLE_ZETA && is_p6_valence);
    bool split_4f = (btype == SadBasisType::DOUBLE_ZETA && Z >= 58);
    bool split_5d = (btype == SadBasisType::DOUBLE_ZETA && Z >= 72);
    bool split_6p = (btype == SadBasisType::DOUBLE_ZETA && Z >= 81);

    fill(cfg.occ_6s, 1, split_6s);
    if (idx + 7 <= n_bf) fill(cfg.occ_4f, 7, split_4f);
    if (idx + 5 <= n_bf) fill(cfg.occ_5d, 5, split_5d);
    fill(cfg.occ_6p, 3, split_6p);

    if (Z <= 86) return D;

    bool is_p7_valence = (Z <= 118);
    bool split_7s = (btype == SadBasisType::DOUBLE_ZETA && is_p7_valence);
    bool split_5f = (btype == SadBasisType::DOUBLE_ZETA && Z >= 90);
    bool split_6d = (btype == SadBasisType::DOUBLE_ZETA && Z >= 104);
    bool split_7p = (btype == SadBasisType::DOUBLE_ZETA && Z >= 113);

    fill(cfg.occ_7s, 1, split_7s);
    if (idx + 7 <= n_bf) fill(cfg.occ_5f, 7, split_5f);
    if (idx + 5 <= n_bf) fill(cfg.occ_6d, 5, split_6d);
    fill(cfg.occ_7p, 3, split_7p);

    if (Z <= 118) return D;

    fill(cfg.occ_8s, 1, (btype == SadBasisType::DOUBLE_ZETA));

    return D;
}

}
