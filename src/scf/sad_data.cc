/**
 * @file src/scf/sad_data.cc
 * @brief Database Densitas Atomik (H sampai Og/Z=118+)
 * @details Menangani splitting valence orbital secara otomatis hingga Periode 8.
 * Menggunakan Lambda helper untuk mengurangi duplikasi kode.
 */

#include "mshqc/sad.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef I
#undef I
#endif

namespace mshqc {

// 1. STRUKTUR KONFIGURASI ELEKTRON (Hingga Periode 8)
struct AtomConfig {
    // P1
    double occ_1s = 0;
    // P2
    double occ_2s = 0; double occ_2p = 0;
    // P3
    double occ_3s = 0; double occ_3p = 0;
    // P4
    double occ_4s = 0; double occ_3d = 0; double occ_4p = 0;
    // P5
    double occ_5s = 0; double occ_4d = 0; double occ_5p = 0;
    // P6
    double occ_6s = 0; double occ_4f = 0; double occ_5d = 0; double occ_6p = 0;
    // P7
    double occ_7s = 0; double occ_5f = 0; double occ_6d = 0; double occ_7p = 0;
    // P8
    double occ_8s = 0;
};

// 2. DATABASE PENGISIAN ELEKTRON (AUFBAU RULE)
AtomConfig get_config(int Z) {
    AtomConfig c;
    
    // Periode 1 (H-He)
    if (Z >= 1) c.occ_1s = std::min(2.0, (double)Z);
    
    // Periode 2 (Li-Ne)
    if (Z > 2) {
        double rem = Z - 2;
        c.occ_2s = std::min(2.0, rem); rem -= c.occ_2s;
        if (rem > 0) c.occ_2p = std::min(6.0, rem);
    }
    
    // Periode 3 (Na-Ar)
    if (Z > 10) {
        double rem = Z - 10;
        c.occ_3s = std::min(2.0, rem); rem -= c.occ_3s;
        if (rem > 0) c.occ_3p = std::min(6.0, rem);
    }

    // Periode 4 (K-Kr)
    if (Z > 18) {
        double rem = Z - 18;
        c.occ_4s = std::min(2.0, rem); rem -= c.occ_4s;
        if (rem > 0) { c.occ_3d = std::min(10.0, rem); rem -= c.occ_3d; }
        if (rem > 0) { c.occ_4p = std::min(6.0, rem); rem -= c.occ_4p; }
    }

    // Periode 5 (Rb-Xe)
    if (Z > 36) {
        double rem = Z - 36;
        c.occ_5s = std::min(2.0, rem); rem -= c.occ_5s;
        if (rem > 0) { c.occ_4d = std::min(10.0, rem); rem -= c.occ_4d; }
        if (rem > 0) { c.occ_5p = std::min(6.0, rem); rem -= c.occ_5p; }
    }

    // Periode 6 (Cs-Rn) - Termasuk Lanthanides (4f)
    if (Z > 54) {
        double rem = Z - 54;
        c.occ_6s = std::min(2.0, rem); rem -= c.occ_6s;
        if (rem > 0) { c.occ_4f = std::min(14.0, rem); rem -= c.occ_4f; }
        if (rem > 0) { c.occ_5d = std::min(10.0, rem); rem -= c.occ_5d; }
        if (rem > 0) { c.occ_6p = std::min(6.0, rem); rem -= c.occ_6p; }
    }

    // Periode 7 (Fr-Og) - Termasuk Actinides (5f)
    if (Z > 86) {
        double rem = Z - 86;
        c.occ_7s = std::min(2.0, rem); rem -= c.occ_7s;
        if (rem > 0) { c.occ_5f = std::min(14.0, rem); rem -= c.occ_5f; }
        if (rem > 0) { c.occ_6d = std::min(10.0, rem); rem -= c.occ_6d; }
        if (rem > 0) { c.occ_7p = std::min(6.0, rem); rem -= c.occ_7p; }
    }

    // Periode 8 (Hypothetical)
    if (Z > 118) {
        double rem = Z - 118;
        c.occ_8s = std::min(2.0, rem); rem -= c.occ_8s;
    }

    return c;
}

// 3. GENERATOR DENSITAS
Eigen::MatrixXd SADGuess::get_atomic_density(int Z, SadBasisType btype, int n_bf) {
    AtomConfig cfg = get_config(Z);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_bf, n_bf);
    int idx = 0; // Current Basis Function Index

    // --- HELPER LAMBDA: MENGISI ORBITAL ---
    auto fill = [&](double occ, int deg, bool split) {
        double avg = occ / (double)deg;
        
        if (split) {
            // Cek safety: apakah cukup ruang di basis set?
            if (idx + 2 * deg <= n_bf) {
                // Inner (Compact) - 80% density
                for(int k=0; k<deg; ++k) D(idx+k, idx+k) = avg * 0.8;
                // Outer (Diffuse) - 20% density
                for(int k=deg; k<2*deg; ++k) D(idx+k, idx+k) = avg * 0.2;
                idx += 2 * deg;
            } else if (idx + deg <= n_bf) {
                // Fallback jika tidak cukup ruang
                for(int k=0; k<deg; ++k) D(idx+k, idx+k) = avg;
                idx += deg;
            }
        } else {
            // Single Zeta
            if (idx + deg <= n_bf) {
                for(int k=0; k<deg; ++k) D(idx+k, idx+k) = avg;
                idx += deg;
            }
        }
    };

    // -------------------------------------------------------------------------
    // PERIODE 1: 1s
    // -------------------------------------------------------------------------
    bool split_1s = (btype == SadBasisType::DOUBLE_ZETA && Z <= 2);
    fill(cfg.occ_1s, 1, split_1s);

    if (Z <= 2) return D;

    // -------------------------------------------------------------------------
    // PERIODE 2: 2s, 2p
    // -------------------------------------------------------------------------
    bool split_p2 = (btype == SadBasisType::DOUBLE_ZETA && Z <= 10);
    fill(cfg.occ_2s, 1, split_p2);
    fill(cfg.occ_2p, 3, split_p2);

    if (Z <= 10) return D;

    // -------------------------------------------------------------------------
    // PERIODE 3: 3s, 3p
    // -------------------------------------------------------------------------
    bool split_p3 = (btype == SadBasisType::DOUBLE_ZETA && Z <= 18);
    fill(cfg.occ_3s, 1, split_p3);
    fill(cfg.occ_3p, 3, split_p3);

    if (Z <= 18) return D;

    // -------------------------------------------------------------------------
    // PERIODE 4: 4s, 3d, 4p
    // -------------------------------------------------------------------------
    bool is_p4_valence = (Z <= 36);
    bool split_4s = (btype == SadBasisType::DOUBLE_ZETA && is_p4_valence);
    bool split_4p = (btype == SadBasisType::DOUBLE_ZETA && is_p4_valence && Z >= 31);
    bool split_3d = (btype == SadBasisType::DOUBLE_ZETA && is_p4_valence && Z >= 21); 

    fill(cfg.occ_4s, 1, split_4s);
    if (idx + 5 <= n_bf) fill(cfg.occ_3d, 5, split_3d);
    fill(cfg.occ_4p, 3, split_4p);

    if (Z <= 36) return D;

    // -------------------------------------------------------------------------
    // PERIODE 5: 5s, 4d, 5p
    // -------------------------------------------------------------------------
    bool is_p5_valence = (Z <= 54);
    bool split_5s = (btype == SadBasisType::DOUBLE_ZETA && is_p5_valence);
    bool split_5p = (btype == SadBasisType::DOUBLE_ZETA && is_p5_valence && Z >= 49);
    bool split_4d = (btype == SadBasisType::DOUBLE_ZETA && Z >= 39);

    fill(cfg.occ_5s, 1, split_5s);
    if (idx + 5 <= n_bf) fill(cfg.occ_4d, 5, split_4d);
    fill(cfg.occ_5p, 3, split_5p);

    if (Z <= 54) return D;

    // -------------------------------------------------------------------------
    // PERIODE 6: 6s, 4f, 5d, 6p
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // PERIODE 7: 7s, 5f, 6d, 7p (Fr-Og)
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // PERIODE 8: 8s
    // -------------------------------------------------------------------------
    fill(cfg.occ_8s, 1, (btype == SadBasisType::DOUBLE_ZETA));

    return D;
}

} // namespace mshqc