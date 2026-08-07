/**
 * @file src/foundation/fcidump.cc
 */

#include "mshqc/foundation/fcidump.h"
#include "mshqc/integrals/eri_transformer.h" // Gunakan transformer yang sudah ada
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace mshqc {

using integrals::ERITransformer;

void export_fcidump(const std::string& filename, 
                    const Molecule& mol, 
                    const SCFResult& scf, 
                    std::shared_ptr<IntegralEngine> integrals,
                    double tol) 
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        return;
    }

    int norb = scf.C_alpha.cols(); // Jumlah spatial orbital
    int nelec = scf.n_occ_alpha + scf.n_occ_beta;
    int ms2 = scf.n_occ_alpha - scf.n_occ_beta + 1; // Multiplisitas Spin: 2S + 1

    // =========================================================================
    // =========================================================================
        << ", NELEC= " << std::setw(3) << nelec 
        << ", MS2= " << ms2 << ",\n";
    
    // Tulis simetri (untuk saat ini asumsikan C1 point group / ISYM=1)
    out << " ORBSYM=";
    for (int i = 0; i < norb; ++i) {
        out << "1" << (i == norb - 1 ? "" : ",");
    }
    out << ",\n ISYM=1,\n&END\n";

    // Format output angka
    out << std::scientific << std::setprecision(12);

    // =========================================================================
    // 2. TRANSFORMASI & TULIS INTEGRAL 2-ELEKTRON (MO BASIS)
    // =========================================================================
    const auto& eri_ao = integrals->compute_eri();
    const Eigen::MatrixXd& C = scf.C_alpha; // Ekspor berdasarkan Alpha (RHF/ROHF standard)

    // Menggunakan TBLIS Anda untuk full transformasi (AO -> MO)
    Eigen::Tensor<double, 4> eri_mo = ERITransformer::transform_oovv_mixed(
        eri_ao, C, C, C, C, norb, norb, norb, norb, norb
    );

    // Iterasi indeks sesuai notasi kimiawi (ij|kl)
    for (int i = 0; i < norb; ++i) {
        for (int j = 0; j <= i; ++j) {
            int ij = i * (i + 1) / 2 + j;
            for (int k = 0; k < norb; ++k) {
                for (int l = 0; l <= k; ++l) {
                    int kl = k * (k + 1) / 2 + l;
                    if (ij >= kl) {
                        double val = eri_mo(i, k, j, l); // Perhatikan notasi fisik ke kimiawi
                        if (std::abs(val) > tol) {
                            out << std::setw(22) << val << " "
                                << std::setw(4) << (i + 1) << " "
                                << std::setw(4) << (j + 1) << " "
                                << std::setw(4) << (k + 1) << " "
                                << std::setw(4) << (l + 1) << "\n";
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // 3. TRANSFORMASI & TULIS INTEGRAL 1-ELEKTRON (MO BASIS)
    // =========================================================================
    Eigen::MatrixXd H_core_ao = integrals->compute_core_hamiltonian();
    Eigen::MatrixXd H_core_mo = C.transpose() * H_core_ao * C;

    for (int i = 0; i < norb; ++i) {
        for (int j = 0; j <= i; ++j) {
            double val = H_core_mo(i, j);
            if (std::abs(val) > tol) {
                out << std::setw(22) << val << " "
                    << std::setw(4) << (i + 1) << " "
                    << std::setw(4) << (j + 1) << " "
                    << "   0    0\n";
            }
        }
    }

    // =========================================================================
    // 4. TULIS ENERGI TOLAKAN INTI (CORE ENERGY)
    // =========================================================================
    out << std::setw(22) << mol.nuclear_repulsion_energy() << "    0    0    0    0\n";
    
    out.close();
}

} // namespace mshqc