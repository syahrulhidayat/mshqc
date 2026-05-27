/**
 * @file include/mshqc/fcidump.h
 * @brief FCIDUMP Exporter for Qiskit / OpenFermion Compatibility
 */

#pragma once

#include "mshqc/molecule.h"
#include "mshqc/scf.h"
#include "mshqc/integrals.h"
#include <string>
#include <memory>

namespace mshqc {

/**
 * @brief Ekspor hasil SCF dan Integral ke format standar FCIDUMP.
 * 
 * @param filename Nama file output (misal: "H2O.FCIDUMP").
 * @param mol Objek Molekul (untuk mendapatkan tolakan inti & jumlah elektron).
 * @param scf Hasil SCF yang sudah konvergen (mengandung matriks koefisien C).
 * @param integrals Pointer ke IntegralEngine (untuk mengambil H_core dan ERI).
 * @param tol Batas toleransi untuk mengabaikan integral nol (default: 1e-10).
 */
void export_fcidump(const std::string& filename, 
                    const Molecule& mol, 
                    const SCFResult& scf, 
                    std::shared_ptr<IntegralEngine> integrals,
                    double tol = 1e-10);

} // namespace mshqc