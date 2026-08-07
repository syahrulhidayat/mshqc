/**
 * @file include/mshqc/fcidump.h
 */

#pragma once

#include "mshqc/core/molecule.h"
#include "mshqc/scf/scf.h"
#include "mshqc/ints/integrals.h"
#include <string>
#include <memory>

namespace mshqc {

/**
 * 
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

} 

