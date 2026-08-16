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
 * @file include/mshqc/fcidump.h
 * @brief FCIDUMP Exporter for Qiskit / OpenFermion Compatibility
 */

#pragma once

#include "mshqc/core/molecule.h"
#include "mshqc/scf/scf.h"
#include "mshqc/ints/integrals.h"
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

} 

