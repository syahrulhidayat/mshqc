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

















#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/array.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>

#include <thread>
#include <omp.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#if defined(_WIN32)
    #include <windows.h>
    #include <vector>
#elif defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
#elif defined(__linux__)
    #include <fstream>
    #include <string>
    #include <unordered_set>
#endif


#include "mshqc/core/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/symmetry/point_group.h"
#include "mshqc/symmetry/petite_list.h"
#include "mshqc/symmetry/molecule_sym.h"
#include "mshqc/scf/scf.h"
#include "mshqc/scf/diis.h"
#include "mshqc/integrals/screening.h"
#include "mshqc/core/fock_builder.h"


#include "mshqc/mp2/mp2.h"
#include "mshqc/mp3/mp3.h"
#include "mshqc/mp3/omp3.h"






#include "mshqc/gradient/gradient.h"
#include "mshqc/gradient/optimizer.h"


#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/integrals/eri_transformer.h"

namespace nb = nanobind;
using namespace mshqc;
using namespace mshqc::integrals;




namespace mshqc_auto_tune {

    void set_env_if_empty(const char* name, const char* value) {
        if (std::getenv(name) == nullptr) {
        #if defined(_WIN32)
            _putenv_s(name, value);
        #else
            setenv(name, value, 0);
        #endif
        }
    }

    int get_physical_cores() {
    #if defined(_WIN32)
        DWORD length = 0;
        GetLogicalProcessorInformation(nullptr, &length);
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (!GetLogicalProcessorInformation(buffer.data(), &length)) return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
        int physical_cores = 0;
        for (const auto& info : buffer) { if (info.Relationship == RelationProcessorCore) physical_cores++; }
        return physical_cores > 0 ? physical_cores : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    #elif defined(__APPLE__)
        int count = 0;
        size_t size = sizeof(count);
        if (sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0) return count;
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    #elif defined(__linux__)
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line, current_phys_id = "0", current_core_id = "0";
        std::unordered_set<std::string> unique_cores;
        if (cpuinfo.is_open()) {
            while (std::getline(cpuinfo, line)) {
                if (line.find("physical id") == 0) current_phys_id = line.substr(line.find(":") + 1);
                else if (line.find("core id") == 0) {
                    current_core_id = line.substr(line.find(":") + 1);
                    unique_cores.insert(current_phys_id + "_" + current_core_id);
                }
            }
            if (!unique_cores.empty()) return unique_cores.size();
        }
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    #else
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    #endif
    }

    void initialize_hpc_environment() {
        int num_threads = get_physical_cores();
        if (const char* env_omp = std::getenv("OMP_NUM_THREADS")) {
            num_threads = std::atoi(env_omp);
        }

        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);

        set_env_if_empty("OPENBLAS_NUM_THREADS", "1");
        set_env_if_empty("MKL_NUM_THREADS", "1");
        set_env_if_empty("TBLIS_NUM_THREADS", "1");
        set_env_if_empty("VECLIB_MAXIMUM_THREADS", "1");
        set_env_if_empty("NUMEXPR_NUM_THREADS", "1");
        set_env_if_empty("OMP_MAX_ACTIVE_LEVELS", "1");
    }
}

NB_MODULE(_mshqc, m) {

    mshqc_auto_tune::initialize_hpc_environment();

    m.doc() = "MSHQC: Modern Quantum Chemistry Library";

    using ERITensor = Eigen::Tensor<double, 4, 0, long>;

    nb::class_<ERITensor>(m, "ERITensor")
        .def("size", &ERITensor::size)
        .def("dimension", &ERITensor::dimension)
        .def("shape", [](const ERITensor& t) {
            return std::make_tuple(t.dimension(0), t.dimension(1), t.dimension(2), t.dimension(3));
        })
        .def("__repr__", [](const ERITensor& t) {
            return "<mshqc.ERITensor shape=(" +
                   std::to_string(t.dimension(0)) + ", " +
                   std::to_string(t.dimension(1)) + ", " +
                   std::to_string(t.dimension(2)) + ", " +
                   std::to_string(t.dimension(3)) + ")>";
        });




    nb::class_<Atom>(m, "Atom")
        .def(nb::init<int, double, double, double>(),
             nb::arg("atomic_number"), nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def_rw("atomic_number", &Atom::atomic_number)
        .def_rw("x", &Atom::x)
        .def_rw("y", &Atom::y)
        .def_rw("z", &Atom::z)
        .def("position", &Atom::position);

    nb::class_<Molecule>(m, "Molecule")
        .def(nb::init<>())
        .def(nb::init<int, int>(), nb::arg("charge"), nb::arg("multiplicity"))
        .def("add_atom", nb::overload_cast<int, double, double, double>(&Molecule::add_atom),
             nb::arg("Z"), nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def("add_atom", nb::overload_cast<const Atom&>(&Molecule::add_atom), nb::arg("atom"))
        .def("n_atoms", &Molecule::n_atoms)
        .def("atom", &Molecule::atom, nb::arg("i"))
        .def("atoms", &Molecule::atoms)
        .def("total_nuclear_charge", &Molecule::total_nuclear_charge)
        .def("n_electrons", &Molecule::n_electrons)
        .def("charge", &Molecule::charge)
        .def("set_charge", &Molecule::set_charge, nb::arg("q"))
        .def("multiplicity", &Molecule::multiplicity)
        .def("set_multiplicity", &Molecule::set_multiplicity, nb::arg("m"))
        .def("nuclear_repulsion_energy", &Molecule::nuclear_repulsion_energy);

    nb::enum_<AngularMomentum>(m, "AngularMomentum")
        .value("S", AngularMomentum::S)
        .value("P", AngularMomentum::P)
        .value("D", AngularMomentum::D)
        .value("F", AngularMomentum::F)
        .value("G", AngularMomentum::G)
        .value("H", AngularMomentum::H);

    nb::class_<GaussianPrimitive>(m, "GaussianPrimitive")
        .def(nb::init<double, double>(), nb::arg("exponent"), nb::arg("coefficient"))
        .def_rw("exponent", &GaussianPrimitive::exponent)
        .def_rw("coefficient", &GaussianPrimitive::coefficient);

    nb::class_<Shell>(m, "Shell")
        .def(nb::init<AngularMomentum, int, const std::array<double, 3>&>(),
             nb::arg("am"), nb::arg("center"), nb::arg("center_pos"))
        .def_prop_ro("angular_momentum", &Shell::angular_momentum)
        .def_prop_ro("center", &Shell::center)
        .def_prop_ro("primitives", &Shell::primitives);

    nb::class_<BasisSet>(m, "BasisSet")
        .def(nb::init<>())
        .def(nb::init<const std::string&, const Molecule&, const std::string&>(),
             nb::arg("basis_name"), nb::arg("mol"), nb::arg("basis_dir") = "data/basis")
        .def("read_gbs", &BasisSet::read_gbs, nb::arg("basis_file"), nb::arg("mol"))
        .def("add_shell", &BasisSet::add_shell, nb::arg("shell"))
        .def("n_shells", &BasisSet::n_shells)
        .def("n_basis_functions", &BasisSet::n_basis_functions)
        .def("shell", &BasisSet::shell, nb::arg("i"))
        .def("name", &BasisSet::name);

    nb::class_<IntegralEngine>(m, "IntegralEngine")
        .def(nb::init<const Molecule&, const BasisSet&>(), nb::arg("mol"), nb::arg("basis"))
        .def("compute_overlap", &IntegralEngine::compute_overlap, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_kinetic", &IntegralEngine::compute_kinetic, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_nuclear", &IntegralEngine::compute_nuclear, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_eri", &IntegralEngine::compute_eri, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_eri_diagonal", &IntegralEngine::compute_eri_diagonal)
        .def("compute_eri_column", &IntegralEngine::compute_eri_column, nb::arg("pivot_index"));

    nb::class_<CholeskyDecompositionResult>(m, "CholeskyDecompositionResult")
        .def_ro("n_vectors", &CholeskyDecompositionResult::n_vectors)
        .def_ro("n_basis", &CholeskyDecompositionResult::n_basis)
        .def_ro("threshold", &CholeskyDecompositionResult::threshold)
        .def_ro("compression_ratio", &CholeskyDecompositionResult::compression_ratio)
        .def_ro("converged", &CholeskyDecompositionResult::converged);

    nb::class_<CholeskyERI>(m, "CholeskyERI")
        .def(nb::init<double>(), nb::arg("threshold") = 1e-9)
        .def(nb::init<const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("basis"), nb::arg("integrals"))
        .def("set_threshold", &integrals::CholeskyERI::set_threshold)
        .def("set_print_level", &integrals::CholeskyERI::set_print_level)
        .def("n_vectors", &integrals::CholeskyERI::n_vectors)
        .def("threshold", &integrals::CholeskyERI::threshold)
        .def("compute", &integrals::CholeskyERI::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("decompose", &integrals::CholeskyERI::decompose, nb::call_guard<nb::gil_scoped_release>())
        .def("get_L_mat", &integrals::CholeskyERI::get_L_mat, nb::rv_policy::reference_internal)
        .def("reconstruct", &integrals::CholeskyERI::reconstruct)
        .def("reconstruct_full", &integrals::CholeskyERI::reconstruct_full)
        .def("is_decomposed", &integrals::CholeskyERI::is_decomposed)
        .def("decomposed", &integrals::CholeskyERI::decomposed);




    nb::enum_<SymOpType>(m, "SymOpType")
        .value("Identity", SymOpType::Identity)
        .value("Rotation", SymOpType::Rotation)
        .value("Reflection", SymOpType::Reflection)
        .value("Inversion", SymOpType::Inversion)
        .value("ImproperRotation", SymOpType::ImproperRotation)
        .export_values();

    nb::class_<SymmetryOperation>(m, "SymmetryOperation")
        .def_rw("type", &SymmetryOperation::type)
        .def_rw("order", &SymmetryOperation::order)
        .def_rw("matrix", &SymmetryOperation::matrix)
        .def_rw("name", &SymmetryOperation::name);

    nb::class_<PointGroup>(m, "PointGroup")
        .def(nb::init<const Molecule&, double>(), "Initialize and auto-detect symmetry",
            nb::arg("mol"), nb::arg("tolerance") = 1e-6)
        .def("detect", &PointGroup::detect)
        .def("get_symbol", &PointGroup::get_symbol)
        .def("get_order", &PointGroup::get_order)
        .def("get_aligned_molecule", &PointGroup::get_aligned_molecule)
        .def("get_operations", &PointGroup::get_operations, nb::rv_policy::reference_internal);


    nb::class_<UniqueShellPair>(m, "UniqueShellPair")
        .def_rw("p", &UniqueShellPair::p)
        .def_rw("q", &UniqueShellPair::q)
        .def_rw("weight", &UniqueShellPair::weight);

    nb::class_<PetiteList>(m, "PetiteList")
        .def(nb::init<const BasisSet&, const PointGroup&>())
        .def("build", &PetiteList::build)
        .def("get_unique_pairs", &PetiteList::get_unique_pairs);

    nb::class_<BasisSymmetrizer>(m, "BasisSymmetrizer")
        .def(nb::init<const BasisSet&, const PointGroup&, const PetiteList&>())
        .def("symmetrize", &BasisSymmetrizer::symmetrize);




    nb::class_<DIIS>(m, "DIIS")
        .def(nb::init<int>(), nb::arg("max_vectors") = 8)
        .def("clear", &DIIS::clear)
        .def("add_iteration", &DIIS::add_iteration, nb::arg("F"), nb::arg("err"), nb::arg("P"))
        .def("extrapolate", &DIIS::extrapolate);

    nb::class_<mshqc::integrals::ShellPair>(m, "ShellPair")
        .def_rw("sh_a", &mshqc::integrals::ShellPair::sh_a)
        .def_rw("sh_b", &mshqc::integrals::ShellPair::sh_b)
        .def_rw("max_val", &mshqc::integrals::ShellPair::max_val)
        .def_prop_rw("center",
            [](const mshqc::integrals::ShellPair& sp) {
                return std::make_tuple(sp.center(0), sp.center(1), sp.center(2));
            },
            [](mshqc::integrals::ShellPair& sp, const std::tuple<double,double,double>& t) {
                sp.center = Eigen::Vector3d(std::get<0>(t), std::get<1>(t), std::get<2>(t));
            }
        );

    nb::class_<mshqc::integrals::Screening>(m, "Screening")
        .def(nb::init<const BasisSet&>(), nb::arg("basis"))
        .def("reset", &mshqc::integrals::Screening::reset)
        .def("compute", &mshqc::integrals::Screening::compute, nb::call_guard<nb::gil_scoped_release>(), nb::arg("integrals"))
        .def("print_stats", &mshqc::integrals::Screening::print_stats, nb::arg("threshold"))
        .def("get_significant_pairs", &mshqc::integrals::Screening::get_significant_pairs, nb::arg("threshold"))
        .def("is_significant",
             static_cast<bool (mshqc::integrals::Screening::*)(int, int, int, int, double) const>(&mshqc::integrals::Screening::is_significant),
             nb::arg("sh_a"), nb::arg("sh_b"), nb::arg("sh_c"), nb::arg("sh_d"), nb::arg("threshold"))
        .def("get_schwarz_val", &mshqc::integrals::Screening::get_schwarz_val)
        .def("max_schwarz", &mshqc::integrals::Screening::max_schwarz);




    nb::class_<SCFConfig>(m, "SCFConfig")
        .def(nb::init<>())
        .def_rw("scf_type", &SCFConfig::scf_type)
        .def_rw("eri_method", &SCFConfig::eri_method)
        .def_rw("cholesky_threshold", &SCFConfig::cholesky_threshold)
        .def_rw("df_threshold", &SCFConfig::df_threshold)
        .def_rw("max_iterations", &SCFConfig::max_iterations)
        .def_rw("energy_threshold", &SCFConfig::energy_threshold)
        .def_rw("density_threshold", &SCFConfig::density_threshold)
        .def_rw("diis_threshold", &SCFConfig::diis_threshold)
        .def_rw("diis_max_vectors", &SCFConfig::diis_max_vectors)
        .def_rw("print_level", &SCFConfig::print_level)
        .def_rw("use_df", &SCFConfig::use_df)
        .def_rw("aux_basis_name", &SCFConfig::aux_basis_name);

    nb::class_<SCFResult>(m, "SCFResult")
        .def(nb::init<>())
        .def_rw("energy_electronic", &SCFResult::energy_electronic)
        .def_rw("energy_nuclear", &SCFResult::energy_nuclear)
        .def_rw("energy_total", &SCFResult::energy_total)
        .def_rw("orbital_energies_alpha", &SCFResult::orbital_energies_alpha)
        .def_rw("orbital_energies_beta", &SCFResult::orbital_energies_beta)
        .def_rw("C_alpha", &SCFResult::C_alpha)
        .def_rw("C_beta", &SCFResult::C_beta)
        .def_rw("P_alpha", &SCFResult::P_alpha)
        .def_rw("P_beta", &SCFResult::P_beta)
        .def_rw("F_alpha", &SCFResult::F_alpha)
        .def_rw("F_beta", &SCFResult::F_beta)
        .def_rw("iterations", &SCFResult::iterations)
        .def_rw("converged", &SCFResult::converged)
        .def_rw("n_occ_alpha", &SCFResult::n_occ_alpha)
        .def_rw("n_occ_beta", &SCFResult::n_occ_beta);




    nb::class_<UHF>(m, "UHF")
        .def("__init__", [](UHF *t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, PointGroup* pg, PetiteList* pl,
                            int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*){});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*){}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*){}) : nullptr;
            new (t) UHF(mol, basis, sp_int, sp_pg, sp_pl, na, nb, conf);
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("pg"), nb::arg("pl"), nb::arg("n_alpha"), nb::arg("n_beta"),
           nb::arg("config") = SCFConfig())
        .def("__init__", [](UHF *t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*){});
            new (t) UHF(mol, basis, sp_int, nullptr, nullptr, na, nb, conf);
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("n_alpha"), nb::arg("n_beta"), nb::arg("config") = SCFConfig())
        .def("compute", &UHF::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("energy", &UHF::energy);

    nb::class_<RHF>(m, "RHF")
        .def("__init__", [](RHF *t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, PointGroup* pg, PetiteList* pl,
                            const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*){});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*){}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*){}) : nullptr;
            new (t) RHF(mol, basis, sp_int, sp_pg, sp_pl, conf);
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("pg"), nb::arg("pl"), nb::arg("config") = SCFConfig())
        .def("__init__", [](RHF *t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*){});
            new (t) RHF(mol, basis, sp_int, nullptr, nullptr, conf);
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"), nb::arg("config") = SCFConfig())
        .def("compute", &RHF::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("energy", &RHF::energy);

    nb::class_<ROHF>(m, "ROHF")
        .def("__init__", [](ROHF *t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, PointGroup* pg, PetiteList* pl,
                            int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*){});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*){}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*){}) : nullptr;
            new (t) ROHF(mol, basis, sp_int, sp_pg, sp_pl, na, nb, conf);
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("pg"), nb::arg("pl"), nb::arg("n_alpha"), nb::arg("n_beta"), nb::arg("config") = SCFConfig())
        .def("__init__", [](ROHF *t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*){});
            new (t) ROHF(mol, basis, sp_int, nullptr, nullptr, na, nb, conf);
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("n_alpha"), nb::arg("n_beta"), nb::arg("config") = SCFConfig())
        .def("compute", &ROHF::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("energy", &ROHF::energy);




    nb::class_<MP2Config>(m, "MP2Config")
        .def(nb::init<>())
        .def_rw("scf_type", &MP2Config::scf_type)
        .def_rw("eri_method", &MP2Config::eri_method)
        .def_rw("use_df", &MP2Config::use_df)
        .def_rw("aux_basis_name", &MP2Config::aux_basis_name)
        .def_rw("df_threshold", &MP2Config::df_threshold)
        .def_rw("cholesky_threshold", &MP2Config::cholesky_threshold)
        .def_rw("max_iterations", &MP2Config::max_iterations)
        .def_rw("energy_threshold", &MP2Config::energy_threshold)
        .def_rw("gradient_threshold", &MP2Config::gradient_threshold)
        .def_rw("print_level", &MP2Config::print_level)
        .def_rw("opt_method", &MP2Config::opt_method);

    nb::class_<MP2Result>(m, "MP2Result")
        .def(nb::init<>())
        .def_rw("energy_scf", &MP2Result::energy_scf)
        .def_rw("energy_mp2_ss", &MP2Result::energy_mp2_ss)
        .def_rw("energy_mp2_os", &MP2Result::energy_mp2_os)
        .def_rw("energy_mp2_corr", &MP2Result::energy_mp2_corr)
        .def_rw("energy_total", &MP2Result::energy_total)
        .def_rw("converged", &MP2Result::converged)
        .def_rw("iterations", &MP2Result::iterations)
        .def_rw("n_occ_alpha", &MP2Result::n_occ_alpha)
        .def_rw("n_occ_beta", &MP2Result::n_occ_beta)
        .def_rw("n_virt_alpha", &MP2Result::n_virt_alpha)
        .def_rw("n_virt_beta", &MP2Result::n_virt_beta)
        .def_rw("C_alpha", &MP2Result::C_alpha)
        .def_rw("C_beta", &MP2Result::C_beta)
        .def_rw("orbital_energies_alpha", &MP2Result::orbital_energies_alpha)
        .def_rw("orbital_energies_beta", &MP2Result::orbital_energies_beta);

    nb::class_<foundation::RMP2>(m, "RMP2")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>,
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &foundation::RMP2::compute, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<mshqc::UMP2>(m, "UMP2")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>,
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &mshqc::UMP2::compute, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<OMP2>(m, "OMP2")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>,
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &OMP2::compute, nb::call_guard<nb::gil_scoped_release>(), "Run OMP2 optimization");



    nb::class_<MP3Result>(m, "MP3Result")
        .def(nb::init<>())
        .def_rw("e_hf", &MP3Result::e_hf)
        .def_rw("e_mp2", &MP3Result::e_mp2)
        .def_rw("e_mp3", &MP3Result::e_mp3)
        .def_rw("e3_aa", &MP3Result::e3_aa)
        .def_rw("e3_bb", &MP3Result::e3_bb)
        .def_rw("e3_ab", &MP3Result::e3_ab)
        .def_rw("e_corr_total", &MP3Result::e_corr_total)
        .def_rw("e_total", &MP3Result::e_total)
        .def_rw("converged", &MP3Result::converged)
        .def_rw("iterations", &MP3Result::iterations);

    nb::class_<RMP3>(m, "RMP3")
        .def(nb::init<const SCFResult&, const MP2Result&, const MP2Config&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("scf_guess"), nb::arg("mp2_guess"), nb::arg("config"), nb::arg("integrals"))
        .def("compute", &RMP3::compute, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<UMP3>(m, "UMP3")
        .def(nb::init<const SCFResult&, const MP2Result&, const MP2Config&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("scf_guess"), nb::arg("mp2_guess"), nb::arg("config"), nb::arg("integrals"))
        .def("compute", &UMP3::compute, nb::call_guard<nb::gil_scoped_release>());


    nb::class_<OMP3>(m, "OMP3")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>,
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &OMP3::compute_omp3, nb::call_guard<nb::gil_scoped_release>(), "Run Orbital-Optimized MP3");



    nb::class_<gradient::GradientResult>(m, "GradientResult")
        .def(nb::init<>())
        .def_rw("energy", &gradient::GradientResult::energy)
        .def_rw("gradient", &gradient::GradientResult::gradient);

    nb::class_<gradient::OptConfig>(m, "OptConfig")
        .def(nb::init<>())
        .def_rw("max_iterations", &gradient::OptConfig::max_iterations);

    nb::class_<gradient::OptResult>(m, "OptResult")
        .def(nb::init<>())
        .def_rw("converged", &gradient::OptResult::converged)
        .def_rw("n_iterations", &gradient::OptResult::n_iterations)
        .def_rw("final_energy", &gradient::OptResult::final_energy);




    m.def("bohr_to_angstrom", [](double bohr) { return bohr * 0.529177210903; }, "Convert Bohr to Angstrom");
    m.def("angstrom_to_bohr", [](double angstrom) { return angstrom / 0.529177210903; }, "Convert Angstrom to Bohr");
    m.def("hartree_to_ev", [](double hartree) { return hartree * 27.211386245988; }, "Convert Hartree to eV");
    m.def("hartree_to_kcal", [](double hartree) { return hartree * 627.5094740631; }, "Convert Hartree to kcal/mol");


    m.def("map_jit_tensor_2d", [](uintptr_t ptr_address, size_t rows, size_t cols) {
        double* raw_ptr = reinterpret_cast<double*>(ptr_address);
        size_t shape[2] = {rows, cols};
        return nb::ndarray<nb::numpy, double, nb::c_contig>(
            raw_ptr, 2, shape);
    }, "Ekstraksi memori L-Value JIT 2D menjadi NumPy array zero-copy",
       nb::arg("ptr_address"), nb::arg("rows"), nb::arg("cols"));


    m.def("map_jit_tensor", [](uintptr_t ptr_address, size_t n1, size_t n2, size_t n3, size_t n4) {
        double* raw_ptr = reinterpret_cast<double*>(ptr_address);
        size_t shape[4] = {n1, n2, n3, n4};


        return nb::ndarray<nb::numpy, double, nb::c_contig>(
            raw_ptr, 4, shape);
    }, "Fungsi ekstraksi memori JIT absolut",
       nb::arg("ptr_address"), nb::arg("n1"), nb::arg("n2"), nb::arg("n3"), nb::arg("n4"));
}
