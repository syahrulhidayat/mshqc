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
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <omp.h>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
  #include <windows.h>
#elif defined(__APPLE__)
  #include <dlfcn.h>
  #include <sys/sysctl.h>
  #include <sys/types.h>
#elif defined(__linux__)
  #include <dlfcn.h>
  #include <fstream>
  #include <sched.h>
  #include <unordered_set>
#endif

#include "mshqc/basis.h"
#include "mshqc/core/fock_builder.h"
#include "mshqc/core/molecule.h"
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/integrals/screening.h"
#include "mshqc/ints/integrals.h"

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
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    }
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buffer.data(), &length)) {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    }
    int physical_cores = 0;
    for (const auto& info : buffer) {
        if (info.Relationship == RelationProcessorCore) physical_cores++;
    }
    return physical_cores > 0
        ? physical_cores
        : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
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
        if (!unique_cores.empty()) return static_cast<int>(unique_cores.size());
    }
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#else
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
#endif
}

void initialize_hpc_environment() {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    unsigned int logical_cpus = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned int i = 0; i < logical_cpus; ++i) CPU_SET(i, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
#endif

    int num_threads = get_physical_cores();
    if (const char* env_omp = std::getenv("OMP_NUM_THREADS")) {
        int parsed = std::atoi(env_omp);
        if (parsed > 0) num_threads = parsed;
    }

    std::string threads_str = std::to_string(num_threads);
    set_env_if_empty("OMP_NUM_THREADS", threads_str.c_str());
#if defined(__APPLE__) || defined(_WIN32)
  #if defined(_WIN32)
    _putenv_s("TBLIS_NUM_THREADS", "1");
    _putenv_s("MKL_NUM_THREADS", "1");
    _putenv_s("OPENBLAS_NUM_THREADS", "1");
  #else
    set_env_if_empty("TBLIS_NUM_THREADS", "1");
  #endif
    set_env_if_empty("OMP_WAIT_POLICY", "PASSIVE");
    set_env_if_empty("KMP_BLOCKTIME", "0");
#else
    set_env_if_empty("TBLIS_NUM_THREADS", threads_str.c_str());
#endif
    set_env_if_empty("OPENBLAS_NUM_THREADS", "1");
    set_env_if_empty("MKL_NUM_THREADS", "1");
    set_env_if_empty("VECLIB_MAXIMUM_THREADS", "1");
    set_env_if_empty("NUMEXPR_NUM_THREADS", "1");
    set_env_if_empty("OMP_MAX_ACTIVE_LEVELS", "1");
    set_env_if_empty("HDF5_USE_FILE_LOCKING", "FALSE");

    Eigen::setNbThreads(1);
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    omp_set_max_active_levels(1);

#if defined(_WIN32)
    using set_threads_fn = void (*)(int);
    for (const char* dll : {"libtblis.dll", "tblis.dll"}) {
        if (HMODULE h = GetModuleHandleA(dll)) {
            if (auto fn = reinterpret_cast<set_threads_fn>(GetProcAddress(h, "tblis_set_num_threads"))) fn(1);
        }
    }
    for (const char* dll : {"mkl_rt.2.dll", "mkl_rt.3.dll", "mkl_rt.dll"}) {
        if (HMODULE h = GetModuleHandleA(dll)) {
            if (auto fn = reinterpret_cast<set_threads_fn>(GetProcAddress(h, "MKL_Set_Num_Threads"))) fn(1);
        }
    }
#elif defined(__linux__) || defined(__APPLE__)
    using set_threads_fn = void (*)(int);
    if (auto fn = reinterpret_cast<set_threads_fn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"))) fn(1);
    if (auto fn = reinterpret_cast<set_threads_fn>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads"))) fn(1);
    if (auto fn = reinterpret_cast<set_threads_fn>(dlsym(RTLD_DEFAULT, "tblis_set_num_threads"))) {
  #if defined(__APPLE__)
        fn(1);
  #else
        fn(num_threads);
  #endif
    }
#endif
}

int symbol_to_z(const std::string& sym) {
    static const std::unordered_map<std::string, int> table = {
        {"H",1},{"He",2},{"Li",3},{"Be",4},{"B",5},{"C",6},{"N",7},{"O",8},{"F",9},{"Ne",10},
        {"Na",11},{"Mg",12},{"Al",13},{"Si",14},{"P",15},{"S",16},{"Cl",17},{"Ar",18},
        {"K",19},{"Ca",20},{"Sc",21},{"Ti",22},{"V",23},{"Cr",24},{"Mn",25},{"Fe",26},
        {"Co",27},{"Ni",28},{"Cu",29},{"Zn",30},{"Ga",31},{"Ge",32},{"As",33},{"Se",34},
        {"Br",35},{"Kr",36},{"Rb",37},{"Sr",38},{"Y",39},{"Zr",40},{"Nb",41},{"Mo",42},
        {"Tc",43},{"Ru",44},{"Rh",45},{"Pd",46},{"Ag",47},{"Cd",48},{"In",49},{"Sn",50},
        {"Sb",51},{"Te",52},{"I",53},{"Xe",54},
        {"Cs",55},{"Ba",56},{"La",57},{"Ce",58},{"Pr",59},{"Nd",60},{"Pm",61},{"Sm",62},{"Eu",63},{"Gd",64},
        {"Tb",65},{"Dy",66},{"Ho",67},{"Er",68},{"Tm",69},{"Yb",70},{"Lu",71},
        {"Hf",72},{"Ta",73},{"W",74},{"Re",75},{"Os",76},{"Ir",77},{"Pt",78},{"Au",79},{"Hg",80},
        {"Tl",81},{"Pb",82},{"Bi",83},{"Po",84},{"At",85},{"Rn",86},
        {"Fr",87},{"Ra",88},{"Ac",89},{"Th",90},{"Pa",91},{"U",92},{"Np",93},{"Pu",94},
        {"Am",95},{"Cm",96},{"Bk",97},{"Cf",98},{"Es",99},{"Fm",100},{"Md",101},{"No",102},
        {"Lr",103},{"Rf",104},{"Db",105},{"Sg",106},{"Bh",107},{"Hs",108},{"Mt",109},{"Ds",110},
        {"Rg",111},{"Cn",112},{"Nh",113},{"Fl",114},{"Mc",115},{"Lv",116},{"Ts",117},{"Og",118}
    };
    if (sym.empty()) throw std::invalid_argument("Empty atom symbol");
    std::string key = sym;
    key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
    for (size_t i = 1; i < key.size(); ++i) {
        key[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(key[i])));
    }
    auto it = table.find(key);
    if (it == table.end()) throw std::invalid_argument("Unknown element symbol: " + sym);
    return it->second;
}

double unit_scale(const std::string& units) {
    if (!units.empty() && (units[0] == 'a' || units[0] == 'A')) {
        return 1.0 / 0.529177210903;
    }
    return 1.0;
}

int standard_frozen_core_count(const Molecule& mol) {
    int nf = 0;
    for (size_t i = 0; i < mol.n_atoms(); ++i) {
        const int Z = mol.atom(i).atomic_number;
        if (Z <= 2) nf += 0;
        else if (Z <= 10) nf += 1;
        else if (Z <= 18) nf += 5;
        else if (Z <= 36) nf += 9;
        else if (Z <= 54) nf += 18;
        else if (Z <= 71) nf += 27;
        else if (Z <= 86) nf += 34;
        else nf += 43;
    }
    return nf;
}

} // namespace mshqc_auto_tune

namespace mshqc {
void bind_kernels_module(nanobind::module_& m);
#ifndef MSHQC_THIN_CORE_BUILD
void bind_legacy_module(nanobind::module_& m, nanobind::class_<Molecule>& mol_cls);
#endif
} // namespace mshqc

NB_MODULE(_mshqc, m) {
    mshqc_auto_tune::initialize_hpc_environment();
    m.doc() = "MSHQC: Thin-Core C++20 / Python Quantum Chemistry Engine";

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

    nb::class_<Molecule> mol_cls(m, "Molecule");
    mol_cls
        .def(nb::init<int, int>(), nb::arg("charge") = 0, nb::arg("multiplicity") = 1)
        .def("add_atom", [](Molecule& mol, int Z, double x, double y, double z, const std::string& units) {
            double s = mshqc_auto_tune::unit_scale(units);
            mol.add_atom(Z, x * s, y * s, z * s);
        }, nb::arg("Z"), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("units") = "bohr")
        .def("add_atom", [](Molecule& mol, const std::string& symbol, double x, double y, double z, const std::string& units) {
            double s = mshqc_auto_tune::unit_scale(units);
            int Z = mshqc_auto_tune::symbol_to_z(symbol);
            mol.add_atom(Z, x * s, y * s, z * s);
        }, nb::arg("symbol"), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("units") = "bohr")
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
        .def("nuclear_repulsion_energy", &Molecule::nuclear_repulsion_energy)
        .def("set_atom_position", &Molecule::set_atom_position,
             nb::arg("i"), nb::arg("x"), nb::arg("y"), nb::arg("z"));

    nb::enum_<AngularMomentum>(m, "AngularMomentum")
        .value("S", AngularMomentum::S)
        .value("P", AngularMomentum::P)
        .value("D", AngularMomentum::D)
        .value("F", AngularMomentum::F)
        .value("G", AngularMomentum::G)
        .value("H", AngularMomentum::H)
        .value("I", AngularMomentum::I)
        .value("K", AngularMomentum::K);

    nb::class_<GaussianPrimitive>(m, "GaussianPrimitive")
        .def(nb::init<double, double>(), nb::arg("exponent"), nb::arg("coefficient"))
        .def_rw("exponent", &GaussianPrimitive::exponent)
        .def_rw("coefficient", &GaussianPrimitive::coefficient);

    nb::class_<Shell>(m, "Shell")
        .def(nb::init<AngularMomentum, int, const std::array<double, 3>&>(),
             nb::arg("am"), nb::arg("center"), nb::arg("center_pos"))
        .def_prop_ro("angular_momentum", &Shell::angular_momentum)
        .def_prop_ro("center", &Shell::center)
        .def_prop_ro("primitives", &Shell::primitives)
        .def("l", &Shell::l)
        .def("n_functions", &Shell::n_functions);

    nb::class_<BasisSet>(m, "BasisSet")
        .def(nb::init<>())
        .def(nb::init<const std::string&, const Molecule&, const std::string&>(),
             nb::arg("basis_name"), nb::arg("mol"), nb::arg("basis_dir") = "data/basis")
        .def("read_gbs", &BasisSet::read_gbs, nb::arg("basis_file"), nb::arg("mol"))
        .def("add_shell", &BasisSet::add_shell, nb::arg("shell"))
        .def("n_shells", &BasisSet::n_shells)
        .def("n_basis_functions", &BasisSet::n_basis_functions)
        .def("n_basis", &BasisSet::n_basis_functions)
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
        .def("__init__", [](CholeskyERI* chol, const Molecule& mol, const BasisSet& basis, double threshold) {
            auto ints = std::make_shared<IntegralEngine>(mol, basis);
            new (chol) CholeskyERI(basis, ints);
            chol->set_threshold(threshold);
            chol->compute();
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("threshold") = 1e-9)
        .def("set_threshold", &CholeskyERI::set_threshold)
        .def("set_print_level", &CholeskyERI::set_print_level)
        .def("n_vectors", &CholeskyERI::n_vectors)
        .def("n_cholesky_vectors", &CholeskyERI::n_vectors)
        .def("threshold", &CholeskyERI::threshold)
        .def("compute", &CholeskyERI::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("decompose", &CholeskyERI::decompose, nb::call_guard<nb::gil_scoped_release>())
        .def("get_L_mat", &CholeskyERI::get_L_mat, nb::rv_policy::reference_internal)
        .def("reconstruct", &CholeskyERI::reconstruct)
        .def("reconstruct_full", &CholeskyERI::reconstruct_full)
        .def("is_decomposed", &CholeskyERI::is_decomposed)
        .def("decomposed", &CholeskyERI::decomposed);

    nb::class_<ShellPair>(m, "ShellPair")
        .def_rw("sh_a", &ShellPair::sh_a)
        .def_rw("sh_b", &ShellPair::sh_b)
        .def_rw("max_val", &ShellPair::max_val)
        .def_prop_rw(
            "center",
            [](const ShellPair& sp) { return std::make_tuple(sp.center(0), sp.center(1), sp.center(2)); },
            [](ShellPair& sp, const std::tuple<double, double, double>& t) {
                sp.center = Eigen::Vector3d(std::get<0>(t), std::get<1>(t), std::get<2>(t));
            });

    nb::class_<Screening>(m, "Screening")
        .def(nb::init<const BasisSet&>(), nb::arg("basis"))
        .def("reset", &Screening::reset)
        .def("compute", &Screening::compute, nb::call_guard<nb::gil_scoped_release>(), nb::arg("integrals"))
        .def("print_stats", &Screening::print_stats, nb::arg("threshold"))
        .def("get_significant_pairs", &Screening::get_significant_pairs, nb::arg("threshold"))
        .def("is_significant",
             static_cast<bool (Screening::*)(int, int, int, int, double) const>(&Screening::is_significant),
             nb::arg("sh_a"), nb::arg("sh_b"), nb::arg("sh_c"), nb::arg("sh_d"), nb::arg("threshold"))
        .def("get_schwarz_val", &Screening::get_schwarz_val)
        .def("max_schwarz", &Screening::max_schwarz);

    m.def("bohr_to_angstrom", [](double bohr) { return bohr * 0.529177210903; }, "Convert Bohr to Angstrom");
    m.def("angstrom_to_bohr", [](double angstrom) { return angstrom / 0.529177210903; }, "Convert Angstrom to Bohr");
    m.def("hartree_to_ev", [](double hartree) { return hartree * 27.211386245988; }, "Convert Hartree to eV");
    m.def("hartree_to_kcal", [](double hartree) { return hartree * 627.5094740631; }, "Convert Hartree to kcal/mol");
    m.def("standard_frozen_core", &mshqc_auto_tune::standard_frozen_core_count, nb::arg("mol"),
          "Compute standard noble-gas frozen core orbital count (Psi4 convention)");

    mshqc::bind_kernels_module(m);
#ifndef MSHQC_THIN_CORE_BUILD
    mshqc::bind_legacy_module(m, mol_cls);
#endif
}
