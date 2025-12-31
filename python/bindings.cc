// python/bindings.cc - Complete Python bindings for MSHQC library
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <unsupported/Eigen/CXX11/Tensor>

// Core headers
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/scf.h"
#include "mshqc/cholesky_uhf.h"
#include "mshqc/cholesky_rohf.h"
#include "mshqc/cholesky_rhf.h"

// MP headers
#include "mshqc/mp2.h"
#include "mshqc/ump2.h"
#include "mshqc/ump3.h"
#include "mshqc/cholesky_ump2.h"
#include "mshqc/cholesky_ump3.h"
#include "mshqc/foundation/rmp2.h"
#include "mshqc/foundation/rmp3.h"
#include "mshqc/omp3.h"
#include "mshqc/cholesky_omp2.h"
#include "mshqc/cholesky_omp3.h"
#include "mshqc/cholesky_rmp2.h"
#include "mshqc/cholesky_rmp3.h"
#include "mshqc/foundation/wavefunction.h"

// CI headers
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/cis.h"
#include "mshqc/ci/cisd.h"
#include "mshqc/ci/cisdt.h"
#include "mshqc/ci/fci.h"
#include "mshqc/ci/cipsi.h"

// MCSCF headers
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/casscf.h"
#include "mshqc/mcscf/sa_casscf.h"
#include "mshqc/mcscf/cholesky_casscf.h"
#include "mshqc/mcscf/cholesky_sa_casscf.h"
#include "mshqc/mcscf/caspt2.h"
#include "mshqc/mcscf/cholesky_caspt2.h"
#include "mshqc/mcscf/cholesky_sa_caspt2.h"
#include "mshqc/mcscf/cholesky_sa_caspt3.h"
#include "mshqc/mcscf/cholesky_uno.h"

// Gradient headers
#include "mshqc/gradient/gradient.h"
#include "mshqc/gradient/optimizer.h"
// Integral headers
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/integrals/eri_transformer.h"


namespace py = pybind11;
using namespace mshqc;
using namespace mshqc::mcscf;
using namespace mshqc::integrals;


PYBIND11_MODULE(_mshqc, m) {
    m.doc() = "MSHQC: Modern Quantum Chemistry Library";



    using ERITensor = Eigen::Tensor<double, 4, 0, long>;

    py::class_<ERITensor>(m, "ERITensor")
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

    // ========================================================================
    // Core Classes: Molecule, Basis, Integrals
    // ========================================================================
    
    py::class_<Atom>(m, "Atom")
        .def(py::init<int, double, double, double>(),
             py::arg("atomic_number"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("atomic_number", &Atom::atomic_number)
        .def_readwrite("x", &Atom::x)
        .def_readwrite("y", &Atom::y)
        .def_readwrite("z", &Atom::z)
        .def("position", &Atom::position);

    py::class_<Molecule>(m, "Molecule")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("charge"), py::arg("multiplicity"))
        .def("add_atom", py::overload_cast<int, double, double, double>(&Molecule::add_atom),
             py::arg("Z"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("add_atom", py::overload_cast<const Atom&>(&Molecule::add_atom),
             py::arg("atom"))
        .def("n_atoms", &Molecule::n_atoms)
        .def("atom", &Molecule::atom, py::arg("i"))
        .def("atoms", &Molecule::atoms)
        .def("total_nuclear_charge", &Molecule::total_nuclear_charge)
        .def("n_electrons", &Molecule::n_electrons)
        .def("charge", &Molecule::charge)
        .def("set_charge", &Molecule::set_charge, py::arg("q"))
        .def("multiplicity", &Molecule::multiplicity)
        .def("set_multiplicity", &Molecule::set_multiplicity, py::arg("m"))
        .def("nuclear_repulsion_energy", &Molecule::nuclear_repulsion_energy);

    py::enum_<AngularMomentum>(m, "AngularMomentum")
        .value("S", AngularMomentum::S)
        .value("P", AngularMomentum::P)
        .value("D", AngularMomentum::D)
        .value("F", AngularMomentum::F)
        .value("G", AngularMomentum::G)
        .value("H", AngularMomentum::H);

    py::class_<GaussianPrimitive>(m, "GaussianPrimitive")
        .def(py::init<double, double>(), py::arg("exponent"), py::arg("coefficient"))
        .def_readwrite("exponent", &GaussianPrimitive::exponent)
        .def_readwrite("coefficient", &GaussianPrimitive::coefficient);

    // Shell class - use property for methods that return values
    py::class_<Shell>(m, "Shell")
        .def(py::init<AngularMomentum, int, const std::array<double, 3>&>(),
             py::arg("am"), py::arg("center"), py::arg("center_pos"))
        .def_property_readonly("angular_momentum", &Shell::angular_momentum)
        .def_property_readonly("center", &Shell::center)
        .def_property_readonly("primitives", &Shell::primitives);

    py::class_<BasisSet>(m, "BasisSet")
        .def(py::init<>())
        .def(py::init<const std::string&, const Molecule&, const std::string&>(),
             py::arg("basis_name"), py::arg("mol"), py::arg("basis_dir") = "data/basis")
        .def("read_gbs", &BasisSet::read_gbs, py::arg("basis_file"), py::arg("mol"))
        .def("add_shell", &BasisSet::add_shell, py::arg("shell"))
        .def("n_shells", &BasisSet::n_shells)
        .def("n_basis_functions", &BasisSet::n_basis_functions)
        .def("shell", &BasisSet::shell, py::arg("i"))
        .def("name", &BasisSet::name);

    py::class_<IntegralEngine, std::shared_ptr<IntegralEngine>>(m, "IntegralEngine")
        .def(py::init<const Molecule&, const BasisSet&>(),
             py::arg("mol"), py::arg("basis"))
        .def("compute_overlap", &IntegralEngine::compute_overlap)
        .def("compute_kinetic", &IntegralEngine::compute_kinetic)
        .def("compute_nuclear", &IntegralEngine::compute_nuclear)
        .def("compute_eri", &IntegralEngine::compute_eri)
        .def("compute_eri_diagonal", &IntegralEngine::compute_eri_diagonal,"Compute diagonal ERI elements (pq|pq)")
        .def("compute_eri_column", &IntegralEngine::compute_eri_column, "Compute specific ERI column (pq|rs) for a pivot index",py::arg("pivot_index"));
    
    py::class_<CholeskyDecompositionResult>(m, "CholeskyDecompositionResult")
        .def_readonly("n_vectors", &CholeskyDecompositionResult::n_vectors)
        .def_readonly("n_basis", &CholeskyDecompositionResult::n_basis)
        .def_readonly("threshold", &CholeskyDecompositionResult::threshold)
        .def_readonly("compression_ratio", &CholeskyDecompositionResult::compression_ratio)
        .def_readonly("converged", &CholeskyDecompositionResult::converged);

    py::class_<CholeskyERI>(m, "CholeskyERI")
        .def(py::init<double>(), py::arg("threshold") = 1e-6)
        .def("decompose", &CholeskyERI::decompose)
        .def("get_L_vectors", &CholeskyERI::get_L_vectors, py::return_value_policy::reference_internal)
        .def("reconstruct", &CholeskyERI::reconstruct)
        .def("n_vectors", &CholeskyERI::n_vectors)
        .def("threshold", &CholeskyERI::threshold);

    // ========================================================================
    // SCF: Configuration, Results, and Solvers
    // ========================================================================
    
    py::class_<SCFConfig>(m, "SCFConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &SCFConfig::max_iterations)
        .def_readwrite("energy_threshold", &SCFConfig::energy_threshold)
        .def_readwrite("density_threshold", &SCFConfig::density_threshold)
        .def_readwrite("diis_threshold", &SCFConfig::diis_threshold)
        .def_readwrite("diis_max_vectors", &SCFConfig::diis_max_vectors)
        .def_readwrite("print_level", &SCFConfig::print_level)
        .def_readwrite("level_shift", &SCFConfig::level_shift);

    py::class_<SCFResult>(m, "SCFResult")
        .def(py::init<>())
        .def_readwrite("energy_electronic", &SCFResult::energy_electronic)
        .def_readwrite("energy_nuclear", &SCFResult::energy_nuclear)
        .def_readwrite("energy_total", &SCFResult::energy_total)
        .def_readwrite("orbital_energies_alpha", &SCFResult::orbital_energies_alpha)
        .def_readwrite("orbital_energies_beta", &SCFResult::orbital_energies_beta)
        .def_readwrite("C_alpha", &SCFResult::C_alpha)
        .def_readwrite("C_beta", &SCFResult::C_beta)
        .def_readwrite("P_alpha", &SCFResult::P_alpha)
        .def_readwrite("P_beta", &SCFResult::P_beta)
        .def_readwrite("F_alpha", &SCFResult::F_alpha)
        .def_readwrite("F_beta", &SCFResult::F_beta)
        .def_readwrite("iterations", &SCFResult::iterations)
        .def_readwrite("converged", &SCFResult::converged)
        .def_readwrite("gradient_norm", &SCFResult::gradient_norm)
        .def_readwrite("n_occ_alpha", &SCFResult::n_occ_alpha)
        .def_readwrite("n_occ_beta", &SCFResult::n_occ_beta);

    py::class_<UHF>(m, "UHF")
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, int, int, const SCFConfig&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("n_alpha"), py::arg("n_beta"), 
             py::arg("config") = SCFConfig())
        .def("compute", &UHF::compute);
    
    py::class_<RHF>(m, "RHF")
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const SCFConfig&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("config") = SCFConfig())
        .def("compute", &RHF::compute);
    
    py::class_<ROHF>(m, "ROHF")
        .def(py::init<const Molecule&, const BasisSet&, int, int, const SCFConfig&>(),
             py::arg("molecule"), py::arg("basis"),
             py::arg("n_alpha"), py::arg("n_beta"),
             py::arg("config") = SCFConfig())
        .def("run", &ROHF::run);
    
    py::class_<CholeskyUHFConfig>(m, "CholeskyUHFConfig")
        .def(py::init<>())
        .def_readwrite("cholesky_threshold", &CholeskyUHFConfig::cholesky_threshold)
        .def_readwrite("print_level", &CholeskyUHFConfig::print_level)
        .def_readwrite("max_iterations", &CholeskyUHFConfig::max_iterations)
        .def_readwrite("energy_threshold", &CholeskyUHFConfig::energy_threshold);
        

    py::class_<CholeskyUHF>(m, "CholeskyUHF")
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, int, int,
                      const CholeskyUHFConfig&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("n_alpha"), py::arg("n_beta"),
             py::arg("config") = CholeskyUHFConfig())
        .def("compute", &CholeskyUHF::compute)
        .def("set_cholesky_vectors", &CholeskyUHF::set_cholesky_vectors);

    // ========================================================================
    // BINDING: CholeskyROHF (NEW)
    // ========================================================================

    py::class_<CholeskyROHFConfig, SCFConfig>(m, "CholeskyROHFConfig")
        .def(py::init<>())
        .def_readwrite("cholesky_threshold", &CholeskyROHFConfig::cholesky_threshold)
        .def_readwrite("screen_exchange", &CholeskyROHFConfig::screen_exchange);

    py::class_<CholeskyROHF>(m, "CholeskyROHF")
        // Constructor 1: Standard (Decompose from scratch)
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, int, int,
                      const CholeskyROHFConfig&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("n_alpha"), py::arg("n_beta"),
             py::arg("config") = CholeskyROHFConfig())
             
        // Constructor 2: Reuse Vectors (Untuk efisiensi pipeline)
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, int, int,
                      const CholeskyROHFConfig&,
                      const integrals::CholeskyERI&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("n_alpha"), py::arg("n_beta"),
             py::arg("config"), py::arg("existing_cholesky"))
             
        .def("compute", &CholeskyROHF::compute);
    
    
    // ========================================================================
    // BINDING: CholeskyRHF (Closed-Shell Restricted)
    // ========================================================================

    py::class_<CholeskyRHFConfig, SCFConfig>(m, "CholeskyRHFConfig")
        .def(py::init<>())
        .def_readwrite("cholesky_threshold", &CholeskyRHFConfig::cholesky_threshold)
        .def_readwrite("screen_exchange", &CholeskyRHFConfig::screen_exchange);

    py::class_<CholeskyRHF>(m, "CholeskyRHF")
        // Constructor 1: Standard (Decompose from scratch)
        // Note: RHF tidak butuh n_alpha/n_beta, dia hitung dari molekul
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>,
                      const CholeskyRHFConfig&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("config") = CholeskyRHFConfig())
             
        // Constructor 2: Reuse Vectors (High Efficiency)
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>,
                      const CholeskyRHFConfig&,
                      const integrals::CholeskyERI&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("config"), py::arg("existing_cholesky"))
             
        .def("compute", &CholeskyRHF::compute)
        .def("energy", &CholeskyRHF::energy);
    
    // ========================================================================
    // MP2/MP3 Methods
    // ========================================================================
    
    py::class_<OMP2Result>(m, "OMP2Result")
        .def(py::init<>())
        // Energi
        .def_readwrite("energy_scf", &OMP2Result::energy_scf)
        .def_readwrite("energy_mp2_ss", &OMP2Result::energy_mp2_ss)
        .def_readwrite("energy_mp2_os", &OMP2Result::energy_mp2_os)
        .def_readwrite("energy_mp2_corr", &OMP2Result::energy_mp2_corr)
        .def_readwrite("energy_total", &OMP2Result::energy_total)
        // Informasi Konvergensi (PENTING: Tambahkan ini)
        .def_readwrite("converged", &OMP2Result::converged)
        .def_readwrite("iterations", &OMP2Result::iterations)
        // Informasi Dimensi
        .def_readwrite("n_occ_alpha", &OMP2Result::n_occ_alpha)
        .def_readwrite("n_occ_beta", &OMP2Result::n_occ_beta)
        .def_readwrite("n_virt_alpha", &OMP2Result::n_virt_alpha)
        .def_readwrite("n_virt_beta", &OMP2Result::n_virt_beta)
        // Orbitals (Sangat penting agar OMP3 bisa mengambil hasil orbital OMP2)
        .def_readwrite("C_alpha", &OMP2Result::C_alpha)
        .def_readwrite("C_beta", &OMP2Result::C_beta)
        .def_readwrite("orbital_energies_alpha", &OMP2Result::orbital_energies_alpha)
        .def_readwrite("orbital_energies_beta", &OMP2Result::orbital_energies_beta);
    
    py::class_<OMP2>(m, "OMP2")
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const SCFResult&>(),
             py::arg("mol"), py::arg("basis"), 
             py::arg("integrals"), py::arg("scf_guess"))
        .def("compute", &OMP2::compute, "Run OMP2 optimization")
        .def("set_max_iterations", &OMP2::set_max_iterations)
        .def("set_convergence_threshold", &OMP2::set_convergence_threshold)
        .def("set_gradient_threshold", &OMP2::set_gradient_threshold);

    py::class_<UMP2Result>(m, "UMP2Result")
        .def(py::init<>())
        .def_readwrite("e_corr_ss_aa", &UMP2Result::e_corr_ss_aa)
        .def_readwrite("e_corr_ss_bb", &UMP2Result::e_corr_ss_bb)
        .def_readwrite("e_corr_os", &UMP2Result::e_corr_os)
        .def_readwrite("e_corr_total", &UMP2Result::e_corr_total)
        .def_readwrite("e_total", &UMP2Result::e_total);

    // UMP3Result binding
    py::class_<UMP3Result>(m, "UMP3Result")
        .def(py::init<>())
        .def_readwrite("e_uhf", &UMP3Result::e_uhf)
        .def_readwrite("e_mp2", &UMP3Result::e_mp2)
        .def_readwrite("e_mp3_corr", &UMP3Result::e_mp3)
        .def_readwrite("e_corr_total", &UMP3Result::e_corr_total)
        .def_readwrite("e_total", &UMP3Result::e_total)
        .def_readwrite("e3_aa", &UMP3Result::e3_aa)
        .def_readwrite("e3_bb", &UMP3Result::e3_bb)
        .def_readwrite("e3_ab", &UMP3Result::e3_ab);
    // 1. OMP3Result (Belum ada sebelumnya)
    py::class_<OMP3Result>(m, "OMP3Result")
        .def(py::init<>())
        .def_readwrite("energy_total", &OMP3Result::energy_total)
        .def_readwrite("energy_mp2_corr", &OMP3Result::energy_mp2_corr)
        .def_readwrite("energy_mp3_corr", &OMP3Result::energy_mp3_corr)
        .def_readwrite("energy_omp2", &OMP3Result::energy_omp2)
        .def_readwrite("energy_omp3", &OMP3Result::energy_omp3)
        .def_readwrite("converged", &OMP3Result::converged)
        .def_readwrite("iterations", &OMP3Result::iterations)
        // Export orbital untuk analisis
        .def_readwrite("orbital_energies_alpha", &OMP3Result::orbital_energies_alpha)
        .def_readwrite("orbital_energies_beta", &OMP3Result::orbital_energies_beta)
        .def_readwrite("C_alpha", &OMP3Result::C_alpha)
        .def_readwrite("C_beta", &OMP3Result::C_beta);

    // 2. OMP3 Class
    py::class_<OMP3>(m, "OMP3")
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const OMP2Result&>(),
             py::arg("mol"), py::arg("basis"), 
             py::arg("integrals"), py::arg("omp2_result"))
        .def("compute", &OMP3::compute, "Run OMP3 optimization")
        .def("set_max_iterations", &OMP3::set_max_iterations)
        .def("set_convergence_threshold", &OMP3::set_convergence_threshold);

    // Foundation namespace classes - check actual member names
    py::class_<foundation::RMP2Result>(m, "RMP2Result")
        .def(py::init<>())
        .def_readwrite("e_corr", &foundation::RMP2Result::e_corr)
        .def_readwrite("e_rhf", &foundation::RMP2Result::e_rhf)
        .def_readwrite("e_total", &foundation::RMP2Result::e_total);

    py::class_<foundation::RMP3Result>(m, "RMP3Result")
        .def(py::init<>())
        .def_readwrite("e_mp2", &foundation::RMP3Result::e_mp2)
        .def_readwrite("e_mp3", &foundation::RMP3Result::e_mp3)
        .def_readwrite("e_total", &foundation::RMP3Result::e_total);

    // MP2 classes

    // UMP2 - fixed constructor (needs BasisSet)
    py::class_<UMP2>(m, "UMP2")
        .def(py::init<const SCFResult&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("uhf_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &UMP2::compute);

    

    py::class_<foundation::RMP2>(m, "RMP2")
        .def(py::init<const SCFResult&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("rhf_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &foundation::RMP2::compute);

    py::class_<foundation::RMP3>(m, "RMP3")
        .def(py::init<const SCFResult&, const foundation::RMP2Result&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("rhf_result"), py::arg("rmp2_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &foundation::RMP3::compute);

    // UMP3 - fixed constructor (needs UMP2Result and BasisSet)
    py::class_<UMP3>(m, "UMP3")
        .def(py::init<const SCFResult&, const UMP2Result&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             py::arg("uhf_result"), py::arg("ump2_result"), py::arg("basis"), py::arg("integrals"))
        .def("compute", &UMP3::compute);
    
    // ========================================================================
    // BINDING: CholeskyUMP2
    // ========================================================================

    py::class_<CholeskyUMP2Config>(m, "CholeskyUMP2Config")
        .def(py::init<>())
        .def_readwrite("cholesky_threshold", &CholeskyUMP2Config::cholesky_threshold)
        .def_readwrite("use_on_the_fly", &CholeskyUMP2Config::use_on_the_fly)
        .def_readwrite("validate_energy", &CholeskyUMP2Config::validate_energy)
        .def_readwrite("print_level", &CholeskyUMP2Config::print_level);

    py::class_<CholeskyUMP2Result>(m, "CholeskyUMP2Result")
        .def_readonly("e_corr_ss_aa", &CholeskyUMP2Result::e_corr_ss_aa)
        .def_readonly("e_corr_ss_bb", &CholeskyUMP2Result::e_corr_ss_bb)
        .def_readonly("e_corr_os", &CholeskyUMP2Result::e_corr_os)
        .def_readonly("e_corr_total", &CholeskyUMP2Result::e_corr_total)
        .def_readonly("e_total", &CholeskyUMP2Result::e_total)
        .def_readonly("n_cholesky_vectors", &CholeskyUMP2Result::n_cholesky_vectors)
        .def_readonly("compression_ratio", &CholeskyUMP2Result::compression_ratio)
        .def_readonly("memory_mb", &CholeskyUMP2Result::memory_mb)
        .def_readonly("time_cholesky_s", &CholeskyUMP2Result::time_cholesky_s)
        .def_readonly("time_transform_s", &CholeskyUMP2Result::time_transform_s)
        .def_readonly("time_energy_s", &CholeskyUMP2Result::time_energy_s);

    py::class_<CholeskyUMP2>(m, "CholeskyUMP2")
        .def(py::init<const SCFResult&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const CholeskyUMP2Config&, const integrals::CholeskyERI&>())
        .def("compute", &CholeskyUMP2::compute, "Compute UMP2 energy using Cholesky vectors")
        .def("compute_t2_amplitudes", &CholeskyUMP2::compute_t2_amplitudes)
        .def("get_cholesky", &CholeskyUMP2::get_cholesky, py::return_value_policy::reference);

    // ========================================================================
    // BINDING: CholeskyUMP3
    // ========================================================================

    py::class_<CholeskyUMP3Config>(m, "CholeskyUMP3Config")
        .def(py::init<>())
        .def_readwrite("cholesky_threshold", &CholeskyUMP3Config::cholesky_threshold)
        .def_readwrite("use_ump2_result", &CholeskyUMP3Config::use_ump2_result)
        .def_readwrite("store_intermediates", &CholeskyUMP3Config::store_intermediates)
        .def_readwrite("print_level", &CholeskyUMP3Config::print_level);

    py::class_<CholeskyUMP3Result>(m, "CholeskyUMP3Result")
        .def_readonly("e_mp2_total", &CholeskyUMP3Result::e_mp2_total)
        .def_readonly("e_mp3_ss_aa", &CholeskyUMP3Result::e_mp3_ss_aa)
        .def_readonly("e_mp3_ss_bb", &CholeskyUMP3Result::e_mp3_ss_bb)
        .def_readonly("e_mp3_os", &CholeskyUMP3Result::e_mp3_os)
        .def_readonly("e_mp3_total", &CholeskyUMP3Result::e_mp3_total)
        .def_readonly("e_corr_total", &CholeskyUMP3Result::e_corr_total)
        .def_readonly("e_total", &CholeskyUMP3Result::e_total)
        .def_readonly("n_cholesky_vectors", &CholeskyUMP3Result::n_cholesky_vectors)
        .def_readonly("time_mp3_s", &CholeskyUMP3Result::time_mp3_s);

    py::class_<CholeskyUMP3>(m, "CholeskyUMP3")
        // Konstruktor dari scratch
        .def(py::init<const SCFResult&, const BasisSet&, std::shared_ptr<IntegralEngine>, const CholeskyUMP3Config&>())
        // Konstruktor efisien dengan reuse objek UMP2
        .def(py::init<const CholeskyUMP2&, const CholeskyUMP3Config&>())
        .def("compute", &CholeskyUMP3::compute, "Compute UMP3 correlation energy using Block-Contraction")
        .def("initialize_cholesky", &CholeskyUMP3::initialize_cholesky)
        .def("transform_cholesky_vectors", &CholeskyUMP3::transform_cholesky_vectors);



    // ========================================================================
    // BINDING: CholeskyOMP2
    // ========================================================================

    py::class_<CholeskyOMP2Config>(m, "CholeskyOMP2Config")
        .def(py::init<>())
        .def_readwrite("max_iterations", &CholeskyOMP2Config::max_iterations)
        .def_readwrite("energy_threshold", &CholeskyOMP2Config::energy_threshold)
        .def_readwrite("gradient_threshold", &CholeskyOMP2Config::gradient_threshold)
        .def_readwrite("cholesky_threshold", &CholeskyOMP2Config::cholesky_threshold)
        .def_readwrite("print_level", &CholeskyOMP2Config::print_level);

    py::class_<CholeskyOMP2>(m, "CholeskyOMP2")
        // Constructor 1: Standard (Decompose sendiri)
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const SCFResult&,
                      const CholeskyOMP2Config&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("scf_guess"), py::arg("config") = CholeskyOMP2Config())
             
        // Constructor 2: Reuse Vectors (Sangat Cepat)
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const SCFResult&,
                      const CholeskyOMP2Config&,
                      const integrals::CholeskyERI&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("scf_guess"), py::arg("config"), py::arg("existing_cholesky"))
             
        .def("compute", &CholeskyOMP2::compute);


    // ========================================================================
    // BINDING: CholeskyOMP3
    // ========================================================================

    py::class_<CholeskyOMP3Config>(m, "CholeskyOMP3Config")
        .def(py::init<>())
        .def_readwrite("max_iterations", &CholeskyOMP3Config::max_iterations)
        .def_readwrite("energy_threshold", &CholeskyOMP3Config::energy_threshold)
        .def_readwrite("gradient_threshold", &CholeskyOMP3Config::gradient_threshold)
        .def_readwrite("cholesky_threshold", &CholeskyOMP3Config::cholesky_threshold)
        .def_readwrite("print_level", &CholeskyOMP3Config::print_level);

    py::class_<CholeskyOMP3Result>(m, "CholeskyOMP3Result")
        .def(py::init<>())
        .def_readonly("energy_scf", &CholeskyOMP3Result::energy_scf)
        .def_readonly("energy_mp2_corr", &CholeskyOMP3Result::energy_mp2_corr)
        .def_readonly("energy_mp3_corr", &CholeskyOMP3Result::energy_mp3_corr)
        .def_readonly("energy_total", &CholeskyOMP3Result::energy_total)
        .def_readonly("converged", &CholeskyOMP3Result::converged)
        .def_readonly("iterations", &CholeskyOMP3Result::iterations)
        .def_readwrite("C_alpha", &CholeskyOMP3Result::C_alpha)
        .def_readwrite("C_beta", &CholeskyOMP3Result::C_beta)
        .def_readwrite("orbital_energies_alpha", &CholeskyOMP3Result::orbital_energies_alpha)
        .def_readwrite("orbital_energies_beta", &CholeskyOMP3Result::orbital_energies_beta);

    py::class_<CholeskyOMP3>(m, "CholeskyOMP3")
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, 
                      const OMP2Result&, 
                      const CholeskyOMP3Config&,
                      const integrals::CholeskyERI&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"),
             py::arg("omp2_guess"), py::arg("config"), py::arg("cholesky_vectors"))
        .def("compute", &CholeskyOMP3::compute);

    // ========================================================================
    // BINDING: CholeskyRMP2 (NEW)
    // ========================================================================

    py::class_<CholeskyRMP2Config>(m, "CholeskyRMP2Config")
        .def(py::init<>())
        .def_readwrite("cholesky_threshold", &CholeskyRMP2Config::cholesky_threshold)
        .def_readwrite("print_level", &CholeskyRMP2Config::print_level);

    py::class_<CholeskyRMP2>(m, "CholeskyRMP2")
        // Constructor 1: Standard
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const SCFResult&,
                      const CholeskyRMP2Config&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("rhf_result"), py::arg("config") = CholeskyRMP2Config())
             
        // Constructor 2: Reuse Vectors (Sangat Cepat)
        .def(py::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const SCFResult&,
                      const CholeskyRMP2Config&,
                      const integrals::CholeskyERI&>(),
             py::arg("molecule"), py::arg("basis"), py::arg("integrals"),
             py::arg("rhf_result"), py::arg("config"), py::arg("existing_cholesky"))
             
        .def("compute", &CholeskyRMP2::compute);

    // 1. Binding Struct CholeskyRMP2Result (Penting untuk Input RMP3)
    // Pastikan namespace sesuai dengan cholesky_rmp2.h (mshqc::foundation)
    py::class_<foundation::CholeskyRMP2Result>(m, "CholeskyRMP2Result")
        .def_readonly("e_rhf", &foundation::CholeskyRMP2Result::e_rhf)
        .def_readonly("e_corr", &foundation::CholeskyRMP2Result::e_corr)
        .def_readonly("e_total", &foundation::CholeskyRMP2Result::e_total)
        // Kita tidak perlu expose vector Cholesky ke Python secara raw, 
        // tapi objek ini harus ada agar bisa dipassing ke RMP3
        .def_readonly("n_chol_vectors", &foundation::CholeskyRMP2Result::n_chol_vectors);

    // 2. Binding Class CholeskyRMP3
    py::class_<foundation::CholeskyRMP3>(m, "CholeskyRMP3")
        .def(py::init<const SCFResult&, const foundation::CholeskyRMP2Result&, const BasisSet&>(),
             py::arg("rhf_result"), 
             py::arg("crmp2_result"), 
             py::arg("basis"))
        .def("compute", &foundation::CholeskyRMP3::compute);

    // ========================================================================
    // CI Methods
    // ========================================================================
    
    /*py::class_<ci::Determinant>(m, "Determinant")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&>(),
             py::arg("alpha_occ"), py::arg("beta_occ"))
        .def("n_alpha", &ci::Determinant::n_alpha)
        .def("n_beta", &ci::Determinant::n_beta)
        .def("excitation_level", &ci::Determinant::excitation_level, py::arg("other"));

    py::class_<ci::CIIntegrals>(m, "CIIntegrals")
        .def(py::init<>())
        .def_readwrite("e_nuc", &ci::CIIntegrals::e_nuc);

    // FCI Result - check actual structure
    py::class_<ci::FCIResult>(m, "FCIResult")
        .def(py::init<>())
        .def_readwrite("determinants", &ci::FCIResult::determinants)
        .def_readwrite("converged", &ci::FCIResult::converged);

    py::class_<ci::CIS>(m, "CIS")
        .def(py::init<const ci::CIIntegrals&, const ci::Determinant&, int, int, int, int>(),
             py::arg("ints"), py::arg("hf_det"), 
             py::arg("n_occ_alpha"), py::arg("n_occ_beta"),
             py::arg("n_virt_alpha"), py::arg("n_virt_beta"))
        .def("compute", &ci::CIS::compute, py::arg("n_roots") = 5);

    py::class_<ci::CISD>(m, "CISD")
        .def(py::init<const ci::CIIntegrals&, const ci::Determinant&, int, int, int, int>(),
             py::arg("ints"), py::arg("hf_det"),
             py::arg("n_occ_alpha"), py::arg("n_occ_beta"),
             py::arg("n_virt_alpha"), py::arg("n_virt_beta"))
        .def("compute", &ci::CISD::compute);

    py::class_<ci::CISDT>(m, "CISDT")
        .def(py::init<const ci::CIIntegrals&, const ci::Determinant&, int, int, int, int>(),
             py::arg("ints"), py::arg("hf_det"),
             py::arg("n_occ_alpha"), py::arg("n_occ_beta"),
             py::arg("n_virt_alpha"), py::arg("n_virt_beta"))
        .def("compute", &ci::CISDT::compute);

    py::class_<ci::FCI>(m, "FCI")
        .def(py::init<const ci::CIIntegrals&, int, int, int, int>(),
             py::arg("ints"), py::arg("n_orbitals"),
             py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_roots") = 1)
        .def("compute", &ci::FCI::compute);

    // CIPSI Result - minimal structure
    py::class_<ci::CIPSIResult>(m, "CIPSIResult")
        .def(py::init<>())
        .def_readwrite("converged", &ci::CIPSIResult::converged);

    py::class_<ci::CIPSI>(m, "CIPSI")
        .def(py::init<const ci::CIIntegrals&, int, int, int>(),
             py::arg("ints"), py::arg("n_orbitals"),
             py::arg("n_alpha"), py::arg("n_beta"))
        .def("compute", &ci::CIPSI::compute);*/

    // ========================================================================
    // MCSCF Methods
    // ========================================================================
    
    py::class_<mcscf::ActiveSpace>(m, "ActiveSpace")
        .def(py::init<>())
        .def(py::init<int, int, int, int>(),
             py::arg("n_inactive"), py::arg("n_active"), 
             py::arg("n_virtual"), py::arg("n_elec_active"))
        .def_static("CAS_Frozen", &mcscf::ActiveSpace::CAS_Frozen,
             py::arg("n_frozen_orb"), py::arg("n_active_orb"),
             py::arg("n_total_orb"), py::arg("n_total_elec"),
             "Create Active Space defining Frozen Orbitals")
        .def_static("CAS", &mcscf::ActiveSpace::CAS,
                   py::arg("n_elec"), py::arg("n_orb"),
                   py::arg("n_total_orb"), py::arg("n_total_elec"))
        .def("n_inactive", &mcscf::ActiveSpace::n_inactive)
        .def("n_active", &mcscf::ActiveSpace::n_active)
        .def("n_virtual", &mcscf::ActiveSpace::n_virtual)
        .def("n_elec_active", &mcscf::ActiveSpace::n_elec_active)
        // [TAMBAHAN]
        .def("inactive_indices", &mcscf::ActiveSpace::inactive_indices)
        .def("active_indices", &mcscf::ActiveSpace::active_indices)
        .def("virtual_indices", &mcscf::ActiveSpace::virtual_indices)
        .def("__repr__", &mcscf::ActiveSpace::to_string);
    py::class_<mcscf::CASResult>(m, "CASResult")
        .def(py::init<>())
        .def_readwrite("e_casscf", &mcscf::CASResult::e_casscf)
        .def_readwrite("e_nuclear", &mcscf::CASResult::e_nuclear)
        .def_readwrite("n_iterations", &mcscf::CASResult::n_iterations)
        .def_readwrite("converged", &mcscf::CASResult::converged)
        .def_readwrite("C_mo", &mcscf::CASResult::C_mo)
        .def_readwrite("orbital_energies", &mcscf::CASResult::orbital_energies)
        .def_readwrite("ci_coeffs", &mcscf::CASResult::ci_coeffs)
        .def_readwrite("determinants", &mcscf::CASResult::determinants)
        .def_readwrite("n_determinants", &mcscf::CASResult::n_determinants)
        .def_readwrite("active_space", &mcscf::CASResult::active_space);

    py::class_<mcscf::CASSCF>(m, "CASSCF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const mcscf::ActiveSpace&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), py::arg("active_space"))
        .def("compute", &mcscf::CASSCF::compute, py::arg("initial_guess"))
        // [TAMBAHAN PENTING] Agar bisa set iterasi seperti di C++
        .def("set_max_iterations", &mcscf::CASSCF::set_max_iterations)
        .def("set_energy_threshold", &mcscf::CASSCF::set_energy_threshold)
        .def("set_gradient_threshold", &mcscf::CASSCF::set_gradient_threshold)
        .def("set_ci_solver", &mcscf::CASSCF::set_ci_solver);
    py::class_<mcscf::CASPT2Result1>(m, "CASPT2Result1")
        .def(py::init<>())
        .def_readwrite("e_casscf", &mcscf::CASPT2Result1::e_casscf)
        .def_readwrite("e_pt2", &mcscf::CASPT2Result1::e_pt2)
        .def_readwrite("e_total", &mcscf::CASPT2Result1::e_total)
        .def_readwrite("ipea_shift_used", &mcscf::CASPT2Result1::ipea_shift_used)
        .def_readwrite("converged", &mcscf::CASPT2Result1::converged)
        .def_readwrite("status_message", &mcscf::CASPT2Result1::status_message);

    py::class_<mcscf::CASPT2>(m, "CASPT2")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const mcscf::CASResult&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), py::arg("casscf_result"))
        .def("compute", &mcscf::CASPT2::compute)
        .def("set_ipea_shift", &mcscf::CASPT2::set_ipea_shift)
        .def("set_imaginary_shift", &mcscf::CASPT2::set_imaginary_shift);
    
    py::class_<CholeskyCASSCF>(m, "CholeskyCASSCF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&>())
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&, const std::vector<Eigen::VectorXd>&>())
        .def("compute", &CholeskyCASSCF::compute);

   py::class_<UNOResult>(m, "UNOResult")
        .def(py::init<>())
        .def_readwrite("C_uno", &UNOResult::C_uno)
        .def_readwrite("occupations", &UNOResult::occupations)
        .def_readwrite("entropy", &UNOResult::entropy)
        // [TAMBAHAN]
        .def_readwrite("suggested_n_active", &UNOResult::suggested_n_active)
        .def_readwrite("suggested_n_electrons", &UNOResult::suggested_n_electrons)
        .def_readwrite("active_indices", &UNOResult::active_indices);

    py::class_<CholeskyUNO>(m, "CholeskyUNO")
        .def(py::init<const SCFResult&, std::shared_ptr<IntegralEngine>, int>())
        .def("compute", &CholeskyUNO::compute)
        .def("print_report", &CholeskyUNO::print_report, py::arg("threshold") = 0.02)
        .def("save_orbitals", &CholeskyUNO::save_orbitals);
    
    
    // State-Averaged CASSCF
    py::class_<SACASConfig>(m, "SACASConfig")
        .def(py::init<>())
        .def_readwrite("n_states", &SACASConfig::n_states)
        .def_readwrite("max_iter", &SACASConfig::max_iter)
        .def_readwrite("cholesky_thresh", &SACASConfig::cholesky_thresh)
        // TAMBAHAN BARU:
        .def_readwrite("weights", &SACASConfig::weights)
        .def_readwrite("e_thresh", &SACASConfig::e_thresh)
        .def_readwrite("grad_thresh", &SACASConfig::grad_thresh)
        .def_readwrite("print_level", &SACASConfig::print_level)
        .def_readwrite("rotation_damping", &SACASConfig::rotation_damping)
        .def_readwrite("shift", &SACASConfig::shift)
        .def("set_equal_weights", &SACASConfig::set_equal_weights);

    py::class_<SACASResult>(m, "SACASResult")
        .def(py::init<>())
        .def_readwrite("e_avg", &SACASResult::e_avg)
        .def_readwrite("state_energies", &SACASResult::state_energies)
        .def_readwrite("C_mo", &SACASResult::C_mo)
        .def_readwrite("orbital_energies", &SACASResult::orbital_energies) // <-- Penting untuk PT2/PT3
        .def_readwrite("converged", &SACASResult::converged)
        .def_readwrite("ci_vectors", &SACASResult::ci_vectors)      // Tambahan akses
        .def_readwrite("rdm1_states", &SACASResult::rdm1_states);   // Tambahan akses
    py::class_<CholeskySACASSCF>(m, "CholeskySACASSCF")
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&, const SACASConfig&>())
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&, const SACASConfig&, const std::vector<Eigen::VectorXd>&>())
        .def("compute", py::overload_cast<const SCFResult&>(&CholeskySACASSCF::compute))
        .def("compute", py::overload_cast<const Eigen::MatrixXd&>(&CholeskySACASSCF::compute));
    
    py::class_<CASPT2Config>(m, "CASPT2Config")
        .def(py::init<>())
        .def_readwrite("shift", &CASPT2Config::shift)
        .def_readwrite("print_level", &CASPT2Config::print_level)
        // [TAMBAHAN]
        .def_readwrite("zero_thresh", &CASPT2Config::zero_thresh)
        .def_readwrite("use_tblis", &CASPT2Config::use_tblis)
        .def_readwrite("export_amplitudes", &CASPT2Config::export_amplitudes);

    py::class_<CASPT2Result>(m, "CASPT2Result")
        .def(py::init<>())
        .def_readwrite("e_cas", &CASPT2Result::e_cas)
        .def_readwrite("e_pt2", &CASPT2Result::e_pt2)
        .def_readwrite("e_total", &CASPT2Result::e_total)
        // [TAMBAHAN]
        .def_readwrite("amplitudes", &CASPT2Result::amplitudes);

    // Register Dummy/Opaque MOIntegrals if needed, or just hide it
    // py::class_<MOIntegrals, std::shared_ptr<MOIntegrals>>(m, "MOIntegrals"); 

    py::class_<CholeskySACASPT2>(m, "CholeskySACASPT2")
        .def(py::init<const SACASResult&, const std::vector<Eigen::VectorXd>&, int, const ActiveSpace&, const CASPT2Config&>())
        // FIX: Explicit overload for no arguments
        .def("compute", [](CholeskySACASPT2& self) {
            return self.compute(nullptr); 
        }, "Compute CASPT2 without precomputed MO integrals")
        // Optional: Keep the original if you ever bind MOIntegrals
        // .def("compute", &CholeskySACASPT2::compute, py::arg("mo_ints")) 
        ;
    // --- Cholesky Single-State CASPT2 Bindings ---
    py::class_<mcscf::CholeskyCASPT2Result>(m, "CholeskyCASPT2Result")
        .def(py::init<>())
        .def_readwrite("e_casscf", &mcscf::CholeskyCASPT2Result::e_casscf)
        .def_readwrite("e_pt2", &mcscf::CholeskyCASPT2Result::e_pt2)
        .def_readwrite("e_total", &mcscf::CholeskyCASPT2Result::e_total)
        .def_readwrite("time_total_s", &mcscf::CholeskyCASPT2Result::time_total_s)
        .def_readwrite("converged", &mcscf::CholeskyCASPT2Result::converged);

    py::class_<mcscf::CholeskyCASPT2>(m, "CholeskyCASPT2")
        // Constructor Standard
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const CASResult&, double>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), 
             py::arg("result"), py::arg("threshold") = 1e-6)
        // Constructor Reuse Cholesky (Penting untuk pipeline)
        .def(py::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const CASResult&, const integrals::CholeskyERI&>(),
             py::arg("mol"), py::arg("basis"), py::arg("integrals"), 
             py::arg("result"), py::arg("cholesky_eri"))
        .def("compute", &mcscf::CholeskyCASPT2::compute)
        .def("set_ipea_shift", &mcscf::CholeskyCASPT2::set_ipea_shift)
        .def("set_imaginary_shift", &mcscf::CholeskyCASPT2::set_imaginary_shift);

    // Cholesky SA-CASPT3
    py::class_<CASPT3Config>(m, "CASPT3Config")
        .def(py::init<>())
        .def_readwrite("shift", &CASPT3Config::shift)
        .def_readwrite("zero_thresh", &CASPT3Config::zero_thresh) // <-- Tambahkan jika belum ada
        .def_readwrite("print_level", &CASPT3Config::print_level);

    py::class_<CASPT3Result>(m, "CASPT3Result")
        .def(py::init<>())
        .def_readwrite("e_cas", &CASPT3Result::e_cas)
        .def_readwrite("e_pt2", &CASPT3Result::e_pt2)
        .def_readwrite("e_pt3", &CASPT3Result::e_pt3) // Ini std::vector<double>
        .def_readwrite("e_total", &CASPT3Result::e_total);

    py::class_<CholeskySACASPT3>(m, "CholeskySACASPT3")
        .def(py::init<const SACASResult&, const std::vector<Eigen::VectorXd>&, int, const ActiveSpace&, const CASPT3Config&>())
        .def("compute", &CholeskySACASPT3::compute);

    // ========================================================================
    // Gradient and Optimization
    // ========================================================================
    
    py::class_<gradient::GradientResult>(m, "GradientResult")
        .def(py::init<>())
        .def_readwrite("energy", &gradient::GradientResult::energy)
        .def_readwrite("gradient", &gradient::GradientResult::gradient);

    // AnalyticalGradient is abstract - don't expose constructor
    // Just document that it's used internally

    py::class_<gradient::OptConfig>(m, "OptConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &gradient::OptConfig::max_iterations);

    py::class_<gradient::OptResult>(m, "OptResult")
        .def(py::init<>())
        .def_readwrite("converged", &gradient::OptResult::converged)
        .def_readwrite("n_iterations", &gradient::OptResult::n_iterations)
        .def_readwrite("final_energy", &gradient::OptResult::final_energy);

    py::class_<PT2Amplitudes>(m, "PT2Amplitudes")
        .def(py::init<>())
        .def_readwrite("t2_core", &PT2Amplitudes::t2_core)
        .def_readwrite("t2_active", &PT2Amplitudes::t2_active)
        .def_readwrite("t2_semi1", &PT2Amplitudes::t2_semi1)
        .def_readwrite("t2_semi2", &PT2Amplitudes::t2_semi2);



    // GeometryOptimizer uses function callback - simplified binding
    // Users should create custom wrapper if needed

    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    m.def("bohr_to_angstrom", [](double bohr) { return bohr * 0.529177210903; },
          "Convert Bohr to Angstrom");
    
    m.def("angstrom_to_bohr", [](double angstrom) { return angstrom / 0.529177210903; },
          "Convert Angstrom to Bohr");
    
    m.def("hartree_to_ev", [](double hartree) { return hartree * 27.211386245988; },
          "Convert Hartree to eV");
    
    m.def("hartree_to_kcal", [](double hartree) { return hartree * 627.5094740631; },
          "Convert Hartree to kcal/mol");

    


}
