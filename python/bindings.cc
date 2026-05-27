#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/array.h>
#include <nanobind/eigen/dense.h>
// Core headers
#include "mshqc/molecule.h"
#include "mshqc/basis.h"
#include "mshqc/integrals.h"
#include "mshqc/symmetry/point_group.h"
#include "mshqc/symmetry/petite_list.h"
#include "mshqc/symmetry/molecule_sym.h"
#include "mshqc/scf.h"
#include "mshqc/diis.h"
#include "mshqc/integrals/screening.h"
#include "mshqc/core/fock_builder.h"


// MP headers
#include "mshqc/mp2.h"
//#include "mshqc/ump3.h"
//#include "mshqc/cholesky_ump2.h"
//#include "mshqc/cholesky_ump3.h"
//#include "mshqc/foundation/rmp3.h"
#include "mshqc/foundation/fcidump.h"
//#include "mshqc/omp3.h"
//#include "mshqc/cholesky_omp2.h"
//#include "mshqc/cholesky_omp3.h"
//#include "mshqc/cholesky_rmp2.h"
//#include "mshqc/cholesky_rmp3.h"
#include "mshqc/foundation/wavefunction.h"

// CI headers
#include "mshqc/ci/determinant.h"
#include "mshqc/ci/fci.h"


// MCSCF headers
#include "mshqc/mcscf/active_space.h"
#include "mshqc/mcscf/sa_casscf.h"
#include "mshqc/mcscf/cholesky_sa_casscf.h"
#include "mshqc/mcscf/cholesky_sa_caspt2.h"
#include "mshqc/mcscf/uno_result.h"
#include "mshqc/mcscf/cholesky_uno.h"
#include "mshqc/mcscf/canonical_uno.h"
#include "mshqc/mcscf/canonical_sa_casscf.h"
#include "mshqc/mcscf/canonical_sa_caspt2.h"

// Gradient headers
#include "mshqc/gradient/gradient.h"
#include "mshqc/gradient/optimizer.h"
// Integral headers
#include "mshqc/integrals/cholesky_eri.h"
#include "mshqc/integrals/eri_transformer.h"


namespace nb = nanobind;
using namespace mshqc;
using namespace mshqc::mcscf;
using namespace mshqc::integrals;


NB_MODULE(_mshqc, m) {
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

    // ========================================================================
    // Core Classes: Molecule, Basis, Integrals
    // ========================================================================
    
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
        .def("add_atom", nb::overload_cast<const Atom&>(&Molecule::add_atom),
             nb::arg("atom"))
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

    // Shell class - use property for methods that return values
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
        .def(nb::init<const Molecule&, const BasisSet&>(),
             nb::arg("mol"), nb::arg("basis"))
        .def("compute_overlap", &IntegralEngine::compute_overlap, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_kinetic", &IntegralEngine::compute_kinetic, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_nuclear", &IntegralEngine::compute_nuclear, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_eri", &IntegralEngine::compute_eri, nb::call_guard<nb::gil_scoped_release>())
        .def("compute_eri_diagonal", &IntegralEngine::compute_eri_diagonal,"Compute diagonal ERI elements (pq|pq)")
        .def("compute_eri_column", &IntegralEngine::compute_eri_column, "Compute specific ERI column (pq|rs) for a pivot index", nb::arg("pivot_index"));

    nb::class_<CholeskyDecompositionResult>(m, "CholeskyDecompositionResult")
        .def_ro("n_vectors", &CholeskyDecompositionResult::n_vectors)
        .def_ro("n_basis", &CholeskyDecompositionResult::n_basis)
        .def_ro("threshold", &CholeskyDecompositionResult::threshold)
        .def_ro("compression_ratio", &CholeskyDecompositionResult::compression_ratio)
        .def_ro("converged", &CholeskyDecompositionResult::converged);

    // ========================================================================
    // BINDING: CholeskyERI (UPDATED FOR DIRECT MODE)
    // ========================================================================
    nb::class_<CholeskyERI>(m, "CholeskyERI")
        // Constructor 1: Legacy (Threshold only)
        .def(nb::init<double>(), nb::arg("threshold") = 1e-9)
        
        // [NEW] Constructor 2: Direct Mode (Basis + Engine)
        .def(nb::init<const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("basis"), nb::arg("integrals"))

        // Setters/Getters
        .def("set_threshold", &integrals::CholeskyERI::set_threshold)
        .def("set_print_level", &integrals::CholeskyERI::set_print_level)
        .def("n_vectors", &integrals::CholeskyERI::n_vectors)
        .def("threshold", &integrals::CholeskyERI::threshold)
        
        // [NEW] Direct Compute Driver
        .def("compute", &integrals::CholeskyERI::compute, nb::call_guard<nb::gil_scoped_release>(),
             "Run optimized direct Cholesky decomposition")

        // Legacy Decompose (Tensor input)
        .def("decompose", &integrals::CholeskyERI::decompose, nb::call_guard<nb::gil_scoped_release>())
        
        // Accessors
        .def("get_L_mat", &integrals::CholeskyERI::get_L_mat, 
             nb::rv_policy::reference_internal)
        .def("reconstruct", &integrals::CholeskyERI::reconstruct)
        .def("reconstruct_full", &integrals::CholeskyERI::reconstruct_full)
        
        // Compatibility (Agar UMP3/lainnya tidak error)
        .def("is_decomposed", &integrals::CholeskyERI::is_decomposed)
        .def("decomposed", &integrals::CholeskyERI::decomposed);

   // ========================================================================
    // BINDING: Symmetry Structures & Classes (SINGLE BLOCK)
    // ========================================================================

    // 1. Enum SymOpType
    nb::enum_<SymOpType>(m, "SymOpType")
        .value("Identity", SymOpType::Identity)
        .value("Rotation", SymOpType::Rotation)
        .value("Reflection", SymOpType::Reflection)
        .value("Inversion", SymOpType::Inversion)
        .value("ImproperRotation", SymOpType::ImproperRotation)
        .export_values();

    // 2. Struct SymmetryOperation
    nb::class_<SymmetryOperation>(m, "SymmetryOperation")
        .def_rw("type", &SymmetryOperation::type)
        .def_rw("order", &SymmetryOperation::order)
        .def_rw("matrix", &SymmetryOperation::matrix)
        .def_rw("name", &SymmetryOperation::name);

    // 3. Class PointGroup (HANYA BOLEH ADA SATU DEFINISI INI)
    nb::class_<PointGroup>(m, "PointGroup")
        .def(nb::init<const Molecule&>(), "Initialize and auto-detect symmetry")
        .def("detect", &PointGroup::detect)
        .def("get_symbol", &PointGroup::get_symbol)
        .def("get_order", &PointGroup::get_order)
        .def("get_aligned_molecule", &PointGroup::get_aligned_molecule)
        .def("get_operations", &PointGroup::get_operations, nb::rv_policy::reference_internal);

    // 4. Unique Shell Pair
    nb::class_<UniqueShellPair>(m, "UniqueShellPair")
        .def_rw("p", &UniqueShellPair::p)
        .def_rw("q", &UniqueShellPair::q)
        .def_rw("weight", &UniqueShellPair::weight);

    // 5. Petite List
    nb::class_<PetiteList>(m, "PetiteList")
        .def(nb::init<const BasisSet&, const PointGroup&>())
        .def("build", &PetiteList::build)
        .def("get_unique_pairs", &PetiteList::get_unique_pairs);

    // 6. Basis Symmetrizer
    nb::class_<BasisSymmetrizer>(m, "BasisSymmetrizer")
        .def(nb::init<const BasisSet&, const PointGroup&, const PetiteList&>())
        .def("symmetrize", &BasisSymmetrizer::symmetrize, "Symmetrize Fock matrix in-place");

   

    // ========================================================================
    // 6. INTEGRAL SCREENING & DIIS BINDINGS
    // ========================================================================

    // --- Binding Class DIIS ---
    nb::class_<DIIS>(m, "DIIS")
        .def(nb::init<int>(), nb::arg("max_vectors") = 8)
        .def("clear", &DIIS::clear)
        .def("add_iteration", &DIIS::add_iteration, 
             nb::arg("F"), nb::arg("err"), nb::arg("P"))
        .def("extrapolate", &DIIS::extrapolate);

    // --- Binding Struct ShellPair (Helper untuk Screening) ---
    // Karena ShellPair ada di dalam namespace mshqc::integrals
    nb::class_<mshqc::integrals::ShellPair>(m, "ShellPair")
        .def_rw("sh_a", &mshqc::integrals::ShellPair::sh_a)
        .def_rw("sh_b", &mshqc::integrals::ShellPair::sh_b)
        .def_rw("max_val", &mshqc::integrals::ShellPair::max_val)
        // Kita butuh wrapper lambda untuk Eigen::Vector3d -> list/tuple di Python
        .def_prop_rw("center", 
            [](const mshqc::integrals::ShellPair& sp) { 
                return std::make_tuple(sp.center(0), sp.center(1), sp.center(2)); 
            },
            [](mshqc::integrals::ShellPair& sp, const std::tuple<double,double,double>& t) {
                sp.center = Eigen::Vector3d(std::get<0>(t), std::get<1>(t), std::get<2>(t));
            }
        );

    // --- Binding Class Screening ---
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
    // ========================================================================
    // SCF: Configuration, Results, and Solvers
    // ========================================================================
    
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
        // HAPUS: level_shift (sudah dihapus dari header baru scf.h)

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
   
    // ========================================================================
    // BINDING: FCIDUMP Exporter
    // ========================================================================
    m.def("export_fcidump", &mshqc::export_fcidump,
          nb::arg("filename"), nb::arg("mol"), nb::arg("scf"), 
          nb::arg("integrals"), nb::arg("tol") = 1e-10,
          nb::call_guard<nb::gil_scoped_release>(),
          "Export SCF and Integral results to standard FCIDUMP format");
    // ========================================================================
    // FIX: UHF Binding (Aman dari Double Free)
    // ========================================================================
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

    // ========================================================================
    // FIX: RHF Binding (Aman dari Double Free)
    // ========================================================================
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

    // ========================================================================
    // FIX: ROHF Binding (Aman dari Double Free)
    // ========================================================================
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
    
    // ------------------------------------------------------------------------
    // 1. CONFIG & RESULT STRUCTURES
    // ------------------------------------------------------------------------
    // Ekspos MP2Config ke Python agar user bisa memilih mode "df", "cholesky", dll.
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
        .def_rw("print_level", &MP2Config::print_level);
        

    // Ganti OMP2Result menjadi MP2Result (Universal)
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

    // ------------------------------------------------------------------------
    // 2. SOLVER CLASSES
    // ------------------------------------------------------------------------

    // RMP2
    nb::class_<foundation::RMP2>(m, "RMP2")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &foundation::RMP2::compute, nb::call_guard<nb::gil_scoped_release>());

    // UMP2
    nb::class_<mshqc::UMP2>(m, "UMP2")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &mshqc::UMP2::compute, nb::call_guard<nb::gil_scoped_release>());

    // OMP2
    nb::class_<OMP2>(m, "OMP2")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const SCFResult&, const MP2Config&, std::shared_ptr<PointGroup>, std::shared_ptr<PetiteList>>(), 
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
             nb::arg("config"), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &OMP2::compute, nb::call_guard<nb::gil_scoped_release>(), "Run OMP2 optimization");

    // OMP3 Class (Perhatikan: argumen konstruktornya kini menerima MP2Result)
    /*nb::class_<OMP3>(m, "OMP3")
        .def(nb::init<const Molecule&, const BasisSet&, 
                      std::shared_ptr<IntegralEngine>, const MP2Result&, const SCFConfig&>(),
             nb::arg("mol"), nb::arg("basis"), 
             nb::arg("integrals"), nb::arg("mp2_result"), nb::arg("config"))
        .def("compute", &OMP3::compute, nb::call_guard<nb::gil_scoped_release>(), "Run OMP3 optimization")
        .def("set_max_iterations", &OMP3::set_max_iterations)
        .def("set_convergence_threshold", &OMP3::set_convergence_threshold);  */  
    
    // ========================================================================
    // MP2 & MP3 BINDINGS (COMPLETE & FIXED)
    // ========================================================================

    // ------------------------------------------------------------------------
    // 1. RESULT STRUCTURES
    // -----------------------------------------------------------------------
    // UMP2 Result
    /*nb::class_<UMP2Result>(m, "UMP2Result")
        .def(nb::init<>())
        .def_rw("e_corr_ss_aa", &UMP2Result::e_corr_ss_aa)
        .def_rw("e_corr_ss_bb", &UMP2Result::e_corr_ss_bb)
        .def_rw("e_corr_os", &UMP2Result::e_corr_os)
        .def_rw("e_corr_total", &UMP2Result::e_corr_total)
        .def_rw("e_total", &UMP2Result::e_total);

    // UMP3 Result
    nb::class_<UMP3Result>(m, "UMP3Result")
        .def(nb::init<>())
        .def_rw("e_uhf", &UMP3Result::e_uhf)
        .def_rw("e_mp2", &UMP3Result::e_mp2)
        .def_rw("e_mp3", &UMP3Result::e_mp3)
        .def_rw("e3_aa", &UMP3Result::e3_aa)
        .def_rw("e3_bb", &UMP3Result::e3_bb)
        .def_rw("e3_ab", &UMP3Result::e3_ab)
        .def_rw("e_corr_total", &UMP3Result::e_corr_total)
        .def_rw("e_total", &UMP3Result::e_total);

    // OMP3 Result
    nb::class_<OMP3Result>(m, "OMP3Result")
        .def(nb::init<>())
        .def_rw("energy_total", &OMP3Result::energy_total)
        .def_rw("energy_mp2_corr", &OMP3Result::energy_mp2_corr)
        .def_rw("energy_mp3_corr", &OMP3Result::energy_mp3_corr)
        .def_rw("energy_omp2", &OMP3Result::energy_omp2)
        .def_rw("energy_omp3", &OMP3Result::energy_omp3)
        .def_rw("converged", &OMP3Result::converged)
        .def_rw("iterations", &OMP3Result::iterations)
        .def_rw("orbital_energies_alpha", &OMP3Result::orbital_energies_alpha)
        .def_rw("orbital_energies_beta", &OMP3Result::orbital_energies_beta)
        .def_rw("C_alpha", &OMP3Result::C_alpha)
        .def_rw("C_beta", &OMP3Result::C_beta);

    // RMP2 Result (Foundation)
    nb::class_<foundation::RMP2Result>(m, "RMP2Result")
        .def(nb::init<>())
        .def_rw("e_corr", &foundation::RMP2Result::e_corr)
        .def_rw("e_rhf", &foundation::RMP2Result::e_rhf)
        .def_rw("e_total", &foundation::RMP2Result::e_total);

    // RMP3 Result (Foundation)
    nb::class_<foundation::RMP3Result>(m, "RMP3Result")
        .def(nb::init<>())
        .def_rw("e_mp2", &foundation::RMP3Result::e_mp2)
        .def_rw("e_mp3", &foundation::RMP3Result::e_mp3)
        .def_rw("e_total", &foundation::RMP3Result::e_total);

    // ------------------------------------------------------------------------
    // 2. SOLVER CLASSES (Updated Constructors)
    // ------------------------------------------------------------------------
    // UMP3 Class (With Symmetry Support)
    nb::class_<UMP3>(m, "UMP3")
        .def(nb::init<const SCFResult&, 
                      const UMP2Result&, 
                      const BasisSet&, 
                      std::shared_ptr<IntegralEngine>,
                      std::shared_ptr<PointGroup>>(), // Argumen Baru: PG
             nb::arg("uhf"), 
             nb::arg("ump2"), 
             nb::arg("basis"), 
             nb::arg("integrals"),
             nb::arg("pg") = nullptr)
        .def("compute", &UMP3::compute, nb::call_guard<nb::gil_scoped_release>());

    // RMP3 Class (Standard)
    nb::class_<foundation::RMP3>(m, "RMP3")
        .def(nb::init<const SCFResult&, const foundation::RMP2Result&, const BasisSet&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("rhf_result"), nb::arg("rmp2_result"), nb::arg("basis"), nb::arg("integrals"))
        .def("compute", &foundation::RMP3::compute, nb::call_guard<nb::gil_scoped_release>());*/

    // ========================================================================
    // CI Methods
    // ========================================================================
    
    nb::class_<ci::Determinant>(m, "Determinant")
        .def(nb::init<>())
        .def(nb::init<const std::vector<int>&, const std::vector<int>&>(),
             nb::arg("alpha_occ"), nb::arg("beta_occ"))
        .def("n_alpha", &ci::Determinant::n_alpha)
        .def("n_beta", &ci::Determinant::n_beta)
        .def("excitation_level", &ci::Determinant::excitation_level, nb::arg("other"));

    nb::class_<ci::CIIntegrals>(m, "CIIntegrals")
        .def(nb::init<>())
        .def_rw("e_nuc", &ci::CIIntegrals::e_nuc);

    // FCI Result - check actual structure
    nb::class_<ci::FCIResult>(m, "FCIResult")
        .def(nb::init<>())
        .def_rw("determinants", &ci::FCIResult::determinants)
        .def_rw("converged", &ci::FCIResult::converged);

    


    nb::class_<ci::FCI>(m, "FCI")
        .def(nb::init<const ci::CIIntegrals&, int, int, int, int>(),
             nb::arg("ints"), nb::arg("n_orbitals"),
             nb::arg("n_alpha"), nb::arg("n_beta"), nb::arg("n_roots") = 1)
        .def("compute", &ci::FCI::compute, nb::call_guard<nb::gil_scoped_release>());


    // ========================================================================
    // MCSCF Methods
    // ========================================================================
    
    nb::class_<mcscf::ActiveSpace>(m, "ActiveSpace")
        .def(nb::init<>())
        .def(nb::init<int, int, int, int>(),
             nb::arg("n_inactive"), nb::arg("n_active"), 
             nb::arg("n_virtual"), nb::arg("n_elec_active"))
        .def_static("CAS_Frozen", &mcscf::ActiveSpace::CAS_Frozen,
             nb::arg("n_frozen_orb"), nb::arg("n_active_orb"),
             nb::arg("n_total_orb"), nb::arg("n_total_elec"),
             "Create Active Space defining Frozen Orbitals")
        .def_static("CAS", &mcscf::ActiveSpace::CAS,
                   nb::arg("n_elec"), nb::arg("n_orb"),
                   nb::arg("n_total_orb"), nb::arg("n_total_elec"))
        .def("n_inactive", &mcscf::ActiveSpace::n_inactive)
        .def("n_active", &mcscf::ActiveSpace::n_active)
        .def("n_virtual", &mcscf::ActiveSpace::n_virtual)
        .def("n_elec_active", &mcscf::ActiveSpace::n_elec_active)
        // [TAMBAHAN]
        .def("inactive_indices", &mcscf::ActiveSpace::inactive_indices)
        .def("active_indices", &mcscf::ActiveSpace::active_indices)
        .def("virtual_indices", &mcscf::ActiveSpace::virtual_indices)
        .def("__repr__", &mcscf::ActiveSpace::to_string);
    

   nb::class_<UNOResult>(m, "UNOResult")
        .def(nb::init<>())
        .def_rw("C_uno", &UNOResult::C_uno)
        .def_rw("occupations", &UNOResult::occupations)
        .def_rw("entropy", &UNOResult::entropy)
        // [TAMBAHAN]
        .def_rw("suggested_n_active", &UNOResult::suggested_n_active)
        .def_rw("suggested_n_electrons", &UNOResult::suggested_n_electrons)
        .def_rw("active_indices", &UNOResult::active_indices);

    nb::class_<CholeskyUNO>(m, "CholeskyUNO")
        .def(nb::init<const SCFResult&, std::shared_ptr<IntegralEngine>, int>())
        .def("compute", &CholeskyUNO::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("print_report", &CholeskyUNO::print_report, nb::arg("threshold") = 0.02)
        .def("save_orbitals", &CholeskyUNO::save_orbitals);
    


    nb::class_<CanonicalUNO>(m, "CanonicalUNO")
        .def(nb::init<const SCFResult&, std::shared_ptr<IntegralEngine>, int>(),
             nb::arg("uhf_res"), nb::arg("integrals"), nb::arg("n_basis"))
        .def("compute", &CanonicalUNO::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("print_report", &CanonicalUNO::print_report, nb::arg("threshold") = 0.02)
        .def("save_orbitals", &CanonicalUNO::save_orbitals);
    
    // State-Averaged CASSCF
    nb::class_<SACASConfig>(m, "SACASConfig")
        .def(nb::init<>())
        .def_rw("n_states", &SACASConfig::n_states)
        .def_rw("max_iter", &SACASConfig::max_iter)
        .def_rw("cholesky_thresh", &SACASConfig::cholesky_thresh)
        // TAMBAHAN BARU:
        .def_rw("weights", &SACASConfig::weights)
        .def_rw("e_thresh", &SACASConfig::e_thresh)
        .def_rw("grad_thresh", &SACASConfig::grad_thresh)
        .def_rw("print_level", &SACASConfig::print_level)
        .def_rw("rotation_damping", &SACASConfig::rotation_damping)
        .def_rw("shift", &SACASConfig::shift)
        .def("set_equal_weights", &SACASConfig::set_equal_weights);

    nb::class_<SACASResult>(m, "SACASResult")
        .def(nb::init<>())
        .def_rw("e_avg", &SACASResult::e_avg)
        .def_rw("state_energies", &SACASResult::state_energies)
        .def_rw("C_mo", &SACASResult::C_mo)
        .def_rw("orbital_energies", &SACASResult::orbital_energies) // <-- Penting untuk PT2/PT3
        .def_rw("converged", &SACASResult::converged)
        .def_rw("ci_vectors", &SACASResult::ci_vectors)      // Tambahan akses
        .def_rw("rdm1_states", &SACASResult::rdm1_states);   // Tambahan akses
    nb::class_<CholeskySACASSCF>(m, "CholeskySACASSCF")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&, const SACASConfig&>())
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, const ActiveSpace&, const SACASConfig&, const std::vector<Eigen::VectorXd>&>())
        .def("compute", nb::overload_cast<const SCFResult&>(&CholeskySACASSCF::compute), nb::call_guard<nb::gil_scoped_release>())
        .def("compute", nb::overload_cast<const Eigen::MatrixXd&>(&CholeskySACASSCF::compute), nb::call_guard<nb::gil_scoped_release>());


    nb::class_<CanonicalSACASSCF>(m, "CanonicalSACASSCF")
        .def(nb::init<const Molecule&, const BasisSet&, std::shared_ptr<IntegralEngine>, 
                      const ActiveSpace&, const SACASConfig&>(),
             nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), 
             nb::arg("active_space"), nb::arg("config"))
             
        // Overload 1: Compute dari SCFResult (UHF/RHF)
        .def("compute", nb::overload_cast<const SCFResult&>(&CanonicalSACASSCF::compute), nb::call_guard<nb::gil_scoped_release>(),
             nb::arg("initial_guess"))
             
        // Overload 2: Compute dari Matrix Orbital (misal dari UNO)
        .def("compute", nb::overload_cast<const Eigen::MatrixXd&>(&CanonicalSACASSCF::compute), nb::call_guard<nb::gil_scoped_release>(),
             nb::arg("initial_orbitals"));
    
    nb::class_<CASPT2Config>(m, "CASPT2Config")
        .def(nb::init<>())
        .def_rw("shift", &CASPT2Config::shift)
        .def_rw("print_level", &CASPT2Config::print_level)
        // [TAMBAHAN]
        .def_rw("zero_thresh", &CASPT2Config::zero_thresh)
        .def_rw("use_tblis", &CASPT2Config::use_tblis)
        .def_rw("export_amplitudes", &CASPT2Config::export_amplitudes);

    nb::class_<CASPT2Result>(m, "CASPT2Result")
        .def(nb::init<>())
        .def_rw("e_cas", &CASPT2Result::e_cas)
        .def_rw("e_pt2", &CASPT2Result::e_pt2)
        .def_rw("e_total", &CASPT2Result::e_total)
        // [TAMBAHAN]
        .def_rw("amplitudes", &CASPT2Result::amplitudes);

    // Register Dummy/Opaque MOIntegrals if needed, or just hide it
    // nb::class_<MOIntegrals, std::shared_ptr<MOIntegrals>>(m, "MOIntegrals"); 

    nb::class_<CholeskySACASPT2>(m, "CholeskySACASPT2")
        .def(nb::init<const SACASResult&, const std::vector<Eigen::VectorXd>&, int, const ActiveSpace&, const CASPT2Config&>())
        // FIX: Explicit overload for no arguments
        .def("compute", [](CholeskySACASPT2& self) {
            nb::gil_scoped_release release;
            return self.compute(nullptr); 
        }, "Compute CASPT2 without precomputed MO integrals")
        // Optional: Keep the original if you ever bind MOIntegrals
        // .def("compute", &CholeskySACASPT2::compute, nb::arg("mo_ints")) 
        ;
    // --- Cholesky Single-State CASPT2 Bindings ---


    

    
    nb::class_<CanonicalSACASPT2>(m, "CanonicalSACASPT2")
        .def(nb::init<const SACASResult&, 
                      std::shared_ptr<IntegralEngine>,
                      const BasisSet&,
                      const ActiveSpace&, 
                      const CASPT2Config&>(),
             nb::arg("sacas_result"), 
             nb::arg("integrals"), 
             nb::arg("basis"),
             nb::arg("active_space"), 
             nb::arg("config"))
        .def("compute", &CanonicalSACASPT2::compute, nb::call_guard<nb::gil_scoped_release>());
    
   // [UPDATE INI DI bindings.cc]
    

    // ========================================================================
    // Gradient and Optimization
    // ========================================================================
    
    nb::class_<gradient::GradientResult>(m, "GradientResult")
        .def(nb::init<>())
        .def_rw("energy", &gradient::GradientResult::energy)
        .def_rw("gradient", &gradient::GradientResult::gradient);

    // AnalyticalGradient is abstract - don't expose constructor
    // Just document that it's used internally

    nb::class_<gradient::OptConfig>(m, "OptConfig")
        .def(nb::init<>())
        .def_rw("max_iterations", &gradient::OptConfig::max_iterations);

    nb::class_<gradient::OptResult>(m, "OptResult")
        .def(nb::init<>())
        .def_rw("converged", &gradient::OptResult::converged)
        .def_rw("n_iterations", &gradient::OptResult::n_iterations)
        .def_rw("final_energy", &gradient::OptResult::final_energy);

    nb::class_<PT2Amplitudes>(m, "PT2Amplitudes")
        .def(nb::init<>())
        .def_rw("t2_core", &PT2Amplitudes::t2_core)
        .def_rw("t2_active", &PT2Amplitudes::t2_active)
        .def_rw("t2_semi1", &PT2Amplitudes::t2_semi1)
        .def_rw("t2_semi2", &PT2Amplitudes::t2_semi2);



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
