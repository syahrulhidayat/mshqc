// Copyright 2026 Muhamad Syahrul Hidayat
// Licensed under the Apache License, Version 2.0
// [INJECTED_THIN_CORE_STAGE5_LEGACY_BINDINGS]
// Optional legacy C++ solver bindings — compiled ONLY when MSHQC_THIN_CORE=OFF.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>

#include "mshqc/basis.h"
#include "mshqc/core/molecule.h"
#include "mshqc/gradient/hpc_geom_opt.h"
#include "mshqc/ints/integrals.h"
#include "mshqc/mp2/mp2.h"
#include "mshqc/mp3/mp3.h"
#include "mshqc/mp3/omp3.h"
#include "mshqc/scf/diis.h"
#include "mshqc/scf/scf.h"
#include "mshqc/symmetry/molecule_sym.h"
#include "mshqc/symmetry/petite_list.h"
#include "mshqc/symmetry/point_group.h"

namespace nb = nanobind;

namespace mshqc {

namespace {

SCFConfig normalize_scf_config(const SCFConfig& conf, const BasisSet& basis) {
    SCFConfig c = conf;
    std::string m = c.eri_method;
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return std::tolower(ch); });
    c.eri_method = m;
    if (c.eri_method == "df") c.use_df = true;
    if (c.use_df && (c.aux_basis_name.empty() || c.aux_basis_name == "cc-pVDZ-JKFIT")) {
        c.aux_basis_name = basis.name().empty() ? "cc-pVTZ-RI" : (basis.name() + "-RI");
    }
    return c;
}

MP2Config normalize_mp2_config(const MP2Config& conf, const BasisSet& basis, const SCFResult&) {
    MP2Config c = conf;
    std::string m = c.eri_method;
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return std::tolower(ch); });
    c.eri_method = m;
    if (c.eri_method == "df") c.use_df = true;
    if (c.use_df && c.aux_basis_name.empty()) {
        c.aux_basis_name = basis.name().empty() ? "cc-pVTZ-RI" : (basis.name() + "-RI");
    }
    if (c.gradient_threshold <= 0.0) c.gradient_threshold = 1e-5;
    return c;
}

} // namespace

void bind_legacy_module(nb::module_& m, nb::class_<Molecule>& mol_cls) {
    mol_cls
        .def("optimize_geometry", [](Molecule& mol,
                                     const std::string& method,
                                     const std::string& eri_mode,
                                     const std::string& eri_method,
                                     const std::string& basis,
                                     const std::string& aux_basis,
                                     const std::string& reference,
                                     int max_iter,
                                     double g_rms_tol,
                                     double g_max_tol,
                                     double dx_rms_tol,
                                     double dx_max_tol,
                                     double energy_thresh,
                                     double orb_grad_thresh,
                                     double fd_step_bohr,
                                     bool use_analytic_gradient,
                                     bool verbose) {
            mshqc::gradient::GeomOptConfig cfg;
            cfg.method = method;
            cfg.eri_method = !eri_method.empty() ? eri_method : eri_mode;
            cfg.basis_name = basis;
            cfg.aux_basis_name = aux_basis;
            cfg.reference = reference;
            cfg.max_iter = max_iter;
            cfg.g_rms_tol = g_rms_tol;
            cfg.g_max_tol = g_max_tol;
            cfg.dx_rms_tol = dx_rms_tol;
            cfg.dx_max_tol = dx_max_tol;
            cfg.energy_thresh = energy_thresh;
            cfg.orb_grad_thresh = orb_grad_thresh;
            cfg.fd_step_bohr = fd_step_bohr;
            cfg.use_analytic_gradient = use_analytic_gradient;
            cfg.verbose = verbose;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::optimize_molecule(mol, cfg);
        }, nb::arg("method") = "omp2",
           nb::arg("eri_mode") = "df",
           nb::arg("eri_method") = "",
           nb::arg("basis") = "cc-pVDZ",
           nb::arg("aux_basis") = "",
           nb::arg("reference") = "auto",
           nb::arg("max_iter") = 40,
           nb::arg("g_rms_tol") = 1.0e-8,
           nb::arg("g_max_tol") = 3.0e-8,
           nb::arg("dx_rms_tol") = 1.0e-6,
           nb::arg("dx_max_tol") = 3.0e-6,
           nb::arg("energy_thresh") = 1.0e-12,
           nb::arg("orb_grad_thresh") = 1.0e-10,
           nb::arg("fd_step_bohr") = 5.0e-4,
           nb::arg("use_analytic_gradient") = true,
           nb::arg("verbose") = true)
        .def("frequencies", [](const Molecule& mol,
                               const std::string& method,
                               const std::string& eri_mode,
                               const std::string& eri_method,
                               const std::string& basis,
                               const std::string& reference,
                               double temperature_k,
                               bool verbose) {
            mshqc::gradient::GeomOptConfig cfg;
            cfg.method = method;
            cfg.eri_method = !eri_method.empty() ? eri_method : eri_mode;
            cfg.basis_name = basis;
            cfg.reference = reference;
            cfg.verbose = verbose;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::compute_frequencies(mol, cfg, temperature_k, 1.0);
        }, nb::arg("method") = "omp2",
           nb::arg("eri_mode") = "df",
           nb::arg("eri_method") = "",
           nb::arg("basis") = "cc-pVDZ",
           nb::arg("reference") = "auto",
           nb::arg("temperature_k") = 298.15,
           nb::arg("verbose") = true)
        .def("gradient", [](const Molecule& mol,
                            const std::string& method,
                            const std::string& eri_mode,
                            const std::string& eri_method,
                            const std::string& basis,
                            const std::string& aux_basis,
                            const std::string& reference,
                            double fd_step_bohr,
                            bool use_analytic_gradient) {
            mshqc::gradient::GeomOptConfig cfg;
            cfg.method = method;
            cfg.eri_method = !eri_method.empty() ? eri_method : eri_mode;
            cfg.basis_name = basis;
            cfg.aux_basis_name = aux_basis;
            cfg.reference = reference;
            cfg.fd_step_bohr = fd_step_bohr;
            cfg.use_analytic_gradient = use_analytic_gradient;
            cfg.verbose = false;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::compute_nuclear_gradient_cpp(mol, cfg);
        }, nb::arg("method") = "omp2",
           nb::arg("eri_mode") = "df",
           nb::arg("eri_method") = "",
           nb::arg("basis") = "cc-pVDZ",
           nb::arg("aux_basis") = "",
           nb::arg("reference") = "auto",
           nb::arg("fd_step_bohr") = 5.0e-4,
           nb::arg("use_analytic_gradient") = true)
        .def("scan_pes", [](const Molecule& mol,
                            int atom_i,
                            int atom_j,
                            const std::vector<double>& distances_angstrom,
                            const std::string& method,
                            const std::string& eri_mode,
                            const std::string& basis) {
            mshqc::gradient::GeomOptConfig cfg;
            cfg.method = method;
            cfg.eri_method = eri_mode;
            cfg.basis_name = basis;
            cfg.verbose = false;
            nb::gil_scoped_release release;
            return mshqc::gradient::HPCGeometryOptimizer::scan_bond_pes(
                mol, atom_i, atom_j, distances_angstrom, cfg);
        }, nb::arg("atom_i"), nb::arg("atom_j"), nb::arg("distances_angstrom"),
           nb::arg("method") = "omp2", nb::arg("eri_mode") = "df", nb::arg("basis") = "cc-pVDZ");

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
        .def(nb::init<const Molecule&, double>(), nb::arg("mol"), nb::arg("tolerance") = 1e-6)
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
        .def("build", nb::overload_cast<>(&PetiteList::build))
        .def("build", nb::overload_cast<const Eigen::MatrixXd&, double>(&PetiteList::build),
             nb::arg("schwarz"), nb::arg("cutoff") = 1e-12)
        .def("get_unique_pairs", &PetiteList::get_unique_pairs);

    nb::class_<BasisSymmetrizer>(m, "BasisSymmetrizer")
        .def(nb::init<const BasisSet&, const PointGroup&, const PetiteList&>())
        .def("symmetrize", &BasisSymmetrizer::symmetrize);

    nb::class_<DIIS>(m, "DIIS")
        .def(nb::init<int>(), nb::arg("max_vectors") = 8)
        .def("clear", &DIIS::clear)
        .def("add_iteration", &DIIS::add_iteration, nb::arg("F"), nb::arg("err"))
        .def("extrapolate", &DIIS::extrapolate);

    nb::class_<SCFConfig>(m, "SCFConfig")
        .def("__init__", [](SCFConfig* c, const std::string& eri_method, double threshold,
                            const std::string& aux_basis_name, const std::string& scf_type,
                            int print_level, int max_iterations) {
            new (c) SCFConfig();
            c->eri_method = eri_method;
            c->scf_type = scf_type;
            c->energy_threshold = threshold;
            c->density_threshold = threshold;
            c->cholesky_threshold = threshold;
            c->df_threshold = threshold;
            c->print_level = print_level;
            c->max_iterations = max_iterations;
            c->aux_basis_name = aux_basis_name;
            c->use_df = (eri_method == "df" || eri_method == "DF");
        }, nb::arg("eri_method") = "exact", nb::arg("threshold") = 1e-9,
           nb::arg("aux_basis_name") = "", nb::arg("scf_type") = "incore",
           nb::arg("print_level") = 0, nb::arg("max_iterations") = 100)
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
        .def_rw("aux_basis_name", &SCFConfig::aux_basis_name)
        .def_rw("memory_limit_gb", &SCFConfig::memory_limit_gb)
        .def_rw("level_shift", &SCFConfig::level_shift);

    nb::class_<SCFResult>(m, "SCFResult")
        .def(nb::init<>())
        .def_rw("energy_electronic", &SCFResult::energy_electronic)
        .def_rw("energy_nuclear", &SCFResult::energy_nuclear)
        .def_rw("energy_total", &SCFResult::energy_total)
        .def_prop_ro("e_total", [](const SCFResult& r) { return r.energy_total; })
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
        .def_rw("n_occ_beta", &SCFResult::n_occ_beta)
        .def_ro("L_mat", &SCFResult::L_mat, nb::rv_policy::reference_internal);

    nb::class_<UHF>(m, "UHF")
        .def("__init__", [](UHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, PointGroup* pg, PetiteList* pl,
                            int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*) {}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*) {}) : nullptr;
            new (t) UHF(mol, basis, sp_int, sp_pg, sp_pl, na, nb, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("pg"), nb::arg("pl"), nb::arg("n_alpha"), nb::arg("n_beta"),
           nb::arg("config") = SCFConfig())
        .def("__init__", [](UHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            new (t) UHF(mol, basis, sp_int, nullptr, nullptr, na, nb, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("n_alpha"), nb::arg("n_beta"), nb::arg("config") = SCFConfig())
        .def("__init__", [](UHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, const SCFConfig& conf,
                            PointGroup* pg, PetiteList* pl) {
            int n_elec = mol.n_electrons(), mult = mol.multiplicity();
            int na = (n_elec + mult - 1) / 2, nb = n_elec - na;
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*) {}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*) {}) : nullptr;
            new (t) UHF(mol, basis, sp_int, sp_pg, sp_pl, na, nb, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("config") = SCFConfig(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &UHF::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("energy", &UHF::energy);

    nb::class_<RHF>(m, "RHF")
        .def("__init__", [](RHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, PointGroup* pg, PetiteList* pl,
                            const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*) {}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*) {}) : nullptr;
            new (t) RHF(mol, basis, sp_int, sp_pg, sp_pl, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("pg"), nb::arg("pl"), nb::arg("config") = SCFConfig())
        .def("__init__", [](RHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, const SCFConfig& conf,
                            PointGroup* pg, PetiteList* pl) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*) {}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*) {}) : nullptr;
            new (t) RHF(mol, basis, sp_int, sp_pg, sp_pl, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("config") = SCFConfig(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &RHF::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("energy", &RHF::energy);

    nb::class_<ROHF>(m, "ROHF")
        .def("__init__", [](ROHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, PointGroup* pg, PetiteList* pl,
                            int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*) {}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*) {}) : nullptr;
            new (t) ROHF(mol, basis, sp_int, sp_pg, sp_pl, na, nb, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("pg"), nb::arg("pl"), nb::arg("n_alpha"), nb::arg("n_beta"),
           nb::arg("config") = SCFConfig())
        .def("__init__", [](ROHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, int na, int nb, const SCFConfig& conf) {
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            new (t) ROHF(mol, basis, sp_int, nullptr, nullptr, na, nb, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("n_alpha"), nb::arg("n_beta"), nb::arg("config") = SCFConfig())
        .def("__init__", [](ROHF* t, const Molecule& mol, const BasisSet& basis,
                            IntegralEngine* integrals, const SCFConfig& conf,
                            PointGroup* pg, PetiteList* pl) {
            int n_elec = mol.n_electrons(), mult = mol.multiplicity();
            int na = (n_elec + mult - 1) / 2, nb = n_elec - na;
            auto sp_int = std::shared_ptr<IntegralEngine>(integrals, [](IntegralEngine*) {});
            auto sp_pg = pg ? std::shared_ptr<PointGroup>(pg, [](PointGroup*) {}) : nullptr;
            auto sp_pl = pl ? std::shared_ptr<PetiteList>(pl, [](PetiteList*) {}) : nullptr;
            new (t) ROHF(mol, basis, sp_int, sp_pg, sp_pl, na, nb, normalize_scf_config(conf, basis));
        }, nb::arg("molecule"), nb::arg("basis"), nb::arg("integrals"),
           nb::arg("config") = SCFConfig(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &ROHF::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("energy", &ROHF::energy);

    nb::class_<MP2Config>(m, "MP2Config")
        .def("__init__", [](MP2Config* c, const std::string& eri_method, double threshold,
                            double gradient_threshold, const std::string& opt_method,
                            const std::string& aux_basis_name, const std::string& scf_type,
                            int print_level, int max_iterations) {
            new (c) MP2Config();
            c->eri_method = eri_method;
            c->scf_type = scf_type;
            c->energy_threshold = threshold;
            c->cholesky_threshold = threshold;
            c->df_threshold = threshold;
            c->gradient_threshold = gradient_threshold;
            c->opt_method = opt_method;
            c->aux_basis_name = aux_basis_name;
            c->print_level = print_level;
            c->max_iterations = max_iterations;
            c->use_df = (eri_method == "df" || eri_method == "DF");
        }, nb::arg("eri_method") = "exact", nb::arg("threshold") = 1e-9,
           nb::arg("gradient_threshold") = 1e-5, nb::arg("opt_method") = "soscf",
           nb::arg("aux_basis_name") = "", nb::arg("scf_type") = "incore",
           nb::arg("print_level") = 0, nb::arg("max_iterations") = 50)
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
        .def_rw("opt_method", &MP2Config::opt_method)
        .def_rw("n_frozen", &MP2Config::n_frozen)
        .def_rw("freeze_core", &MP2Config::freeze_core)
        .def_rw("memory_limit_gb", &MP2Config::memory_limit_gb);

    nb::class_<MP2Result>(m, "MP2Result")
        .def(nb::init<>())
        .def_rw("energy_scf", &MP2Result::energy_scf)
        .def_rw("energy_mp2_ss", &MP2Result::energy_mp2_ss)
        .def_rw("energy_mp2_os", &MP2Result::energy_mp2_os)
        .def_rw("energy_mp2_corr", &MP2Result::energy_mp2_corr)
        .def_rw("energy_total", &MP2Result::energy_total)
        .def_prop_ro("e_total", [](const MP2Result& r) { return r.energy_total; })
        .def_prop_ro("e_corr_total", [](const MP2Result& r) { return r.energy_mp2_corr; })
        .def_rw("converged", &MP2Result::converged)
        .def_rw("iterations", &MP2Result::iterations)
        .def_rw("n_occ_alpha", &MP2Result::n_occ_alpha)
        .def_rw("n_occ_beta", &MP2Result::n_occ_beta)
        .def_rw("n_virt_alpha", &MP2Result::n_virt_alpha)
        .def_rw("n_virt_beta", &MP2Result::n_virt_beta)
        .def_rw("C_alpha", &MP2Result::C_alpha)
        .def_rw("C_beta", &MP2Result::C_beta)
        .def_rw("orbital_energies_alpha", &MP2Result::orbital_energies_alpha)
        .def_rw("orbital_energies_beta", &MP2Result::orbital_energies_beta)
        .def_rw("energy_mp2_singles", &MP2Result::energy_mp2_singles);

    nb::class_<foundation::RMP2>(m, "RMP2")
        .def("__init__", [](foundation::RMP2* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            new (t) foundation::RMP2(mol, basis, integrals, scf_guess,
                                     normalize_mp2_config(config, basis, scf_guess), pg, pl);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &foundation::RMP2::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP2::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<UMP2>(m, "UMP2")
        .def("__init__", [](UMP2* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            new (t) UMP2(mol, basis, integrals, scf_guess,
                         normalize_mp2_config(config, basis, scf_guess), pg, pl);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &UMP2::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP2::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<ROMP2, UMP2>(m, "ROMP2")
        .def("__init__", [](ROMP2* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            new (t) ROMP2(mol, basis, integrals, scf_guess,
                          normalize_mp2_config(config, basis, scf_guess), pg, pl);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &ROMP2::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP2::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<OMP2>(m, "OMP2")
        .def("__init__", [](OMP2* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            new (t) OMP2(mol, basis, integrals, scf_guess,
                         normalize_mp2_config(config, basis, scf_guess), pg, pl);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &OMP2::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP2::set_frozen_core, nb::arg("n_frozen"));

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
        .def_prop_ro("energy_total", [](const MP3Result& r) { return r.e_total; })
        .def_prop_ro("energy_scf", [](const MP3Result& r) { return r.e_hf; })
        .def_prop_ro("energy_mp2_corr", [](const MP3Result& r) { return r.e_mp2; })
        .def_prop_ro("energy_mp3_corr", [](const MP3Result& r) { return r.e_mp3; })
        .def_rw("converged", &MP3Result::converged)
        .def_rw("iterations", &MP3Result::iterations)
        .def_rw("C_alpha", &MP3Result::C_alpha)
        .def_rw("C_beta", &MP3Result::C_beta)
        .def_rw("n_occ_alpha", &MP3Result::n_occ_alpha)
        .def_rw("n_occ_beta", &MP3Result::n_occ_beta);

    nb::class_<RMP3>(m, "RMP3")
        .def(nb::init<const SCFResult&, const MP2Result&, const MP2Config&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("scf_guess"), nb::arg("mp2_guess"), nb::arg("config"), nb::arg("integrals"))
        .def("__init__", [](RMP3* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            auto norm_conf = normalize_mp2_config(config, basis, scf_guess);
            if (norm_conf.freeze_core && norm_conf.n_frozen == 0) {
                norm_conf.n_frozen = compute_standard_frozen_core(mol);
            }
            foundation::RMP2 rmp2_solver(mol, basis, integrals, scf_guess, norm_conf, pg, pl);
            MP2Result mp2_res = rmp2_solver.compute();
            new (t) RMP3(scf_guess, mp2_res, norm_conf, integrals);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &RMP3::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP3::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<UMP3>(m, "UMP3")
        .def(nb::init<const SCFResult&, const MP2Result&, const MP2Config&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("scf_guess"), nb::arg("mp2_guess"), nb::arg("config"), nb::arg("integrals"))
        .def("__init__", [](UMP3* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            auto norm_conf = normalize_mp2_config(config, basis, scf_guess);
            if (norm_conf.freeze_core && norm_conf.n_frozen == 0) {
                norm_conf.n_frozen = compute_standard_frozen_core(mol);
            }
            UMP2 ump2_solver(mol, basis, integrals, scf_guess, norm_conf, pg, pl);
            MP2Result mp2_res = ump2_solver.compute();
            new (t) UMP3(scf_guess, mp2_res, norm_conf, integrals);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &UMP3::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP3::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<ROMP3, UMP3>(m, "ROMP3")
        .def(nb::init<const SCFResult&, const MP2Result&, const MP2Config&, std::shared_ptr<IntegralEngine>>(),
             nb::arg("scf_guess"), nb::arg("mp2_guess"), nb::arg("config"), nb::arg("integrals"))
        .def("__init__", [](ROMP3* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            auto norm_conf = normalize_mp2_config(config, basis, scf_guess);
            if (norm_conf.freeze_core && norm_conf.n_frozen == 0) {
                norm_conf.n_frozen = compute_standard_frozen_core(mol);
            }
            ROMP2 romp2_solver(mol, basis, integrals, scf_guess, norm_conf, pg, pl);
            MP2Result mp2_res = romp2_solver.compute();
            new (t) ROMP3(scf_guess, mp2_res, norm_conf, integrals);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &ROMP3::compute, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP3::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<OMP3>(m, "OMP3")
        .def("__init__", [](OMP3* t, const Molecule& mol, const BasisSet& basis,
                            std::shared_ptr<IntegralEngine> integrals, const SCFResult& scf_guess,
                            const MP2Config& config, std::shared_ptr<PointGroup> pg, std::shared_ptr<PetiteList> pl) {
            new (t) OMP3(mol, basis, integrals, scf_guess,
                         normalize_mp2_config(config, basis, scf_guess), pg, pl);
        }, nb::arg("mol"), nb::arg("basis"), nb::arg("integrals"), nb::arg("scf_guess"),
           nb::arg("config") = MP2Config(), nb::arg("pg") = nullptr, nb::arg("pl") = nullptr)
        .def("compute", &OMP3::compute_omp3, nb::call_guard<nb::gil_scoped_release>())
        .def("set_frozen_core", &BaseMP2::set_frozen_core, nb::arg("n_frozen"));

    nb::class_<gradient::GeomOptConfig>(m, "GeomOptConfig")
        .def(nb::init<>())
        .def_rw("method", &gradient::GeomOptConfig::method)
        .def_rw("eri_method", &gradient::GeomOptConfig::eri_method)
        .def_rw("basis_name", &gradient::GeomOptConfig::basis_name)
        .def_rw("aux_basis_name", &gradient::GeomOptConfig::aux_basis_name)
        .def_rw("reference", &gradient::GeomOptConfig::reference)
        .def_rw("opt_method", &gradient::GeomOptConfig::opt_method)
        .def_rw("max_iter", &gradient::GeomOptConfig::max_iter)
        .def_rw("energy_thresh", &gradient::GeomOptConfig::energy_thresh)
        .def_rw("orb_grad_thresh", &gradient::GeomOptConfig::orb_grad_thresh)
        .def_rw("g_rms_tol", &gradient::GeomOptConfig::g_rms_tol)
        .def_rw("g_max_tol", &gradient::GeomOptConfig::g_max_tol)
        .def_rw("dx_rms_tol", &gradient::GeomOptConfig::dx_rms_tol)
        .def_rw("dx_max_tol", &gradient::GeomOptConfig::dx_max_tol)
        .def_rw("trust_radius_init", &gradient::GeomOptConfig::trust_radius_init)
        .def_rw("trust_radius_max", &gradient::GeomOptConfig::trust_radius_max)
        .def_rw("trust_radius_min", &gradient::GeomOptConfig::trust_radius_min)
        .def_rw("fd_step_bohr", &gradient::GeomOptConfig::fd_step_bohr)
        .def_rw("use_analytic_gradient", &gradient::GeomOptConfig::use_analytic_gradient)
        .def_rw("project_trans_rot", &gradient::GeomOptConfig::project_trans_rot)
        .def_rw("verbose", &gradient::GeomOptConfig::verbose)
        .def_rw("xyz_traj_file", &gradient::GeomOptConfig::xyz_traj_file)
        .def_rw("freeze_core", &gradient::GeomOptConfig::freeze_core)
        .def_rw("n_frozen", &gradient::GeomOptConfig::n_frozen)
        .def_rw("memory_limit_gb", &gradient::GeomOptConfig::memory_limit_gb);

    nb::class_<gradient::GeomOptResult>(m, "GeomOptResult")
        .def_ro("converged", &gradient::GeomOptResult::converged)
        .def_ro("iterations", &gradient::GeomOptResult::iterations)
        .def_ro("method", &gradient::GeomOptResult::method)
        .def_ro("eri_method", &gradient::GeomOptResult::eri_method)
        .def_ro("basis_name", &gradient::GeomOptResult::basis_name)
        .def_ro("initial_energy", &gradient::GeomOptResult::initial_energy)
        .def_ro("final_energy", &gradient::GeomOptResult::final_energy)
        .def_ro("final_g_rms", &gradient::GeomOptResult::final_g_rms)
        .def_ro("final_g_max", &gradient::GeomOptResult::final_g_max)
        .def_ro("wall_time_sec", &gradient::GeomOptResult::wall_time_sec)
        .def_ro("coordinates_bohr", &gradient::GeomOptResult::coordinates_bohr)
        .def_ro("coordinates_angstrom", &gradient::GeomOptResult::coordinates_angstrom)
        .def_ro("gradient_ha_bohr", &gradient::GeomOptResult::gradient_ha_bohr)
        .def_ro("energy_history", &gradient::GeomOptResult::energy_history)
        .def_ro("g_rms_history", &gradient::GeomOptResult::g_rms_history)
        .def_ro("bond_lengths_angstrom", &gradient::GeomOptResult::bond_lengths_angstrom);

    nb::class_<gradient::VibThermoResult>(m, "VibThermoResult")
        .def_ro("frequencies_cm1", &gradient::VibThermoResult::frequencies_cm1)
        .def_ro("normal_modes", &gradient::VibThermoResult::normal_modes)
        .def_ro("hessian_ha_bohr2", &gradient::VibThermoResult::hessian_ha_bohr2)
        .def_ro("electronic_energy_ha", &gradient::VibThermoResult::electronic_energy_ha)
        .def_ro("zpe_hartree", &gradient::VibThermoResult::zpe_hartree)
        .def_ro("zpe_kcal_mol", &gradient::VibThermoResult::zpe_kcal_mol)
        .def_ro("thermal_energy_ha", &gradient::VibThermoResult::thermal_energy_ha)
        .def_ro("enthalpy_ha", &gradient::VibThermoResult::enthalpy_ha)
        .def_ro("gibbs_free_energy_ha", &gradient::VibThermoResult::gibbs_free_energy_ha)
        .def_ro("temperature_k", &gradient::VibThermoResult::temperature_k)
        .def_ro("n_imaginary", &gradient::VibThermoResult::n_imaginary);

    nb::class_<gradient::PESScanPoint>(m, "PESScanPoint")
        .def_ro("r_angstrom", &gradient::PESScanPoint::r_angstrom)
        .def_ro("r_bohr", &gradient::PESScanPoint::r_bohr)
        .def_ro("energy_ha", &gradient::PESScanPoint::energy_ha);

    nb::class_<gradient::HPCGeometryOptimizer>(m, "HPCGeometryOptimizer")
        .def_static("optimize_molecule", &gradient::HPCGeometryOptimizer::optimize_molecule,
                    nb::arg("mol"), nb::arg("config") = gradient::GeomOptConfig(),
                    nb::call_guard<nb::gil_scoped_release>())
        .def_static("compute_nuclear_gradient_cpp", &gradient::HPCGeometryOptimizer::compute_nuclear_gradient_cpp,
                    nb::arg("mol"), nb::arg("config") = gradient::GeomOptConfig(),
                    nb::call_guard<nb::gil_scoped_release>())
        .def_static("compute_frequencies", &gradient::HPCGeometryOptimizer::compute_frequencies,
                    nb::arg("mol"), nb::arg("config") = gradient::GeomOptConfig(),
                    nb::arg("temperature_k") = 298.15, nb::arg("pressure_atm") = 1.0,
                    nb::call_guard<nb::gil_scoped_release>())
        .def_static("scan_bond_pes", &gradient::HPCGeometryOptimizer::scan_bond_pes,
                    nb::arg("mol"), nb::arg("atom_i"), nb::arg("atom_j"),
                    nb::arg("distances_angstrom"), nb::arg("config") = gradient::GeomOptConfig(),
                    nb::call_guard<nb::gil_scoped_release>());
}

} // namespace mshqc
