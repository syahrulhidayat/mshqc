 // ==============================================================================
 // Copyright (c) 2026 Syahrul and mshqc contributors
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

#ifndef MSHQC_OPTIMIZER_H
#define MSHQC_OPTIMIZER_H

#include "mshqc/gradient/gradient.h"
#include "mshqc/core/molecule.h"
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <string>
#include <memory>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace gradient {

enum class OptAlgorithm {
    STEEPEST_DESCENT,
    CONJUGATE_GRADIENT,
    BFGS
};

struct OptConfig {
    OptAlgorithm algorithm = OptAlgorithm::BFGS;

    double max_force_thresh = 4.5e-4;
    double rms_force_thresh = 3.0e-4;
    double max_step_thresh  = 1.8e-3;
    double rms_step_thresh  = 1.2e-3;
    double energy_thresh    = 1.0e-6;

    int max_iterations = 100;

    bool use_line_search = true;
    double alpha_init = 1.0;
    double alpha_max = 2.0;
    double rho = 0.5;
    double c1 = 1e-4;
    int max_line_search = 20;

    double trust_radius = 0.3;
    double trust_radius_max = 0.5;
    double trust_radius_min = 0.01;

    int print_level = 1;
    bool print_geometry = true;
};

struct OptResult {
    bool converged = false;
    int n_iterations = 0;

    Molecule final_geometry;
    double final_energy = 0.0;
    Eigen::VectorXd final_gradient;

    double max_force = 0.0;
    double rms_force = 0.0;
    double max_step = 0.0;
    double rms_step = 0.0;
    double energy_change = 0.0;

    std::vector<double> energy_history;
    std::vector<double> max_force_history;
    std::vector<double> rms_force_history;

    std::string algorithm;
    std::string termination_reason;
};

class GeometryOptimizer {
public:

    GeometryOptimizer(
        std::function<GradientResult(const Molecule&)> gradient_func,
        const OptConfig& config = OptConfig()
    );

    OptResult optimize(const Molecule& initial_geom);

    const OptConfig& config() const { return config_; }

    void set_config(const OptConfig& config) { config_ = config; }

private:

    OptConfig config_;
    std::function<GradientResult(const Molecule&)> gradient_func_;

    Molecule current_geom_;
    Eigen::VectorXd current_coords_;
    Eigen::VectorXd current_gradient_;
    double current_energy_;

    Eigen::VectorXd search_direction_;
    Eigen::MatrixXd hessian_inverse_;
    Eigen::VectorXd prev_gradient_;
    Eigen::VectorXd prev_coords_;

    std::vector<double> energy_history_;
    std::vector<double> max_force_history_;
    std::vector<double> rms_force_history_;

    bool step();

    bool check_convergence(const Eigen::VectorXd& step_vector);

    Molecule coords_to_molecule(const Eigen::VectorXd& coords) const;

    Eigen::VectorXd molecule_to_coords(const Molecule& mol) const;

    void steepest_descent_step();

    void conjugate_gradient_step();

    void bfgs_step();

    void initialize_bfgs_hessian();

    void update_bfgs_hessian(const Eigen::VectorXd& s, const Eigen::VectorXd& y);

    double line_search(const Eigen::VectorXd& direction);

    Eigen::VectorXd apply_trust_region(const Eigen::VectorXd& step);

    void print_header();

    void print_iteration(int iter, const Eigen::VectorXd& step_vector);

    void print_results(const OptResult& result);

    void compute_statistics(const Eigen::VectorXd& vec, double& rms, double& max_val);
};

struct TrustRegionConfig {
    int max_micro_iter = 30;
    double micro_thresh = 1e-4;
    int print_level = 0;
    double precond_shift = 1e-4;
};

struct TrustRegionResult {
    Eigen::VectorXd step;
    double predicted_energy_change;
    bool hit_boundary;
};

class TrustRegionSOSCF {
public:
    TrustRegionSOSCF(const TrustRegionConfig& config = TrustRegionConfig());

    TrustRegionResult solve(
        const Eigen::VectorXd& gradient,
        const Eigen::VectorXd& diag_hessian,
        double trust_radius,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> compute_hessian_vector
    );

private:
    TrustRegionConfig config_;

    double compute_boundary_intersection(const Eigen::VectorXd& z, const Eigen::VectorXd& p, double R);

    double compute_model_energy(const Eigen::VectorXd& g, const Eigen::VectorXd& step,
                                std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& compute_H_vec);
};

OptResult optimize_rhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    const OptConfig& config = OptConfig()
);

OptResult optimize_uhf(
    const Molecule& initial_geom,
    const std::string& basis_name,
    int charge,
    int multiplicity,
    const OptConfig& config = OptConfig()
);

}
}

#endif
