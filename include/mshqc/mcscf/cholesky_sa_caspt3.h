#ifndef MSHQC_MCSCF_CHOLESKY_SA_CASPT3_H
#define MSHQC_MCSCF_CHOLESKY_SA_CASPT3_H

#include "mshqc/mcscf/cholesky_sa_casscf.h"
#include "mshqc/mcscf/cholesky_sa_caspt2.h"
#include <vector>
#include <Eigen/Dense>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace mcscf {

struct CASPT3Config {
    double shift = 0.0;         
    double zero_thresh = 1e-12; 
    int print_level = 1;        
};

struct CASPT3Result {
    std::vector<double> e_cas;   
    std::vector<double> e_pt2;   
    std::vector<double> e_pt3;   
    std::vector<double> e_total; 
};

class CholeskySACASPT3 {
public:
    CholeskySACASPT3(const SACASResult& result,
                     const std::vector<Eigen::VectorXd>& vecs,
                     int n_basis,
                     const ActiveSpace& active_space,
                     const CASPT3Config& config);

    CASPT3Result compute();

private:
    SACASResult cas_res_;
    std::vector<Eigen::VectorXd> L_ao_; 
    
    int nbasis_;
    ActiveSpace active_space_;
    CASPT3Config config_;
    
    int n_inact_, n_act_, n_virt_;

    

    double compute_ladder_term(
        int dim1, int dim2, 
        int n_pairs,        
        const std::vector<double>& t2_data, 
        const std::vector<Eigen::MatrixXd>& L_vecs) const;
};

} 

} 


#endif 
