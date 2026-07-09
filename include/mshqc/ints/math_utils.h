#ifndef MSHQC_INTS_MATH_UTILS_H
#define MSHQC_INTS_MATH_UTILS_H




#include <Eigen/Dense>










#include <unsupported/Eigen/CXX11/Tensor>
#ifdef I
#undef I
#endif

namespace mshqc {
namespace utils {

    


    using Tensor4D = Eigen::Tensor<double, 4>;
    using Tensor2D = Eigen::Tensor<double, 2>;

    


    


    


    Eigen::MatrixXd matrix_exponential(const Eigen::MatrixXd& mat);

    


    


    void set_zero(Tensor4D& tensor);

} 


} 



#endif 

