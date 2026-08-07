#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <H5Cpp.h>

namespace mshqc {
namespace utils {

class HDF5TensorIO {
public:

    enum class Mode {
        READ_ONLY,
        WRITE_TRUNCATE,

        READ_WRITE

    };

    HDF5TensorIO(const std::string& filename, Mode mode);

    ~HDF5TensorIO();

    void create_dataset_4d(const std::string& dataset_name,
                           const std::array<long, 4>& dims,
                           const std::array<long, 4>& chunk_dims);

    void write_tensor_4d(const std::string& dataset_name,
                         const Eigen::Tensor<double, 4>& tensor);

    Eigen::Tensor<double, 4> read_tensor_4d(const std::string& dataset_name);

    void write_slice_4d(const std::string& dataset_name,
                        const std::array<long, 4>& offset,
                        const std::array<long, 4>& slice_dims,
                        const double* data_ptr);

    void read_slice_4d(const std::string& dataset_name,
                       const std::array<long, 4>& offset,
                       const std::array<long, 4>& slice_dims,
                       double* data_ptr);

private:
    std::string filename_;
    std::unique_ptr<H5::H5File> file_;

    std::vector<hsize_t> to_hsize(const std::array<long, 4>& dims) const;
};

}

}
