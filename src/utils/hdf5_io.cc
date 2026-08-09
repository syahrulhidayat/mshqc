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

#include "mshqc/utils/hdf5_io.h"
#include <iostream>
#include <stdexcept>

namespace mshqc {
namespace utils {

HDF5TensorIO::HDF5TensorIO(const std::string& filename, Mode mode) : filename_(filename) {
    try {
        if (mode == Mode::WRITE_TRUNCATE) {
            file_ = std::make_unique<H5::H5File>(filename_, H5F_ACC_TRUNC);
        } else if (mode == Mode::READ_WRITE) {
            file_ = std::make_unique<H5::H5File>(filename_, H5F_ACC_RDWR);
        } else {
            file_ = std::make_unique<H5::H5File>(filename_, H5F_ACC_RDONLY);
        }
    } catch (const H5::Exception& e) {
        throw std::runtime_error("HDF5 Error opening file: " + filename_);
    }
}

HDF5TensorIO::~HDF5TensorIO() {
    if (file_) {
        file_->close();
    }
}

std::vector<hsize_t> HDF5TensorIO::to_hsize(const std::array<long, 4>& dims) const {
    return {static_cast<hsize_t>(dims[0]), static_cast<hsize_t>(dims[1]),
            static_cast<hsize_t>(dims[2]), static_cast<hsize_t>(dims[3])};
}

void HDF5TensorIO::create_dataset_4d(const std::string& dataset_name,
                                     const std::array<long, 4>& dims,
                                     const std::array<long, 4>& chunk_dims) {
    auto h_dims = to_hsize(dims);
    auto h_chunks = to_hsize(chunk_dims);

    H5::DataSpace dataspace(4, h_dims.data());
    H5::DSetCreatPropList plist;
    plist.setChunk(4, h_chunks.data());
    plist.setDeflate(6);

    file_->createDataSet(dataset_name, H5::PredType::NATIVE_DOUBLE, dataspace, plist);
}

void HDF5TensorIO::write_tensor_4d(const std::string& dataset_name,
                                   const Eigen::Tensor<double, 4>& tensor) {
    H5::DataSet dataset = file_->openDataSet(dataset_name);
    dataset.write(tensor.data(), H5::PredType::NATIVE_DOUBLE);
}

Eigen::Tensor<double, 4> HDF5TensorIO::read_tensor_4d(const std::string& dataset_name) {
    H5::DataSet dataset = file_->openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();

    std::vector<hsize_t> dims_out(4);
    dataspace.getSimpleExtentDims(dims_out.data(), nullptr);

    Eigen::Tensor<double, 4> tensor(dims_out[0], dims_out[1], dims_out[2], dims_out[3]);
    dataset.read(tensor.data(), H5::PredType::NATIVE_DOUBLE, H5::DataSpace::ALL, dataspace);

    return tensor;
}

void HDF5TensorIO::write_slice_4d(const std::string& dataset_name,
                                  const std::array<long, 4>& offset,
                                  const std::array<long, 4>& slice_dims,
                                  const double* data_ptr) {
    H5::DataSet dataset = file_->openDataSet(dataset_name);
    H5::DataSpace memspace(4, to_hsize(slice_dims).data());

    H5::DataSpace filespace = dataset.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, to_hsize(slice_dims).data(), to_hsize(offset).data());

    dataset.write(data_ptr, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
}

void HDF5TensorIO::read_slice_4d(const std::string& dataset_name,
                                 const std::array<long, 4>& offset,
                                 const std::array<long, 4>& slice_dims,
                                 double* data_ptr) {
    H5::DataSet dataset = file_->openDataSet(dataset_name);
    H5::DataSpace memspace(4, to_hsize(slice_dims).data());

    H5::DataSpace filespace = dataset.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, to_hsize(slice_dims).data(), to_hsize(offset).data());

    dataset.read(data_ptr, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
}

}
}
