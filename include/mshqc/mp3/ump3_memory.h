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

#ifndef MSHQC_UMP3_MEMORY_H
#define MSHQC_UMP3_MEMORY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <map>
#include <string>
#include <iostream>
#ifdef I
#undef I
#endif

namespace mshqc {

class UMP3Workspace {
public:

    using Tensor4D = Eigen::Tensor<double, 4>;

    UMP3Workspace() = default;
    ~UMP3Workspace() { clear_all(); }

    Tensor4D& allocate(const std::string& key, long d1, long d2, long d3, long d4);

    Tensor4D& get(const std::string& key);

    void free(const std::string& key);

    void clear_all();

    double get_memory_usage_mb() const;

private:
    std::map<std::string, Tensor4D> storage_;
};

}

#endif
