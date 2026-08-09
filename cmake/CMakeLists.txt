 # ==============================================================================
 # Copyright (c) 2026 Syahrul and mshqc contributors
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/home/syahrul/mshqc/libint_install \
    -DLIBINT2_SHGAUSS_ORDERING=standard \
    -DMAX_AM_ERI=6 \
    -DENABLE_ERI=ON \
    -DENABLE_ERI2=ON \
    -DENABLE_ERI3=ON \
    -DENABLE_GENERIC_CODE=ON
    # Di CMakeLists.txt utama
find_package(TBLIS REQUIRED) 
# Atau jika manual:
# include_directories(/path/to/tblis/include)
# link_directories(/path/to/tblis/lib)

# Saat target_link_libraries:
target_link_libraries(mshqc PRIVATE TBLIS::tblis) 
# atau simplenya: tblis