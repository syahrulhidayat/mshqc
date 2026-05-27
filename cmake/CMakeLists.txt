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