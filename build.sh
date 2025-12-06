cd ~/mshqc
rm -f ump3_test
rm -f src/mp3/*.o
rm -f src/integrals/*.o
rm -f ump4_test
find . -name "*.o" -type f -delete

g++ "-std=c++17" "-O3" "-fopenmp" \
    "-I/home/syahrul/mshqc/include" \
    "-I/usr/include/eigen3" \
    \
    examples/mp_tests/ump4_test.cc \
    src/core/molecule.cc \
    src/core/basis.cc \
    src/core/integrals.cc \
    src/scf/uhf.cc \
    src/scf/diis.cc \
    src/mp2/ump2.cc \
    src/mp3/ump3.cc \
    src/mp/ump4.cc \
    src/integrals/eri_transformer.cc \
    src/core/spherical_transformer.cc \
    src/core/spherical_integration.cc \
    \
    "-lint2" \
    "-o" ump4_test