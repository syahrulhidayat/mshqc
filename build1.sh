# 1. Masuk ke direktori proyek
cd ~/mshqc

# 2. Hapus file objek lama dan executable lama agar bersih
rm -f rmp_test
find . -name "*.o" -type f -delete

# 3. Compile RMP Test (RHF + RMP2 + RMP3)
g++ -std=c++17 -O3 -fopenmp \
    -I/home/syahrul/mshqc/include \
    -I/usr/include/eigen3 \
    \
    examples/rmp_test.cc \
    src/core/molecule.cc \
    src/core/basis.cc \
    src/core/integrals.cc \
    src/scf/rhf.cc \
    src/scf/diis.cc \
    src/foundation/rmp2.cc \
    src/foundation/rmp3.cc \
    src/foundation/rmp4.cc \
    src/integrals/eri_transformer.cc \
    src/core/spherical_transformer.cc \
    src/core/spherical_integration.cc \
    \
    -lint2 \
    -o rmp_test