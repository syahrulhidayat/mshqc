Converged at vector 1175
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

E = -7.432719 Ha
  [STEP 3] SA-CASSCF... Done. (10.915797s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.313833s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (735.622068s)

====================================================================================================
  RESULTS: aug-cc-pCVQZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432695 | -0.046572 | 0.035564 | -7.44370282 | 0.00000
   1  | -7.309954 | -0.041376 | 0.035518 | -7.31581263 | 0.12789
   2  | -7.297395 | -0.041424 | 0.035549 | -7.30327072 | 0.14043
====================================================================================================

syahrul@syahrul:~/mshqc$ bash build15.sh
Cleaning up...
==================================================
Compiling Cholesky SA-CASSCF Benchmark...
==================================================
In file included from /usr/include/eigen3/unsupported/Eigen/CXX11/Tensor:127,
                 from /home/syahrul/mshqc/include/mshqc/integrals.h:7,
                 from /home/syahrul/mshqc/include/mshqc/scf.h:22,
                 from src/scf/uhf.cc:12:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h: In instantiation of ‘Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::Scalar& Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::operator()(Index, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; Scalar_ = double; int NumIndices_ = 4; int Options_ = 0; IndexType_ = long int; Scalar = double; Index = long int]’:
src/scf/uhf.cc:131:37:   required from here
  131 |                     double val = eri(mu, nu, lam, sig);
      |                                  ~~~^~~~~~~~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  266 |       return operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
      |                                                                            ^~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#1’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
In file included from /usr/include/eigen3/unsupported/Eigen/CXX11/Tensor:127,
                 from /home/syahrul/mshqc/include/mshqc/integrals.h:7,
                 from /home/syahrul/mshqc/include/mshqc/scf.h:22,
                 from src/scf/rhf.cc:28:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h: In instantiation of ‘Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::Scalar& Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::operator()(Index, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; Scalar_ = double; int NumIndices_ = 4; int Options_ = 0; IndexType_ = long int; Scalar = double; Index = long int]’:
src/scf/rhf.cc:148:40:   required from here
  148 |                     double j_term = eri(mu, nu, lam, sig);
      |                                     ~~~^~~~~~~~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  266 |       return operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
      |                                                                            ^~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#1’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
In file included from /usr/include/eigen3/unsupported/Eigen/CXX11/Tensor:71,
                 from /home/syahrul/mshqc/include/mshqc/integrals.h:7,
                 from src/core/integrals.cc:26:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h: In instantiation of ‘Eigen::DSizes<DenseIndex, NumDims>::DSizes(DenseIndex, DenseIndex, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; DenseIndex = long int; int NumDims = 4]’:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h:87:135:   required from ‘Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>::TensorMap(StoragePointerType, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int, long unsigned int}; PlainObjectType = Eigen::Tensor<double, 4>; int Options_ = 0; MakePointer_ = Eigen::MakePointer; StoragePointerType = double*; Index = long int]’
   87 | EIGEN_STRONG_INLINE TensorMap(StoragePointerType dataPtr, Index firstDimension, IndexTypes... otherDimensions) : m_data(dataPtr), m_dimensions(firstDimension, otherDimensions...) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src/core/integrals.cc:372:5:   required from here
  372 |     );
      |     ^
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h:340:167: peringatan: narrowing conversion of ‘otherDimensions#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  340 | s(DenseIndex firstDimension, DenseIndex secondDimension, IndexTypes... otherDimensions) : Base({{firstDimension, secondDimension, otherDimensions...}}) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h:340:167: peringatan: narrowing conversion of ‘otherDimensions#1’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h: In instantiation of ‘Eigen::DSizes<DenseIndex, NumDims>::DSizes(DenseIndex, DenseIndex, IndexTypes ...) [with IndexTypes = {long unsigned int}; DenseIndex = long int; int NumDims = 3]’:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h:87:135:   required from ‘Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>::TensorMap(StoragePointerType, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; PlainObjectType = Eigen::Tensor<double, 3>; int Options_ = 0; MakePointer_ = Eigen::MakePointer; StoragePointerType = double*; Index = long int]’
   87 | EIGEN_STRONG_INLINE TensorMap(StoragePointerType dataPtr, Index firstDimension, IndexTypes... otherDimensions) : m_data(dataPtr), m_dimensions(firstDimension, otherDimensions...) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src/core/integrals.cc:659:5:   required from here
  659 |     );
      |     ^
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h:340:167: peringatan: narrowing conversion of ‘otherDimensions#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  340 | s(DenseIndex firstDimension, DenseIndex secondDimension, IndexTypes... otherDimensions) : Base({{firstDimension, secondDimension, otherDimensions...}}) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~
==================================================
Build SUCCESS! Output: bench_sa_casscf
Running Benchmark...
==================================================
STATE-SPECIFIC CASPT3 BENCHMARK (CLEAN OUTPUT)
Note: Ensure 'config.shift = 0' used for both PT2/PT3.

================================================================================
  BENCHMARK: cc-pVTZ (Li Atom)
================================================================================
  N Basis: 30
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.000000e-08 Ha
======================================================================

Basis functions: 30
Orbital pairs: 900
Full ERI storage: 6.179810e+00 MB

Initializing diagonal elements...
Initial max diagonal: 8.827692e+06 Ha
Target threshold: 1.000000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 316
Max remaining diagonal: 6.366907e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 316
Basis functions: 30
Ratio M/N: 10.53
Compression ratio: 2.8×
Storage: 2.2 MB
======================================================================

Done. (0.2s, 316 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 30
Orbital pairs: 900
Full ERI storage: 6.2e+00 MB

Initializing diagonal elements...
Initial max diagonal: 8.8e+06 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 316
Max remaining diagonal: 6.4e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 316
Basis functions: 30
Ratio M/N: 10.53
Compression ratio: 2.8×
Storage: 2.2 MB
======================================================================

E = -7.432702 Ha
  [STEP 3] SA-CASSCF... Done. (0.220411s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.001834s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (0.148840s)

====================================================================================================
  RESULTS: cc-pVTZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432679 | -0.010762 | 0.001275 | -7.44216579 | 0.00000
   1  | -7.364988 | -0.016376 | 0.001693 | -7.37967103 | 0.06249
   2  | -7.276052 | -0.112498 | 0.143113 | -7.24543672 | 0.19673
====================================================================================================


================================================================================
  BENCHMARK: cc-pVQZ (Li Atom)
================================================================================
  N Basis: 55
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 55
Orbital pairs: 3025
Full ERI storage: 6.98137e+01 MB

Initializing diagonal elements...
Initial max diagonal: 1.60491e+07 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 650
Max remaining diagonal: 9.92556e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 650
Basis functions: 55
Ratio M/N: 11.82
Compression ratio: 4.7×
Storage: 15.0 MB
======================================================================

Done. (2.3s, 650 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 55
Orbital pairs: 3025
Full ERI storage: 7.0e+01 MB

Initializing diagonal elements...
Initial max diagonal: 1.6e+07 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 650
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 650
Basis functions: 55
Ratio M/N: 11.82
Compression ratio: 4.7×
Storage: 15.0 MB
======================================================================

E = -7.432718 Ha
  [STEP 3] SA-CASSCF... Done. (1.193507s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.030670s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (20.200068s)

====================================================================================================
  RESULTS: cc-pVQZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432695 | -0.015905 | 0.004248 | -7.44435179 | 0.00000
   1  | -7.361876 | -0.022398 | 0.008303 | -7.37597125 | 0.06838
   2  | -7.296806 | -0.013094 | 0.003285 | -7.30661490 | 0.13774
====================================================================================================


================================================================================
  BENCHMARK: aug-cc-pCVTZ (Li Atom)
================================================================================
  N Basis: 59
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 59
Orbital pairs: 3481
Full ERI storage: 9.24481e+01 MB

Initializing diagonal elements...
Initial max diagonal: 3.41235e+08 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 590
Max remaining diagonal: 9.92152e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 590
Basis functions: 59
Ratio M/N: 10.00
Compression ratio: 5.9×
Storage: 15.7 MB
======================================================================

Done. (2.0s, 590 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 59
Orbital pairs: 3481
Full ERI storage: 9.2e+01 MB

Initializing diagonal elements...
Initial max diagonal: 3.4e+08 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 590
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 590
Basis functions: 59
Ratio M/N: 10.00
Compression ratio: 5.9×
Storage: 15.7 MB
======================================================================

E = -7.432706 Ha
  [STEP 3] SA-CASSCF... Done. (1.115589s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.033148s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (27.944191s)

====================================================================================================
  RESULTS: aug-cc-pCVTZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432682 | -0.042522 | 0.005819 | -7.46938476 | 0.00000
   1  | -7.248117 | -0.038562 | 0.006134 | -7.28054475 | 0.18884
   2  | -7.174404 | -0.039972 | 0.007311 | -7.20706470 | 0.26232
====================================================================================================


================================================================================
  BENCHMARK: aug-cc-pCVQZ (Li Atom)
================================================================================
  N Basis: 109
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 109
Orbital pairs: 11881
Full ERI storage: 1.07695e+03 MB

Initializing diagonal elements...
Initial max diagonal: 1.56084e+10 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 1175
Max remaining diagonal: 9.87225e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

Done. (32.6s, 1175 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 109
Orbital pairs: 11881
Full ERI storage: 1.1e+03 MB

Initializing diagonal elements...
Initial max diagonal: 1.6e+10 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 1175
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

E = -7.432719 Ha
  [STEP 3] SA-CASSCF... Done. (8.958017s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.130346s)
  [STEP 5] CASPT3 (Shift=0.000000)... ^C
syahrul@syahrul:~/mshqc$ bash build 15.sh
build: build: Adalah sebuah direktori
syahrul@syahrul:~/mshqc$ bash build15.sh
Cleaning up...
==================================================
Compiling Cholesky SA-CASSCF Benchmark...
==================================================
In file included from /usr/include/eigen3/unsupported/Eigen/CXX11/Tensor:127,
                 from /home/syahrul/mshqc/include/mshqc/integrals.h:7,
                 from /home/syahrul/mshqc/include/mshqc/scf.h:22,
                 from src/scf/uhf.cc:12:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h: In instantiation of ‘Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::Scalar& Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::operator()(Index, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; Scalar_ = double; int NumIndices_ = 4; int Options_ = 0; IndexType_ = long int; Scalar = double; Index = long int]’:
src/scf/uhf.cc:131:37:   required from here
  131 |                     double val = eri(mu, nu, lam, sig);
      |                                  ~~~^~~~~~~~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  266 |       return operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
      |                                                                            ^~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#1’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
In file included from /usr/include/eigen3/unsupported/Eigen/CXX11/Tensor:127,
                 from /home/syahrul/mshqc/include/mshqc/integrals.h:7,
                 from /home/syahrul/mshqc/include/mshqc/scf.h:22,
                 from src/scf/rhf.cc:28:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h: In instantiation of ‘Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::Scalar& Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>::operator()(Index, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; Scalar_ = double; int NumIndices_ = 4; int Options_ = 0; IndexType_ = long int; Scalar = double; Index = long int]’:
src/scf/rhf.cc:148:40:   required from here
  148 |                     double j_term = eri(mu, nu, lam, sig);
      |                                     ~~~^~~~~~~~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  266 |       return operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
      |                                                                            ^~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/Tensor.h:266:76: peringatan: narrowing conversion of ‘otherIndices#1’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
In file included from /usr/include/eigen3/unsupported/Eigen/CXX11/Tensor:71,
                 from /home/syahrul/mshqc/include/mshqc/integrals.h:7,
                 from src/core/integrals.cc:26:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h: In instantiation of ‘Eigen::DSizes<DenseIndex, NumDims>::DSizes(DenseIndex, DenseIndex, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; DenseIndex = long int; int NumDims = 4]’:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h:87:135:   required from ‘Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>::TensorMap(StoragePointerType, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int, long unsigned int}; PlainObjectType = Eigen::Tensor<double, 4>; int Options_ = 0; MakePointer_ = Eigen::MakePointer; StoragePointerType = double*; Index = long int]’
   87 | EIGEN_STRONG_INLINE TensorMap(StoragePointerType dataPtr, Index firstDimension, IndexTypes... otherDimensions) : m_data(dataPtr), m_dimensions(firstDimension, otherDimensions...) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src/core/integrals.cc:372:5:   required from here
  372 |     );
      |     ^
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h:340:167: peringatan: narrowing conversion of ‘otherDimensions#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  340 | s(DenseIndex firstDimension, DenseIndex secondDimension, IndexTypes... otherDimensions) : Base({{firstDimension, secondDimension, otherDimensions...}}) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h:340:167: peringatan: narrowing conversion of ‘otherDimensions#1’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h: In instantiation of ‘Eigen::DSizes<DenseIndex, NumDims>::DSizes(DenseIndex, DenseIndex, IndexTypes ...) [with IndexTypes = {long unsigned int}; DenseIndex = long int; int NumDims = 3]’:
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h:87:135:   required from ‘Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>::TensorMap(StoragePointerType, Index, IndexTypes ...) [with IndexTypes = {long unsigned int, long unsigned int}; PlainObjectType = Eigen::Tensor<double, 3>; int Options_ = 0; MakePointer_ = Eigen::MakePointer; StoragePointerType = double*; Index = long int]’
   87 | EIGEN_STRONG_INLINE TensorMap(StoragePointerType dataPtr, Index firstDimension, IndexTypes... otherDimensions) : m_data(dataPtr), m_dimensions(firstDimension, otherDimensions...) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src/core/integrals.cc:659:5:   required from here
  659 |     );
      |     ^
/usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h:340:167: peringatan: narrowing conversion of ‘otherDimensions#0’ from ‘long unsigned int’ to ‘long int’ [-Wnarrowing]
  340 | s(DenseIndex firstDimension, DenseIndex secondDimension, IndexTypes... otherDimensions) : Base({{firstDimension, secondDimension, otherDimensions...}}) {
      |                                                                                                                                   ^~~~~~~~~~~~~~~
==================================================
Build SUCCESS! Output: bench_sa_casscf
Running Benchmark...
==================================================
STATE-SPECIFIC CASPT3 BENCHMARK (CLEAN OUTPUT)
Note: Ensure 'config.shift = 0' used for both PT2/PT3.

================================================================================
  BENCHMARK: cc-pVTZ (Li Atom)
================================================================================
  N Basis: 30
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.000000e-08 Ha
======================================================================

Basis functions: 30
Orbital pairs: 900
Full ERI storage: 6.179810e+00 MB

Initializing diagonal elements...
Initial max diagonal: 8.827692e+06 Ha
Target threshold: 1.000000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 316
Max remaining diagonal: 6.366907e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 316
Basis functions: 30
Ratio M/N: 10.53
Compression ratio: 2.8×
Storage: 2.2 MB
======================================================================

Done. (0.2s, 316 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 30
Orbital pairs: 900
Full ERI storage: 6.2e+00 MB

Initializing diagonal elements...
Initial max diagonal: 8.8e+06 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 316
Max remaining diagonal: 6.4e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 316
Basis functions: 30
Ratio M/N: 10.53
Compression ratio: 2.8×
Storage: 2.2 MB
======================================================================

E = -7.432702 Ha
  [STEP 3] SA-CASSCF... Done. (0.240669s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.001680s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (0.010877s)

====================================================================================================
  RESULTS: cc-pVTZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432679 | -0.010639 | 0.002761 | -7.44055663 | 0.00000
   1  | -7.364988 | -0.015753 | 0.002813 | -7.37792859 | 0.06263
   2  | -7.337928 | -0.105927 | 0.005624 | -7.43823149 | 0.00233
====================================================================================================


================================================================================
  BENCHMARK: cc-pVQZ (Li Atom)
================================================================================
  N Basis: 55
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 55
Orbital pairs: 3025
Full ERI storage: 6.98137e+01 MB

Initializing diagonal elements...
Initial max diagonal: 1.60491e+07 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 650
Max remaining diagonal: 9.92556e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 650
Basis functions: 55
Ratio M/N: 11.82
Compression ratio: 4.7×
Storage: 15.0 MB
======================================================================

Done. (2.3s, 650 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 55
Orbital pairs: 3025
Full ERI storage: 7.0e+01 MB

Initializing diagonal elements...
Initial max diagonal: 1.6e+07 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 650
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 650
Basis functions: 55
Ratio M/N: 11.82
Compression ratio: 4.7×
Storage: 15.0 MB
======================================================================

E = -7.432718 Ha
  [STEP 3] SA-CASSCF... Done. (1.222056s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.019828s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (0.077111s)

====================================================================================================
  RESULTS: cc-pVQZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432695 | -0.015904 | 0.003608 | -7.44499134 | 0.00000
   1  | -7.296806 | -0.013019 | 0.003695 | -7.30613019 | 0.13886
   2  | -7.281337 | -0.180938 | -0.002802 | -7.46507691 | -0.02009
====================================================================================================


================================================================================
  BENCHMARK: aug-cc-pCVTZ (Li Atom)
================================================================================
  N Basis: 59
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 59
Orbital pairs: 3481
Full ERI storage: 9.24481e+01 MB

Initializing diagonal elements...
Initial max diagonal: 3.41235e+08 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 590
Max remaining diagonal: 9.92152e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 590
Basis functions: 59
Ratio M/N: 10.00
Compression ratio: 5.9×
Storage: 15.7 MB
======================================================================

Done. (2.1s, 590 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 59
Orbital pairs: 3481
Full ERI storage: 9.2e+01 MB

Initializing diagonal elements...
Initial max diagonal: 3.4e+08 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 590
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 590
Basis functions: 59
Ratio M/N: 10.00
Compression ratio: 5.9×
Storage: 15.7 MB
======================================================================

E = -7.432705 Ha
  [STEP 3] SA-CASSCF... Done. (1.278373s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.020329s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (0.089444s)

====================================================================================================
  RESULTS: aug-cc-pCVTZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432682 | -0.042803 | 0.018793 | -7.45669188 | 0.00000
   1  | -7.343918 | -0.042136 | 0.018784 | -7.36727021 | 0.08942
   2  | -7.267615 | -0.041138 | 0.018802 | -7.28995204 | 0.16674
====================================================================================================


================================================================================
  BENCHMARK: aug-cc-pCVQZ (Li Atom)
================================================================================
  N Basis: 109
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 109
Orbital pairs: 11881
Full ERI storage: 1.07695e+03 MB

Initializing diagonal elements...
Initial max diagonal: 1.56084e+10 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 1175
Max remaining diagonal: 9.87225e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

Done. (33.3s, 1175 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 109
Orbital pairs: 11881
Full ERI storage: 1.1e+03 MB

Initializing diagonal elements...
Initial max diagonal: 1.6e+10 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 1175
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

E = -7.432719 Ha
  [STEP 3] SA-CASSCF... Done. (9.046314s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.135204s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (1.255091s)

====================================================================================================
  RESULTS: aug-cc-pCVQZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432695 | -0.046569 | 0.030613 | -7.44865132 | 0.00000
   1  | -7.309952 | -0.041385 | 0.030615 | -7.32072254 | 0.12793
   2  | -7.297395 | -0.041435 | 0.030615 | -7.30821573 | 0.14044
====================================================================================================




~~~
==================================================
Build SUCCESS! Output: bench_sa_casscf
Running Benchmark...
==================================================
STATE-SPECIFIC CASPT3 BENCHMARK (CLEAN OUTPUT)
Note: Ensure 'config.shift = 0' used for both PT2/PT3.

================================================================================
  BENCHMARK: cc-pVTZ (Li Atom)
================================================================================
  N Basis: 30
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.000000e-08 Ha
======================================================================

Basis functions: 30
Orbital pairs: 900
Full ERI storage: 6.179810e+00 MB

Initializing diagonal elements...
Initial max diagonal: 8.827692e+06 Ha
Target threshold: 1.000000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 316
Max remaining diagonal: 6.366907e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 316
Basis functions: 30
Ratio M/N: 10.53
Compression ratio: 2.8×
Storage: 2.2 MB
======================================================================

Done. (0.2s, 316 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 30
Orbital pairs: 900
Full ERI storage: 6.2e+00 MB

Initializing diagonal elements...
Initial max diagonal: 8.8e+06 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 316
Max remaining diagonal: 6.4e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 316
Basis functions: 30
Ratio M/N: 10.53
Compression ratio: 2.8×
Storage: 2.2 MB
======================================================================

E = -7.432702 Ha
  [STEP 3] SA-CASSCF... Done. (0.215257s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.001666s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (0.217493s)

====================================================================================================
  RESULTS: cc-pVTZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432679 | -0.008764 | 0.001825 | -7.43961770 | 0.00000
   1  | -7.363897 | -0.008764 | 0.001905 | -7.37075474 | 0.06886
   2  | -7.355598 | -0.008764 | 0.008368 | -7.35599343 | 0.08362
====================================================================================================


================================================================================
  BENCHMARK: cc-pVQZ (Li Atom)
================================================================================
  N Basis: 55
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 55
Orbital pairs: 3025
Full ERI storage: 6.98137e+01 MB

Initializing diagonal elements...
Initial max diagonal: 1.60491e+07 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 650
Max remaining diagonal: 9.92556e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 650
Basis functions: 55
Ratio M/N: 11.82
Compression ratio: 4.7×
Storage: 15.0 MB
======================================================================

Done. (2.4s, 650 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 55
Orbital pairs: 3025
Full ERI storage: 7.0e+01 MB

Initializing diagonal elements...
Initial max diagonal: 1.6e+07 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 650
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 650
Basis functions: 55
Ratio M/N: 11.82
Compression ratio: 4.7×
Storage: 15.0 MB
======================================================================

E = -7.432718 Ha
  [STEP 3] SA-CASSCF... Done. (1.220760s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.028848s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (22.613802s)

====================================================================================================
  RESULTS: cc-pVQZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432695 | -0.012529 | 0.003077 | -7.44214697 | 0.00000
   1  | -7.363024 | -0.012529 | 0.003761 | -7.37179239 | 0.07035
   2  | -7.296806 | -0.012529 | 0.003047 | -7.30628816 | 0.13586
====================================================================================================


================================================================================
  BENCHMARK: aug-cc-pCVTZ (Li Atom)
================================================================================
  N Basis: 59
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 59
Orbital pairs: 3481
Full ERI storage: 9.24481e+01 MB

Initializing diagonal elements...
Initial max diagonal: 3.41235e+08 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 590
Max remaining diagonal: 9.92152e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 590
Basis functions: 59
Ratio M/N: 10.00
Compression ratio: 5.9×
Storage: 15.7 MB
======================================================================

Done. (2.1s, 590 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 59
Orbital pairs: 3481
Full ERI storage: 9.2e+01 MB

Initializing diagonal elements...
Initial max diagonal: 3.4e+08 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 590
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 590
Basis functions: 59
Ratio M/N: 10.00
Compression ratio: 5.9×
Storage: 15.7 MB
======================================================================

E = -7.432706 Ha
  [STEP 3] SA-CASSCF... Done. (1.194354s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.033710s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (28.852935s)

====================================================================================================
  RESULTS: aug-cc-pCVTZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432682 | -0.035358 | 0.007639 | -7.46040170 | 0.00000
   1  | -7.249312 | -0.035358 | 0.007531 | -7.27713935 | 0.18326
   2  | -7.169166 | -0.035358 | 0.007384 | -7.19714087 | 0.26326
====================================================================================================


================================================================================
  BENCHMARK: aug-cc-pCVQZ (Li Atom)
================================================================================
  N Basis: 109
  [STEP 0] Cholesky Decomposition... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.00000e-08 Ha
======================================================================

Basis functions: 109
Orbital pairs: 11881
Full ERI storage: 1.07695e+03 MB

Initializing diagonal elements...
Initial max diagonal: 1.56084e+10 Ha
Target threshold: 1.00000e-08 Ha

Starting Cholesky decomposition...

Converged at vector 1175
Max remaining diagonal: 9.87225e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

Done. (35.1s, 1175 vecs)
  [STEP 1] UHF... 
======================================================================
Cholesky Decomposition of ERIs
======================================================================
Theory: Koch et al. (2003), Beebe & Linderberg (1977)
Threshold: 1.0e-08 Ha
======================================================================

Basis functions: 109
Orbital pairs: 11881
Full ERI storage: 1.1e+03 MB

Initializing diagonal elements...
Initial max diagonal: 1.6e+10 Ha
Target threshold: 1.0e-08 Ha

Starting Cholesky decomposition...

Converged at vector 1175
Max remaining diagonal: 9.9e-09 Ha

======================================================================
Decomposition Complete
======================================================================
Cholesky vectors: 1175
Basis functions: 109
Ratio M/N: 10.78
Compression ratio: 10.1×
Storage: 106.5 MB
======================================================================

E = -7.432719 Ha
  [STEP 3] SA-CASSCF... Done. (9.534026s)
  [STEP 4] CASPT2 (Shift=0.000000)... ======================================================================

Done. (0.138868s)
  [STEP 5] CASPT3 (Shift=0.000000)... ======================================================================

Done. (977.207064s)

====================================================================================================
  RESULTS: aug-cc-pCVQZ | Unit: Hartree (Ha)
====================================================================================================
  St |   E(CASSCF)   |    E(PT2)    |    E(PT3)    |   E(Total)   | Exc (Ha)
  ---|---------------|--------------|--------------|--------------|---------
   0  | -7.432695 | -0.036987 | 0.007972 | -7.46170995 | 0.00000
   1  | -7.309954 | -0.036987 | 0.008011 | -7.33892992 | 0.12278