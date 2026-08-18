# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang Masalah

Penyelesaian persamaan Schrödinger elektronik bagi sistem banyak-elektron merupakan permasalahan inti dalam fisika kuantum molekuler. Dalam kerangka aproksimasi Born–Oppenheimer, persamaan tersebut mengandung interaksi tolak-menolak Coulomb antarelektron ($1/r_{12}$) yang menyebabkan sistem dengan elektron lebih dari satu ($N > 1$) menjadi tidak dapat dipisahkan (*non-separable*) secara analitik. Oleh karena itu, energi elektronik eksak ($E_{exact}$) hanya dapat dihampiri melalui metode komputasi numerik. Selisih antara energi eksak nonrelativistik dengan energi pada limit medan-rerata (*mean-field*) inilah yang secara fundamental didefinisikan sebagai energi korelasi elektron ($E_{corr} = E_{exact} - E_{HF}$). Besaran energi korelasi ini, meskipun secara kuantitatif bernilai kecil dibandingkan total energi elektronik, sangat krusial dalam mendikte fenomena fisis yang signifikan, seperti energi disosiasi ikatan, interaksi dispersi jarak-jauh, serta topologi struktur permukaan energi potensial pada titik kritis.

Secara metodologis, metode Hartree–Fock (HF) mengawali pendekatan fungsi gelombang banyak-elektron melalui determinan Slater tunggal, di mana setiap elektron diasumsikan bergerak dalam medan efektif rerata yang dibangkitkan oleh elektron-elektron lainnya. Skema HF secara konsisten mampu menangkap efek korelasi Fermi—korelasi pertukaran akibat prinsip eksklusi Pauli—melalui sifat antisimetri determinannya. Akan tetapi, metode ini gagal merepresentasikan korelasi dinamis, yakni fenomena penghindaran posisi instan antarelektron yang diakibatkan oleh interaksi tolak-menolak Coulomb secara seketika. Konsekuensi fisis dari pengabaian ini adalah energi HF yang dihasilkan selalu berada di atas batas atas variasional terhadap energi eksak, sehingga deskripsi kuantitatif terhadap proses pemutusan ikatan maupun interaksi antarmolekul lemah menjadi sangat tidak memadai.

Untuk mengoreksi keterbatasan pada pendekatan medan-rerata tersebut, teori perturbasi Møller–Plesset (MP) memartisi Hamiltonian elektronik eksak menjadi operator Fock tak-terganggu ($\hat{H}_0$) dan potensial fluktuasi ($\hat{V}$). Dengan menerapkan prinsip teori perturbasi Rayleigh–Schrödinger terhadap keadaan referensi HF, energi total sistem dapat diekspansi sebagai sebuah deret koreksi. Koreksi pada orde-kedua (MP2) dan orde-ketiga (MP3) mensyaratkan evaluasi numerik terhadap integral tolak-menolak dua-elektron (*Electron Repulsion Integral*, ERI) di dalam basis orbital molekul.

Namun, evaluasi analitik ini dihadapkan pada limitasi komputasional yang masif. Tensor berindeks-empat pada ERI menuntut alokasi penyimpanan yang diskalakan sebesar $\mathcal{O}(N^4)$ terhadap jumlah fungsi basis ($N$). Transformasi tensor dari basis atomik ke basis molekul berskala $\mathcal{O}(N^5)$, sementara proses komputasi pada ekspansi energi MP3 memicu pelonjakan skala hingga $\mathcal{O}(N^6)$ akibat kehadiran evaluasi diagram tangga partikel-partikel dan lubang-lubang tambahan. Penskalaan derajat polinomial eksponensial ini merupakan kendala fisis-matematis sentral yang membatasi penerapan metode perturbasi orde tinggi pada sistem molekuler berskala besar maupun penggunaan basis set atomik berkualitas tinggi.

Dalam kerangka aljabar linear, metode Dekomposisi Cholesky memberikan jalan keluar analitis untuk mengatasi kendala penskalaan ruang komputasi tersebut. Integral (pq|rs) dapat ditinjau sebagai entri dari sebuah matriks simetris yang bersifat semi-definit positif. Memanfaatkan sifat ini, matriks integral difaktorkan secara tak-lengkap (*pivoted incomplete Cholesky factorization*) sehingga tensor berindeks-empat tersebut tereduksi menjadi produk tensor berindeks-tiga. Proses reduksi ini sangat bergantung pada jumlah vektor Cholesky yang dikendalikan ketat oleh parameter ambang dekomposisi ($\delta$). Pendekatan ini mampu mereduksi hambatan memori penyimpanan serta skala waktu transformasi tanpa mengorbankan ketelitian analitik energi korelasi secara drastis, dengan galat residual yang dapat dikendalikan secara sistematis konvergen menuju limit eksak.

Meskipun Dekomposisi Cholesky telah banyak diimplementasikan pada berbagai paket perangkat lunak kimia kuantum komersial, kajian mendalam mengenai efisiensi dan arsitektur algoritma ini pada lingkungan perangkat lunak hibrida berskala kecil masih terbatas. Oleh karena itu, penelitian ini akan mengimplementasikan dan menguji faktorisasi Cholesky secara mandiri di dalam program *mshqc*, sebuah paket komputasi kimia kuantum berbasis C++ dan Python yang dikembangkan dalam kerangka riset ini. Pemilihan objek fisis akan difokuskan pada sistem deret homolog hidrokarbon linier dan kluster air, guna mengamati secara langsung metrik penskalaan komputasi (*computational scaling*) saat ukuran sistem dan jumlah basis fungsi ditingkatkan. Evaluasi luaran energi korelasi pada tataran MP2 dan MP3 dari *mshqc* selanjutnya akan dikomparasikan terhadap limit analitik referensi standar untuk memvalidasi presisi numerik algoritma yang diimplementasikan.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, permasalahan penelitian dirumuskan sebagai berikut:

1. Bagaimana kinerja implementasi algoritma Dekomposisi Cholesky untuk pereduksian tensor integral dua-elektron pada evaluasi energi MP2 dan MP3 di dalam lingkungan program *mshqc*?
2. Seberapa tinggi tingkat presisi numerik energi korelasi MP2 dan MP3 hasil implementasi algoritma tersebut apabila dikomparasikan dengan perhitungan analitik dari perangkat lunak referensi baku?
3. Bagaimana perilaku penskalaan kompleksitas komputasi (*computational scaling*) dari algoritma faktorisasi tersebut terhadap variasi ukuran sistem molekul uji dan peningkatan resolusi basis set atomik?

## 1.3 Batasan Masalah

Agar kajian tetap terarah pada aspek fisis dan komputasional yang relevan, penelitian ini dibatasi pada hal-hal berikut:

* Evaluasi numerik difokuskan pada pengimplementasian algoritma di dalam program mandiri *mshqc* yang berbasis C++ dan Python.
* Hamiltonian elektronik yang digunakan bersifat nonrelativistik; efek relativistik tidak diperhitungkan dalam formulasi.
* Sistem molekul uji difokuskan pada molekul kulit tertutup (*closed-shell*) dengan referensi *Restricted Hartree–Fock* (RHF).
* Objek fisis molekul uji dibatasi pada deret homolog hidrokarbon alkana (mulai dari $CH_4$ hingga $C_8H_{18}$) dan sistem kluster air asimetris $(H_2O)_n$ dengan $n \leq 6$.
* Fungsi basis atomik yang digunakan terbatas pada keluarga basis set *correlation-consistent* (cc-pVDZ, cc-pVTZ, dan cc-pVQZ).
* Parameter ambang batas dekomposisi (*decomposition threshold*, $\delta$) divariasikan secara sistematis pada rentang batas presisi tertentu (misalnya $10^{-4}$ hingga $10^{-8}$ a.u.).
* Program komputasi Psi4 (atau program setara yang menggunakan limit analitik eksak) digunakan murni sebagai referensi standar untuk menghitung persentase galat energi (*energy error*).

## 1.4 Tujuan Penelitian

Sejalan dengan rumusan masalah, penelitian ini bertujuan untuk:

1. Menguji dan menganalisis kinerja implementasi algoritma Dekomposisi Cholesky pada ekspansi energi korelasi MP2 dan MP3 di dalam basis program *mshqc*.
2. Memvalidasi tingkat presisi numerik energi korelasi yang dihasilkan oleh *mshqc* pada sistem molekul uji melalui perbandingan terhadap limit analitik perangkat lunak referensi.
3. Menentukan secara kuantitatif perilaku penskalaan kompleksitas komputasi pada perhitungan energi korelasi ketika dihadapkan pada peningkatan ukuran molekul dan resolusi fungsi basis.

## 1.5 Manfaat Penelitian

Secara komputasional, penelitian ini membuktikan keberhasilan rekayasa perangkat lunak mandiri (*in-house code*) dalam menangani persoalan kompleksitas polinomial melalui reduksi tensor aljabar. Secara teoretis fisika, kajian ini diharapkan memberikan pemahaman yang komprehensif mengenai batasan toleransi *threshold* dan kompromi antara efisiensi waktu dengan presisi pelacakan korelasi dinamis elektron. Hasil penelitian ini juga dapat menjadi fondasi teknis dan referensi metodologis bagi pengembangan fitur lanjutan pada *mshqc*, seperti metode *Coupled-Cluster*, untuk sistem makromolekul pada riset-riset mendatang.