# BAB II
# TINJAUAN PUSTAKA DAN LANDASAN TEORI

## 2.1 Tinjauan Pustaka

Penentuan energi korelasi elektron dengan presisi tinggi telah menjadi fokus utama dalam mekanika kuantum molekuler. Metode Hartree-Fock (HF), meskipun memberikan titik awal yang kokoh untuk fungsi gelombang banyak-elektron, mengabaikan interaksi Coulomb seketika antarelektron (korelasi dinamis). Untuk mengoreksi galat ini, Teori Perturbasi Møller-Plesset (MPPT), yang merupakan aplikasi khusus dari Teori Perturbasi Rayleigh-Schrödinger (RSPT), sering digunakan. Koreksi orde kedua (MP2) memberikan pemulihan energi korelasi yang signifikan, dan orde ketiga (MP3) memberikan tambahan presisi pada efek gangguan yang lebih tinggi. 

Kendala fundamental pada metode MP2 dan MP3 klasik berakar pada evaluasi dan penyimpanan matriks integral tolak-menolak dua-elektron (*Electron Repulsion Integral*, ERI). Transformasi tensor ERI dari basis fungsi atomik ke basis orbital molekul menuntut penskalaan komputasi $O(N^5)$, di mana $N$ adalah jumlah basis spasial, yang membatasi penerapan fisis pada molekul berukuran besar.

Sebagai solusi matematis atas kendala dimensionalitas ini, Dekomposisi Cholesky (CD) diimplementasikan untuk memfaktorkan metrik fungsi gelombang. Secara historis, faktorisasi ini telah dibuktikan validitasnya secara matematis oleh Beebe dan Linderberg (1977) yang menunjukkan bahwa matriks ERI bersifat simetris semi-definit positif. Berbeda dengan pendekatan dekomposisi nilai singular (SVD) yang mengharuskan pembentukan matriks utuh sebelum difaktorkan, dekomposisi Cholesky inkomplit memungkinkan pembentukan vektor dasar secara iteratif dengan batas galat (galat residual) yang terkontrol secara analitik.

Penelitian ini memposisikan dekomposisi Cholesky bukan sekadar sebagai aproksimasi komputasi numerik, melainkan sebagai reduksi ruang dimensi vektor ketat pada formalisme mekanika benda-banyak. Dengan mengintegrasikan faktorisasi metrik ini ke dalam kuantisasi kedua, ekspansi perturbasi energi korelasi MP2 dan MP3 dapat dievaluasi tanpa hilangnya kebenaran analitik yang berada di luar batas toleransi $\delta$.

## 2.2 Landasan Teori

### 2.2.1 Persamaan Schrödinger Elektronik

Dengan menerapkan aproksimasi Born-Oppenheimer, yang memisahkan gerak inti atom yang masif dari gerak elektron yang cepat, Hamiltonian sistem banyak-elektron (Hamiltonian elektronik) dalam satuan atomik dirumuskan sebagai:

$$
\hat{H}_{elec} = -\sum_{i=1}^{N} \frac{1}{2}\nabla_i^2 - \sum_{i=1}^{N}\sum_{A=1}^{M} \frac{Z_A}{r_{iA}} + \sum_{i=1}^{N}\sum_{j>i}^{N} \frac{1}{r_{ij}}
$$

Suku pertama dan kedua merupakan operator satu-elektron yang merepresentasikan energi kinetik elektron dan daya tarik Coulomb antara elektron dan inti (disimbolkan dengan operator inti-inti $\hat{h}_i$). Suku ketiga merupakan operator dua-elektron yang mendeskripsikan tolakan antar elektron secara spesifik.

### 2.2.2 Kuantisasi Kedua (*Second Quantization*)

Untuk menangani sistem banyak-elektron yang mematuhi prinsip eksklusi Pauli secara natural, Hamiltonian elektronik diproyeksikan ke dalam formalisme kuantisasi kedua. Fungsi gelombang tidak lagi direpresentasikan secara spasial, melainkan menggunakan aljabar operator kreasi ($\hat{a}^\dagger$) dan anihilasi ($\hat{a}$) yang bekerja pada ruang Fock. Operator-operator ini memenuhi relasi anti-komutasi fermionik:

$$
\{\hat{a}_p^\dagger, \hat{a}_q\} = \delta_{pq}, \qquad \{\hat{a}_p^\dagger, \hat{a}_q^\dagger\} = \{\hat{a}_p, \hat{a}_q\} = 0
$$

Hamiltonian elektronik dalam representasi kuantisasi kedua menggunakan basis spin-orbital dituliskan secara eksak sebagai:

$$
\hat{H} = \sum_{pq} h_{pq} \hat{a}_p^\dagger \hat{a}_q + \frac{1}{4} \sum_{pqrs} \langle pq || rs \rangle \hat{a}_p^\dagger \hat{a}_q^\dagger \hat{a}_s \hat{a}_r
$$

dengan elemen matriks satu-elektron $h_{pq} = \langle p | \hat{h} | q \rangle$ dan elemen matriks dua-elektron antisimetri $\langle pq || rs \rangle = \langle pq | rs \rangle - \langle pq | sr \rangle$.

### 2.2.3 Referensi Hartree-Fock

Metode Hartree-Fock meminimalkan ekspektasi energi dari determinan Slater tunggal $\Phi_0$ melalui variasi orbital. Kondisi stasioner menghasilkan Persamaan integro-diferensial Roothaan-Hall yang secara ekuivalen melahirkan operator Fock $\hat{F}$:

$$
\hat{F} = \hat{h} + \sum_{j} (\hat{J}_j - \hat{K}_j)
$$

dengan $\hat{J}$ dan $\hat{K}$ merupakan operator Coulomb dan pertukaran (*exchange*). Persamaan nilai eigen untuk operator Fock $\hat{F} | \psi_p \rangle = \epsilon_p | \psi_p \rangle$ memberikan set orbital kanonikal dan energi orbital $\epsilon_p$. Energi total Hartree-Fock tidak sekadar merupakan penjumlahan energi orbital, melainkan memiliki koreksi tolakan elektron:

$$
E_{HF} = \langle \Phi_0 | \hat{H} | \Phi_0 \rangle = \sum_{i} h_{ii} + \frac{1}{2} \sum_{ij} \langle ij || ij \rangle
$$
#### 2.2.3.1 Ekstrapolasi Sub-ruang Iteratif (Metode DIIS)

Penyelesaian iteratif persamaan Roothaan-Hall konvensional rentan terhadap divergensi numerik atau osilasi fisis pada sistem dengan celah energi HOMO-LUMO yang sempit. Untuk memastikan konvergensi matematis menuju keadaan dasar yang stasioner, metode *Direct Inversion in the Iterative Subspace* (DIIS) yang diformulasikan oleh Pulay diterapkan.

Kondisi stasioner konvergensi medan-rerata (*mean-field*) tercapai ketika matriks Fock dan matriks Densitas komut pada basis ortogonal. Galat fisis pada iterasi ke-$i$ direpresentasikan oleh vektor komutator (matriks galat):

$$
\mathbf{e}_i = \mathbf{F}_i \mathbf{P}_i \mathbf{S} - \mathbf{S} \mathbf{P}_i \mathbf{F}_i
$$

dengan $\mathbf{F}_i, \mathbf{P}_i,$ dan $\mathbf{S}$ secara berurutan adalah matriks Fock, Densitas, dan Tumpang-tindih (*Overlap*). Metode DIIS mengasumsikan matriks Fock pada iterasi berikutnya sebagai kombinasi linear dari matriks-matriks Fock pada iterasi sebelumnya:

$$
\mathbf{F}^* = \sum_{i=1}^{n} c_i \mathbf{F}_i
$$

Koefisien ekspansi $c_i$ ditentukan secara variasi dengan meminimalkan norma galat residual $\mathbf{e}^* = \sum c_i \mathbf{e}_i$ di bawah batasan normalisasi $\sum c_i = 1$. Minimisasi ini diselesaikan menggunakan metode pengali Lagrange, yang menghasilkan sistem persamaan linear:

$$
\begin{pmatrix}
B_{11} & B_{12} & \cdots & B_{1n} & -1 \\
B_{21} & B_{22} & \cdots & B_{2n} & -1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
B_{n1} & B_{n2} & \cdots & B_{nn} & -1 \\
-1 & -1 & \cdots & -1 & 0
\end{pmatrix}
\begin{pmatrix}
c_1 \\ c_2 \\ \vdots \\ c_n \\ \lambda
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ \vdots \\ 0 \\ -1
\end{pmatrix}
$$

dengan elemen matriks galat silang $B_{ij} = \text{Tr}(\mathbf{e}_i^\dagger \mathbf{e}_j)$. Penyelesaian persamaan ini meniadakan osilasi fisis secara eksak dan memaksa lintasan iterasi konvergen menuju nilai minimum stasioner dari energi Hartree-Fock.

### 2.2.4 Teori Perturbasi Møller-Plesset

Pada teori perturbasi Møller-Plesset (MPPT), Hamiltonian eksak dipartisi menjadi Hamiltonian tak-terganggu $\hat{H}_0$ yang didefinisikan sebagai jumlah operator Fock dari seluruh elektron, dan potensial perturbasi $\hat{V}$ yang merepresentasikan korelasi Coulomb dinamis yang tersisa:

$$
\hat{H}_0 = \sum_{i} \hat{f}(i) = \sum_{p} \epsilon_p \hat{a}_p^\dagger \hat{a}_p
$$

$$
\hat{V} = \hat{H} - \hat{H}_0 = \frac{1}{4} \sum_{pqrs} \langle pq || rs \rangle \hat{a}_p^\dagger \hat{a}_q^\dagger \hat{a}_s \hat{a}_r - \sum_{pq} V^{HF}_{pq} \hat{a}_p^\dagger \hat{a}_q
$$

Ekspansi Rayleigh-Schrödinger menghasilkan koreksi energi orde nol dan orde pertama yang setara dengan energi Hartree-Fock ($E_{HF} = E^{(0)} + E^{(1)}$). Koreksi korelasi dinamis pertama muncul pada orde kedua (MP2). Dengan menerapkan resolusi identitas pada ruang eksitasi ganda, korelasi energi MP2 terformulasi analitik sebagai:

$$
E^{(2)} = \frac{1}{4} \sum_{ijab} \frac{|\langle ij || ab \rangle|^2}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}
$$

di mana indeks $i, j$ merepresentasikan orbital terhuni (*occupied*) dan $a, b$ merepresentasikan orbital maya (*virtual*). Koreksi orde ketiga (MP3) melibatkan interaksi kopling yang lebih kompleks antar ruang eksitasi ganda:

$$
E^{(3)} = \frac{1}{8} \sum_{ijabcd} \frac{\langle ij || ab \rangle \langle ab || cd \rangle \langle cd || ij \rangle}{D_{ijab} D_{ijcd}} + \frac{1}{8} \sum_{ijklab} \frac{\langle ij || ab \rangle \langle kl || ij \rangle \langle ab || kl \rangle}{D_{ijab} D_{klab}} + \dots
$$

dengan $D_{ijab} = \epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b$. Pengevaluasian tensor $\langle ij || ab \rangle$ menuntut operasi basis spasial yang menumbuhkan hambatan penskalaan fisis eksponensial.

### 2.2.5 Aljabar Dekomposisi Cholesky pada Tensor ERI

Integral ERI atas fungsi basis spasial atomik $\mu, \nu, \lambda, \sigma$ dapat direpresentasikan sebagai elemen metrik dua-elektron:

$$
V_{(\mu\nu),(\lambda\sigma)} = (\mu\nu | \lambda\sigma) = \iint \chi_\mu^*(\mathbf{r}_1) \chi_\nu(\mathbf{r}_1) \frac{1}{r_{12}} \chi_\lambda^*(\mathbf{r}_2) \chi_\sigma(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2
$$

Matriks supersimetris $\mathbf{V}$ berdimensi $N^2 \times N^2$ ini secara matematis bersifat semi-definit positif, sebab operator interaksi $1/r_{12}$ merupakan operator positif. Oleh karena itu, berdasarkan prinsip aljabar matriks, elemen matriks ini mematuhi batas Cauchy-Schwarz:

$$
|(\mu\nu|\lambda\sigma)| \le \sqrt{(\mu\nu|\mu\nu)(\lambda\sigma|\lambda\sigma)}
$$

Batas ketat ini memungkinkan matriks $\mathbf{V}$ difaktorkan menggunakan Dekomposisi Cholesky inkomplit. Matriks faktorisasi direduksi menjadi himpunan vektor Cholesky $L_{\mu\nu}^P$ melalui ekspansi:

$$
(\mu\nu | \lambda\sigma) \approx \sum_{P=1}^{N_{CD}} L_{\mu\nu}^P L_{\lambda\sigma}^P
$$

di mana jumlah vektor Cholesky $N_{CD}$ jauh lebih kecil dari dimensi asli $N^2$. Algoritma faktorisasi ini didikte oleh ambang batas toleransi $\delta$. Nilai diagonal dievaluasi secara iteratif, dan dekomposisi dihentikan ketika elemen diagonal residual mematuhi:

$$
\Delta_{\mu\nu} = (\mu\nu | \mu\nu) - \sum_{P=1}^{M} (L_{\mu\nu}^P)^2 \le \delta
$$

Transformasi dari basis atomik ke basis molekul (orbital $i, j, a, b$) yang secara konvensional berskala $O(N^5)$, dapat didekomposisi menjadi transformasi vektor Cholesky yang berskala $O(N^2 \cdot N_{CD})$. Elemen integral orbital molekul diekstrak tanpa salinan spasial yang redundan melalui kontraksi:

$$
(ia | jb) = \sum_{P=1}^{N_{CD}} L_{ia}^P L_{jb}^P, \qquad \text{dengan} \qquad L_{ia}^P = \sum_{\mu\nu} C_{\mu i}^* C_{\nu a} L_{\mu\nu}^P
$$

Melalui dekomposisi analitik ini, tensor 4-indeks pada perluasan $E^{(2)}$ dan $E^{(3)}$ direduksi secara fisis menjadi produk tensor 3-indeks yang meruntuhkan batasan eksponensial komputasi matriks determinan tanpa menyalahi prinsip mekanika kuantum dari energi korelasi itu sendiri.
### 2.2.6 Optimasi Orbital pada Teori Perturbasi (OMP2 dan OMP3)

Kelemahan mendasar dari ekspansi MP2 dan MP3 kanonikal adalah ketergantungannya yang absolut pada fungsi gelombang referensi orde-nol (determinan Hartree-Fock). Ketika sistem molekuler mengalami peregangan ikatan kovalen yang drastis, pemisahan energi antara orbital terhuni (*occupied*) dan orbital maya (*virtual*) mendekati nol, menyebabkan deret perturbasi berdivergensi (gagal secara fisis). 

Untuk merestorasi kestabilan matematis fungsi gelombang, metode *Orbital-Optimized* Møller-Plesset (OMP2 dan OMP3) diterapkan dengan mengeksekusi relaksasi orbital secara swakonsisten (*self-consistent*) di bawah pengaruh potensial korelasi.

Transformasi orbital molekul (rotasi orbital) diformulasikan menggunakan operator kesatuan (*unitary operator*) eksponensial yang bergantung pada matriks anti-Hermitian $\mathbf{\kappa}$ ($\kappa^\dagger = -\kappa$):

$$
\mathbf{C}(\kappa) = \mathbf{C}_0 \exp(\mathbf{\kappa})
$$

Elemen matriks tak-nol dari $\mathbf{\kappa}$ hanya terdapat pada blok pencampuran antara orbital terhuni ($i, j$) dan orbital maya ($a, b$), yaitu $\kappa_{ai} = -\kappa_{ia}^*$. Parameter rotasi ini memungkinkan orbital untuk beradaptasi melampaui limit medan-rerata. 

Energi korelasi OMP2 dan OMP3 tidak lagi dievaluasi sebagai nilai eigen statis, melainkan sebagai fungsional yang bergantung pada parameter rotasi orbital $\kappa$:

$$
E_{OMP}(\kappa) = \langle \Phi_0(\kappa) | \hat{H} | \Phi_0(\kappa) \rangle + E_{corr}^{(n)}(\kappa)
$$

dengan $E_{corr}^{(n)}$ adalah koreksi perturbasi orde ke-$n$ ($n=2, 3$). Kondisi stasioner untuk mengekstrak energi eksak menuntut gradien energi terhadap semua variabel rotasi orbital bernilai nol:

$$
\frac{\partial E_{OMP}}{\partial \kappa_{ai}} = 0 \qquad \forall a, i
$$

Alih-alih memecahkan derivatif ini melalui metode beda hingga (*finite difference*) yang mengorbankan akurasi analitik, formulasi analitik dicapai dengan mendefinisikan fungsional Lagrangian $\mathcal{L}$ yang memaksakan kepatuhan pada kondisi Brillouin, sehingga rotasi orbital dapat dipetakan langsung ke dalam pembentukan matriks densitas satu-partikel ($P_{pq}$) dan matriks densitas dua-partikel ($\Gamma_{pqrs}$) terelaksasi:

$$
\mathcal{L} = E_{OMP}(\kappa) + \sum_{ai} z_{ai} \frac{\partial E_{HF}}{\partial \kappa_{ai}}
$$

Di mana $z_{ai}$ adalah amplitudo vektor-Z yang memecahkan persamaan *Coupled-Perturbed Hartree-Fock* (CPHF). Melalui substitusi ini, gradien orbital secara analitik direduksi menjadi operasi kontraksi tensor antara elemen matriks integral dan matriks densitas terelaksasi:

$$
\frac{\partial \mathcal{L}}{\partial \kappa_{ai}} = 2 \sum_{p} h_{ap} P_{pi} + 2 \sum_{pqr} \langle ap || qr \rangle \Gamma_{piqr}
$$

Dengan mengeksekusi iterasi *Newton-Raphson* pada kondisi stasioner gradien Lagrangian ini, metode OMP2 dan OMP3 menghasilkan matriks densitas yang merepresentasikan keadaan terganggu secara eksak. Formulasi analitik ini membebaskan sistem dari kelemahan referensi statis dan memperbaiki deskripsi fisis kurva disosiasi molekuler secara signifikan.