# BAB III
# METODE PENELITIAN

## 3.1 Objek dan Variabel Penelitian

Penelitian ini merupakan eksperimen fisika komputasi yang berfokus pada evaluasi numerik energi korelasi elektron. Sistem fisis yang dikaji dibatasi pada molekul kulit tertutup (*closed-shell*) dalam keadaan dasar (*ground state*). 

1. **Objek Eksperimen:** Molekul air ($H_2O$), nitrogen ($N_2$), dan karbon monoksida ($CO$). Pemilihan molekul ini didasarkan pada variasi sifat ikatan (ikatan tunggal, rangkap tiga, dan polaritas) untuk menguji stabilitas fungsional korelasi.
2. **Variabel Bebas:** 
   - Kualitas ruang fungsi basis: Menggunakan basis set atomik tipe *correlation-consistent* Dunning (cc-pVDZ, cc-pVTZ, dan cc-pVQZ) untuk mengontrol dimensi matriks $N$.
   - Ambang batas toleransi Dekomposisi Cholesky ($\delta$): Divariasikan pada rentang $10^{-2}$ hingga $10^{-6}$ untuk mengendalikan galat faktorisasi metrik.
3. **Variabel Terikat:**
   - Energi elektronik Hartree-Fock ($E_{HF}$) dan energi korelasi ($E^{(2)}$ dan $E^{(3)}$).
   - Matriks densitas tereksitasi dari rotasi orbital (pada metode OMP2/OMP3).
   - Metrik penskalaan operasi aljabar linier terhadap ekspansi ruang basis $N$.

## 3.2 Instrumen Penelitian

Instrumen utama yang digunakan dalam eksperimen ini adalah program komputasi kimia kuantum mandiri (`mshqc`) yang berfungsi sebagai *solver* matriks determinan Slater dan tensor interaksi elektron. Instrumen numerik ini ditunjang oleh operasi kontraksi tensor termutakhir untuk merepresentasikan operator matematika benda-banyak tanpa kehilangan presisi analitik.

Sebagai pembanding untuk memvalidasi kebenaran fisis (presisi numerik) dari kalkulasi `mshqc`, digunakan instrumen komputasi standar industri (seperti PySCF) yang dioperasikan pada limit analitik yang identik.

## 3.3 Prosedur Eksperimen Komputasi

Alur eksperimen komputasi dijalankan secara deterministik mengikuti runtutan penyelesaian persamaan aljabar Hamiltonian sebagai berikut:

### 3.3.1 Inisialisasi Keadaan dan Integral Elementer
1. Spesifikasi koordinat inti atomik molekul uji pada ruang Euclidean tiga dimensi (dalam satuan Angstrom atau Bohr).
2. Proyeksi fungsi basis set (cc-pVxZ) ke atas koordinat inti atom.
3. Evaluasi integral satu-elektron (energi kinetik elektron dan tarikan inti-elektron) serta inisialisasi tebakan awal matriks densitas ($\mathbf{P}_0$).

### 3.3.2 Konvergensi Medan-Rerata (Hartree-Fock)
1. Pembentukan operator Fock tak-terganggu ($\hat{H}_0$) melalui kontraksi matriks densitas dengan integral interaksi.
2. Eksekusi iterasi *Self-Consistent Field* (SCF) untuk merelaksasi orbital.
3. Implementasi pembatasan galat komutator melalui metode ekstrapolasi sub-ruang iteratif (DIIS) untuk menghindari osilasi fungsi gelombang.
4. Iterasi dihentikan saat perubahan matriks densitas antar-siklus ($\Delta \mathbf{P}$) jatuh di bawah ambang batas konvergensi absolut $10^{-8}$ Hartree.

### 3.3.3 Faktorisasi Integral Interaksi Elektron (Dekomposisi Cholesky)
1. Inisialisasi matriks integral tolakan dua-elektron (ERI) pada representasi basis atomik.
2. Pelaksanaan faktorisasi metrik semi-definit positif menggunakan algoritma Dekomposisi Cholesky inkomplit dengan parameter toleransi $\delta$.
3. Transformasi vektor dasar Cholesky ($L_{\mu\nu}^P$) dari ruang basis atomik menuju ruang orbital molekul (basis kanonikal atau teroptimasi).
4. Reduksi tensor 4-indeks menjadi tensor 3-indeks untuk mengeliminasi dimensionalitas $O(N^4)$.

### 3.3.4 Evaluasi Energi Korelasi MP2 dan MP3
1. **Pendekatan Kanonikal:** Eksekusi persamaan deret perturbasi Rayleigh-Schrödinger menggunakan energi orbital Hartree-Fock sebagai penyebut eksitasi statis untuk mengekstrak energi korelasi $E^{(2)}$ dan $E^{(3)}$.
2. **Pendekatan Optimasi Orbital (OMP2/OMP3):** Konstruksi fungsional Lagrangian ($\mathcal{L}$). Matriks anti-Hermitian $\kappa$ divariasikan secara iteratif melalui metode optimasi Newton-Raphson hingga gradien rotasi orbital menyentuh nilai nol stasioner ($\partial \mathcal{L} / \partial \kappa_{ai} < 10^{-6}$).

## 3.4 Analisis Data dan Validasi

Data hasil komputasi (energi elektronik dan waktu eksekusi) diekstraksi dan dievaluasi secara kuantitatif melalui dua parameter utama:

1. **Validitas Fisis dan Numerik:** Evaluasi tingkat presisi dilakukan dengan menghitung galat absolut ($\Delta E$) antara energi korelasi yang dihasilkan instrumen `mshqc` ($E_{mshqc}$) dengan nilai limit referensi analitik ($E_{ref}$). Eksperimen dianggap berhasil dan valid secara matematis apabila galat residual memenuhi kriteria akurasi fisis tingkat tinggi, yaitu $\Delta E < 10^{-6}$ Hartree.
2. **Analisis Penskalaan (Scaling Analysis):** Mengukur pengaruh parameter $\delta$ terhadap rasio jumlah vektor Cholesky ($N_{CD}$) yang terbentuk dibandingkan jumlah basis set absolut ($N^2$). Ekstrapolasi dilakukan untuk membuktikan pergeseran metrik kompleksitas kontraksi tensor dari asimtot teoritis eksponensial $O(N^5)$ menuju bentuk tereduksi berkat implementasi Dekomposisi Cholesky.