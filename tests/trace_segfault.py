import os
import mshqc
import sys

def trace_memory_allocation():
    print("[TRACE] 1. Mengalokasikan raw_mol (mshqc.Molecule)...")
    raw_mol = mshqc.Molecule()
    
    print("[TRACE] 2. Memuat data atomik ke dalam memori...")
    raw_mol.add_atom(8, 0.0, 0.0, 0.119262)
    raw_mol.add_atom(1, 0.0, 0.763239, -0.477047)
    raw_mol.add_atom(1, 0.0, -0.763239, -0.477047)
    
    print("[TRACE] 3. Mengalokasikan mesin PointGroup (mshqc.PointGroup)...")
    pg = mshqc.PointGroup(raw_mol, 1e-6)
    
    print("[TRACE] 4. Menjalankan rutinitas pg.detect()...")
    pg.detect()
    
    print("[TRACE] 5. Mengeksekusi ekstraksi pg.get_aligned_molecule()...")
    aligned_mol = pg.get_aligned_molecule()
    
    print("[TRACE] 6. Menginisialisasi BasisSet dan membaca I/O (sto-3g)...")
    basis = mshqc.BasisSet("sto-3g", aligned_mol, "data/basis")
    
    print("[TRACE] 7. Mengalokasikan PetiteList...")
    pl = mshqc.PetiteList(basis, pg)
    
    print("[TRACE] 8. Menjalankan rutinitas pl.build()...")
    pl.build()
    
    print("[TRACE] 9. Membangun IntegralEngine...")
    integrals = mshqc.IntegralEngine(aligned_mol, basis)
    
    print("[TRACE] 10. SELURUH INFRASTRUKTUR MEMORI HPC BERHASIL DIALOKASIKAN.")

if __name__ == "__main__":
    trace_memory_allocation()
    os._exit(0)  