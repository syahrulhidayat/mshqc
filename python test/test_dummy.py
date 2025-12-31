# File: /home/syahrul/mshqc/test_final.py
import os
import sys

# Pastikan path terbaca (opsional jika sudah export di terminal)
sys.path.append("/home/syahrul/mshqc/python")

from mshqc.utils import quick_calculation, benchmark_basis_sets

def main():
    print("=== MSHQC Final Integration Test ===")
    
    # 1. Tes Quick Calculation (Menguji perbaikan Type-Safe di calculators.py)
    # Kita sengaja tidak pass 'basis_dir' untuk menguji fallback ke default (None -> default)
    print("\n[Test 1] Running Quick Calculation (He / STO-3G)...")
    try:
        # Kita pakai basis yang sangat kecil biar cepat
        res = quick_calculation("He", basis="sto-3g", method="rhf")
        
        # Menguji akses atribut energi
        e = getattr(res, 'energy', getattr(res, 'e_total', 'N/A'))
        print(f"✅ Calculation Success! Energy: {e} Ha")
    except Exception as e:
        print(f"❌ Calculation Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Tes Benchmark (Menguji perbaikan getattr di utils.py)
    print("\n[Test 2] Running Benchmark utils...")
    try:
        # Benchmark dummy
        results = benchmark_basis_sets("H", ["sto-3g"], method="rhf")
        if results["sto-3g"]["success"]:
            print("✅ Benchmark Success!")
        else:
            print("❌ Benchmark Failed inside function.")
    except Exception as e:
        print(f"❌ Benchmark Script Error: {e}")

if __name__ == "__main__":
    main()