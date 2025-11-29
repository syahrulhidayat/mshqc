#!/usr/bin/env python3
"""
MSH-QC Installation Diagnostic Script
Checks which methods and features are available
"""

import sys

def check_mshqc_installation():
    """Check MSH-QC installation status"""
    
    print("=" * 60)
    print("MSH-QC Installation Diagnostic")
    print("=" * 60)
    
    # Check Python version
    print(f"\n1. Python Version: {sys.version.split()[0]}")
    
    # Check if mshqc is installed
    try:
        import mshqc
        version = mshqc.__version__ if hasattr(mshqc, '__version__') else 'Unknown'
        print(f"2. MSH-QC Version: {version}")
        print("   Status: ✓ Installed")
    except ImportError as e:
        print("2. MSH-QC Status: ✗ NOT Installed")
        print(f"   Error: {e}")
        return
    
    # Check dependencies
    print("\n3. Checking Dependencies:")
    deps = {
        'numpy': 'NumPy',
        'pybind11': 'pybind11',
    }
    
    for module, name in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'Unknown')
            print(f"   {name}: ✓ {version}")
        except ImportError:
            print(f"   {name}: ✗ NOT installed")
    
    # Check available methods
    print("\n4. Available Methods:")
    
    methods = {
        'SCF Methods': [
            'quick_rhf', 'quick_rohf', 'quick_uhf'
        ],
        'MP Methods': [
            'quick_rmp2', 'quick_ump2', 'quick_omp2',
            'quick_rmp3', 'quick_ump3', 'quick_omp3',
        ],
        'CI Methods': [
            'quick_cis', 'quick_cisd', 'quick_fci'
        ],
        'Multi-Reference': [
            'quick_casscf', 'quick_caspt2'
        ]
    }
    
    total_available = 0
    total_methods = 0
    
    for category, method_list in methods.items():
        print(f"\n   {category}:")
        available = 0
        for method in method_list:
            total_methods += 1
            if hasattr(mshqc, method):
                print(f"      ✓ {method}")
                available += 1
                total_available += 1
            else:
                print(f"      ✗ {method}")
        print(f"      ({available}/{len(method_list)} available)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Methods Available: {total_available}/{total_methods}")
    
    if total_available == total_methods:
        print("\n✓ ALL METHODS AVAILABLE - Full installation!")
    elif total_available > 0:
        print("\n✓ PARTIAL INSTALLATION - Some methods available")
        print("  Install libint2/libcint for more methods")
    else:
        print("\n✗ NO METHODS AVAILABLE - Installation may be incomplete")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    check_mshqc_installation()