#!/bin/bash
# MSH-QC Dependencies Installation Script
# This script helps install required dependencies for MSH-QC

set -e

echo "========================================="
echo "MSH-QC Dependencies Installation Script"
echo "========================================="

# Function to install libint2 from source (MUST be defined BEFORE use)
function install_libint2_from_source() {
    echo "Installing libint2 from source..."
    LIBINT_VERSION="2.9.0"
    
    # Create temp directory
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    # Download
    echo "Downloading libint2 v${LIBINT_VERSION}..."
    if command -v wget >/dev/null 2>&1; then
        wget https://github.com/evaleev/libint/releases/download/v${LIBINT_VERSION}/libint-${LIBINT_VERSION}.tgz || {
            echo "Failed to download libint2"
            return 1
        }
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o libint-${LIBINT_VERSION}.tgz https://github.com/evaleev/libint/releases/download/v${LIBINT_VERSION}/libint-${LIBINT_VERSION}.tgz || {
            echo "Failed to download libint2"
            return 1
        }
    else
        echo "Cannot download libint2: neither wget nor curl available"
        return 1
    fi
    
    # Extract
    echo "Extracting libint2..."
    tar -xzf libint-${LIBINT_VERSION}.tgz
    cd libint-${LIBINT_VERSION}
    
    # Configure and build
    echo "Building libint2 (this may take several minutes)..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    sudo make install
    
    # Update library cache on Linux
    if command -v ldconfig >/dev/null 2>&1; then
        sudo ldconfig
    fi
    
    # Clean up
    cd /
    rm -rf "$TMP_DIR"
    
    echo "✓ libint2 installed to /usr/local"
}

# Function to install libcint (MUST be defined BEFORE use)
function install_libcint() {
    echo "Installing libcint..."
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    # Clone libcint
    if command -v git >/dev/null 2>&1; then
        git clone https://github.com/sunqm/libcint.git
        cd libcint
    else
        echo "Git not available, cannot install libcint"
        return 1
    fi
    
    # Build and install
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    sudo make install
    
    # Update library cache on Linux
    if command -v ldconfig >/dev/null 2>&1; then
        sudo ldconfig
    fi
    
    # Clean up
    cd /
    rm -rf "$TMP_DIR"
    
    echo "✓ libcint installed to /usr/local"
}

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=${VERSION_ID:-"Unknown"}
elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    VER=$(sw_vers -productVersion)
else
    echo "⚠️  Unsupported OS"
    echo "Please install dependencies manually:"
    echo "  - C++ compiler (g++ or clang++)"
    echo "  - CMake >= 3.15"
    echo "  - Eigen3"
    echo "  - libint2"
    echo "  - BLAS/LAPACK"
    exit 1
fi

echo "Detected OS: $OS $VER"
echo ""

# Install dependencies based on OS
if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
    echo "📦 Installing dependencies for Ubuntu/Debian..."
    
    # Update package lists
    echo "Updating package lists..."
    sudo apt-get update
    
    # Install build tools
    echo "Installing build tools..."
    sudo apt-get install -y build-essential cmake git pkg-config
    
    # Install Python development headers
    echo "Installing Python development headers..."
    sudo apt-get install -y python3-dev python3-pip
    
    # Install math libraries
    echo "Installing math libraries..."
    sudo apt-get install -y libeigen3-dev libblas-dev liblapack-dev
    
    # Try to install libint2
    echo "Installing libint2..."
    if apt-cache show libint2-dev >/dev/null 2>&1; then
        sudo apt-get install -y libint2-dev
        echo "✓ libint2 installed from package manager"
    else
        echo "⚠️  libint2-dev not available in repository, building from source..."
        install_libint2_from_source
    fi
    
    # Install OpenMP
    echo "Installing OpenMP..."
    sudo apt-get install -y libomp-dev
    
elif [[ $OS == *"Fedora"* ]] || [[ $OS == *"CentOS"* ]] || [[ $OS == *"Red Hat"* ]]; then
    echo "📦 Installing dependencies for Red Hat/Fedora..."
    
    # Install build tools
    echo "Installing build tools..."
    sudo dnf install -y gcc gcc-c++ cmake git
    
    # Install Python development headers
    echo "Installing Python development headers..."
    sudo dnf install -y python3-devel python3-pip
    
    # Install math libraries
    echo "Installing math libraries..."
    sudo dnf install -y eigen3-devel blas-devel lapack-devel
    
    # Install libint2
    echo "Installing libint2..."
    if dnf list libint2-devel >/dev/null 2>&1; then
        sudo dnf install -y libint2-devel
        echo "✓ libint2 installed from package manager"
    else
        echo "⚠️  libint2-devel not available, building from source..."
        install_libint2_from_source
    fi
    
    # Install OpenMP
    echo "Installing OpenMP..."
    sudo dnf install -y libgomp-devel
    
elif [[ $OS == *"macOS"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing dependencies for macOS..."
    
    # Install Homebrew if not installed
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies
    echo "Installing dependencies via Homebrew..."
    brew update
    brew install cmake eigen libint libomp
    
else
    echo "⚠️  Unsupported OS: $OS"
    echo "Please install dependencies manually:"
    echo "  - C++ compiler"
    echo "  - CMake >= 3.15"
    echo "  - Eigen3"
    echo "  - libint2"
    echo "  - BLAS/LAPACK"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Core dependencies installation completed!"
echo "========================================="
echo ""

# Optional: Install libcint for density fitting
read -p "Do you want to install libcint for density fitting capabilities? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_libcint
fi

# Install Python package
echo ""
echo "========================================="
read -p "Do you want to install MSHQC Python package now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing MSHQC Python package..."
    pip install --upgrade pip setuptools wheel
    pip install pybind11 numpy scipy
    
    # Check if in git repo
    if [ -d ".git" ]; then
        echo "Installing from local repository..."
        pip install -e .
    else
        echo "Installing from GitHub..."
        pip install git+https://github.com/syahrulhidayat/mshqc.git
    fi
    
    echo ""
    echo "✓ MSHQC Python package installed!"
    echo ""
    echo "Test the installation with:"
    echo "  python -c 'import mshqc; print(\"MSHQC successfully installed!\")'"
fi

echo ""
echo "========================================="
echo "✓ All dependencies are now installed!"
echo "========================================="
echo ""
echo "Next steps:"
if [ -d ".git" ]; then
    echo "  1. Build C++ library (optional):"
    echo "     mkdir -p build && cd build"
    echo "     cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "     make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    echo ""
fi
echo "  2. Start using MSHQC:"
echo "     python"
echo "     >>> import mshqc"
echo "     >>> # Your quantum chemistry code here"
echo ""
echo "For documentation, visit: https://github.com/syahrulhidayat/mshqc"
echo "========================================="
