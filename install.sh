#!/bin/bash

# MSH-QC Dependencies Installation Script
# This script helps install required dependencies for MSH-QC

set -e

echo "========================================="
echo "MSH-QC Dependencies Installation Script"
echo "========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=${VERSION_ID}
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
else
    echo "Unsupported OS"
    exit 1
fi

echo "Detected OS: $OS $VER"

# Install dependencies based on OS
if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
    echo "Installing dependencies for Ubuntu/Debian..."
    
    # Update package lists
    sudo apt-get update
    
    # Install build tools
    sudo apt-get install -y build-essential cmake git
    
    # Install math libraries
    sudo apt-get install -y libeigen3-dev libblas-dev liblapack-dev
    
    # Try to install libint2
    echo "Installing libint2..."
    if apt-cache show libint2-dev >/dev/null 2>&1; then
        sudo apt-get install -y libint2-dev
    else
        echo "libint2-dev not available, will build from source..."
        install_libint2_from_source
    fi
    
    # Install OpenMP
    sudo apt-get install -y libomp-dev
    
elif [[ $OS == *"Fedora"* ]] || [[ $OS == *"CentOS"* ]] || [[ $OS == *"Red Hat"* ]]; then
    echo "Installing dependencies for Red Hat/Fedora..."
    
    # Install build tools
    sudo dnf install -y gcc gcc-c++ cmake git
    
    # Install math libraries
    sudo dnf install -y eigen3-devel blas-devel lapack-devel
    
    # Install libint2
    if dnf list libint2-devel >/dev/null 2>&1; then
        sudo dnf install -y libint2-devel
    else
        echo "libint2-devel not available, will build from source..."
        install_libint2_from_source
    fi
    
    # Install OpenMP
    sudo dnf install -y libgomp-devel
    
elif [[ $OS == *"macOS"* ]] || [[ $OS == *"Darwin"* ]]; then
    echo "Installing dependencies for macOS..."
    
    # Install Homebrew if not installed
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew not found. Installing..."
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi
    
    # Install dependencies
    brew update
    brew install cmake eigen libint2 libomp
    
else
    echo "Unsupported OS. Please install dependencies manually."
    echo "Required: C++ compiler, CMake, Eigen3, libint2, BLAS, LAPACK"
    exit 1
fi

echo "========================================="
echo "Dependencies installation completed!"
echo "========================================="

function install_libint2_from_source() {
    echo "Installing libint2 from source..."
    LIBINT_VERSION="2.9.0"
    
    # Create temp directory
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    # Download
    if command -v wget >/dev/null 2>&1; then
        wget https://github.com/evaleevasquez/libint/releases/download/v${LIBINT_VERSION}/libint-${LIBINT_VERSION}.tgz
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o libint-${LIBINT_VERSION}.tgz https://github.com/evaleevasquez/libint/releases/download/v${LIBINT_VERSION}/libint-${LIBINT_VERSION}.tgz
    else
        echo "Cannot download libint2: neither wget nor curl available"
        return 1
    fi
    
    # Extract
    tar -xzf libint-${LIBINT_VERSION}.tgz
    cd libint-${LIBINT_VERSION}
    
    # Configure and build
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    sudo make install
    
    # Clean up
    cd /home/
    rm -rf "$TMP_DIR"
    
    echo "libint2 installed to /usr/local"
}

# Optional: Install libcint for density fitting
read -p "Do you want to install libcint for density fitting capabilities? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
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
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    sudo make install
    
    # Clean up
    cd /home/
    rm -rf "$TMP_DIR"
    
    echo "libcint installed to /usr/local"
fi

echo "========================================="
echo "All dependencies are now installed!"
echo "You can now build MSH-QC by running:"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j$(nproc)"
echo "========================================="