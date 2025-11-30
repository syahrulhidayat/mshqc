#!/bin/bash
# install_dependencies.sh

echo "=== Installing MSHQC System Dependencies ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Detected Debian/Ubuntu system"
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            libeigen3-dev \
            libint2-dev \
            libblas-dev \
            liblapack-dev \
            pkg-config
    elif command -v yum &> /dev/null; then
        # RedHat/CentOS
        echo "Detected RedHat/CentOS system"
        sudo yum install -y \
            gcc-c++ \
            cmake \
            eigen3-devel \
            blas-devel \
            lapack-devel
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS system"
    brew install eigen libint cmake
fi

echo "=== System dependencies installed ==="
