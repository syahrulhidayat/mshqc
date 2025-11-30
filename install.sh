#!/bin/bash
################################################################################
# MSHQC Universal Installation Script
# Supports: Ubuntu, Fedora, Arch, macOS, and Windows (MSYS2/WSL)
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
            OS_VERSION=$VERSION_ID
        elif [ -f /etc/fedora-release ]; then
            OS="fedora"
        elif [ -f /etc/debian_version ]; then
            OS="debian"
        else
            OS="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    
    log_info "Detected OS: $OS"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ] && [ "$OS" != "windows" ]; then 
        log_error "Please do not run this script as root!"
        log_info "The script will ask for sudo when needed."
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for required commands
    local required_cmds=("git" "cmake")
    local missing_cmds=()
    
    for cmd in "${required_cmds[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            missing_cmds+=($cmd)
        fi
    done
    
    if [ ${#missing_cmds[@]} -ne 0 ]; then
        log_error "Missing required commands: ${missing_cmds[*]}"
        log_info "These will be installed in the next step."
        return 1
    fi
    
    log_success "All prerequisites found!"
    return 0
}

# Install dependencies based on OS
install_dependencies() {
    log_info "Installing dependencies for $OS..."
    
    case $OS in
        ubuntu|debian|pop)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                wget \
                curl \
                libeigen3-dev \
                libboost-all-dev \
                libgmp-dev \
                autoconf \
                automake \
                libtool \
                python3-dev \
                python3-pip
            ;;
            
        fedora|rhel|centos|rocky)
            # Enable PowerTools/CRB for RHEL/CentOS
            if [[ $OS == "rhel" ]] || [[ $OS == "centos" ]] || [[ $OS == "rocky" ]]; then
                if [[ $OS_VERSION == 8* ]]; then
                    sudo dnf config-manager --set-enabled powertools || true
                elif [[ $OS_VERSION == 9* ]]; then
                    sudo dnf config-manager --set-enabled crb || true
                fi
            fi
            
            sudo dnf install -y \
                gcc-c++ \
                cmake \
                git \
                wget \
                curl \
                eigen3-devel \
                boost-devel \
                gmp-devel \
                autoconf \
                automake \
                libtool \
                python3-devel \
                python3-pip
            ;;
            
        arch|manjaro)
            sudo pacman -S --noconfirm \
                base-devel \
                cmake \
                git \
                wget \
                curl \
                eigen \
                boost \
                gmp \
                autoconf \
                automake \
                libtool \
                python \
                python-pip
            ;;
            
        macos)
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install cmake eigen boost gmp autoconf automake libtool python@3.11
            ;;
            
        windows)
            log_info "Detected Windows environment"
            if command -v pacman &> /dev/null; then
                # MSYS2 environment
                pacman -S --noconfirm \
                    mingw-w64-x86_64-gcc \
                    mingw-w64-x86_64-cmake \
                    mingw-w64-x86_64-eigen3 \
                    mingw-w64-x86_64-boost \
                    mingw-w64-x86_64-gmp \
                    mingw-w64-x86_64-python \
                    mingw-w64-x86_64-python-pip
            else
                log_error "Please install MSYS2 or use WSL2"
                exit 1
            fi
            ;;
            
        *)
            log_error "Unsupported operating system: $OS"
            log_info "Please install dependencies manually."
            exit 1
            ;;
    esac
    
    log_success "Dependencies installed successfully!"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install required packages
    python3 -m pip install --user \
        numpy \
        pybind11 \
        setuptools \
        wheel
    
    log_success "Python dependencies installed!"
}

# Build and install Libint2
install_libint2() {
    log_info "Building and installing Libint2..."
    
    # Check if already installed
    if pkg-config --exists libint2 2>/dev/null; then
        log_success "Libint2 already installed, skipping..."
        return 0
    fi
    
    local LIBINT_DIR="/tmp/libint-$RANDOM"
    mkdir -p "$LIBINT_DIR"
    cd "$LIBINT_DIR"
    
    log_info "Cloning Libint2 repository..."
    git clone --depth 1 https://github.com/evaleev/libint.git .
    
    log_info "Configuring Libint2..."
    ./autogen.sh
    
    mkdir -p build
    cd build
    
    local PREFIX="/usr/local"
    if [ "$OS" == "macos" ]; then
        PREFIX="/usr/local"
    elif [ "$OS" == "windows" ]; then
        PREFIX="/mingw64"
    fi
    
    ../configure \
        --prefix="$PREFIX" \
        --enable-shared \
        --with-max-am=5 \
        --with-opt-am=3 \
        --enable-generic-code \
        --disable-unrolling
    
    log_info "Building Libint2 (this may take a while)..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    
    log_info "Installing Libint2..."
    if [ "$OS" == "windows" ]; then
        make install
    else
        sudo make install
    fi
    
    # Update library cache
    if [ "$OS" != "macos" ] && [ "$OS" != "windows" ]; then
        sudo ldconfig || true
    fi
    
    cd -
    rm -rf "$LIBINT_DIR"
    
    log_success "Libint2 installed successfully!"
}

# Build MSHQC
build_mshqc() {
    log_info "Building MSHQC..."
    
    # Ensure we're in the project directory
    if [ ! -f "CMakeLists.txt" ]; then
        log_error "CMakeLists.txt not found. Please run this script from the MSHQC root directory."
        exit 1
    fi
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure CMake
    log_info "Configuring CMake..."
    
    local CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=/usr/local
    )
    
    # Add OS-specific arguments
    case $OS in
        ubuntu|debian|pop|fedora|rhel|centos|rocky|arch|manjaro)
            CMAKE_ARGS+=(-DEIGEN3_INCLUDE_DIR=/usr/include/eigen3)
            ;;
        macos)
            CMAKE_ARGS+=(-DEIGEN3_INCLUDE_DIR=$(brew --prefix eigen)/include/eigen3)
            ;;
        windows)
            CMAKE_ARGS+=(-DEIGEN3_INCLUDE_DIR=/mingw64/include/eigen3)
            ;;
    esac
    
    cmake .. "${CMAKE_ARGS[@]}"
    
    # Build
    log_info "Compiling MSHQC..."
    local NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    cmake --build . --config Release -j$NPROC
    
    log_success "MSHQC built successfully!"
    
    cd ..
}

# Install MSHQC
install_mshqc() {
    log_info "Installing MSHQC..."
    
    # Install C++ library
    cd build
    
    if [ "$OS" == "windows" ]; then
        cmake --install . --prefix /mingw64
    else
        sudo cmake --install .
    fi
    
    cd ..
    
    # Install Python bindings
    log_info "Installing Python bindings..."
    python3 -m pip install --user -e .
    
    log_success "MSHQC installed successfully!"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd build
    
    if ctest --output-on-failure -C Release; then
        log_success "All tests passed!"
    else
        log_warning "Some tests failed. Check the output above."
    fi
    
    cd ..
    
    # Test Python bindings
    log_info "Testing Python bindings..."
    if python3 -c "import mshqc; print('MSHQC version:', mshqc.__version__)"; then
        log_success "Python bindings work correctly!"
    else
        log_error "Python bindings test failed!"
        return 1
    fi
}

# Print installation summary
print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    log_success "MSHQC Installation Complete!"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Installation details:"
    echo "  OS: $OS"
    echo "  Install prefix: /usr/local"
    echo "  Build directory: $(pwd)/build"
    echo ""
    echo "To use MSHQC:"
    echo ""
    echo "  C++:"
    echo "    #include <mshqc/molecule.h>"
    echo "    g++ -std=c++17 your_code.cpp -lmshqc"
    echo ""
    echo "  Python:"
    echo "    import mshqc"
    echo ""
    echo "Next steps:"
    echo "  • Run examples: cd examples && cmake -B build && cmake --build build"
    echo "  • Read documentation: docs/QUICKSTART.md"
    echo "  • Report issues: https://github.com/syahrulhidayat/mshqc/issues"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

# Main installation workflow
main() {
    echo "═══════════════════════════════════════════════════════════"
    echo "  MSHQC Universal Installation Script"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Detect OS
    detect_os
    
    # Check if running as root
    check_root
    
    # Install dependencies
    log_info "Step 1/6: Installing system dependencies..."
    install_dependencies
    
    log_info "Step 2/6: Installing Python dependencies..."
    install_python_deps
    
    log_info "Step 3/6: Installing Libint2..."
    install_libint2
    
    log_info "Step 4/6: Building MSHQC..."
    build_mshqc
    
    log_info "Step 5/6: Installing MSHQC..."
    install_mshqc
    
    log_info "Step 6/6: Running tests..."
    run_tests || log_warning "T/home/syahrul/mshqc/src/mp3/ump3.ccests completed with warnings"
    
    # Print summary
    print_summary
}

# Parse command line arguments
SKIP_TESTS=false
SKIP_LIBINT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-libint)
            SKIP_LIBINT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-tests     Skip running tests after installation"
            echo "  --skip-libint    Skip Libint2 installation (use system version)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main installation
main
